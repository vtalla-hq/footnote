"""Fetch FinanceBench questions + only the PDFs they reference.

    python -m scripts.download

Broken/missing docs are logged to data/download_report.json and skipped downstream.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import fitz  # pymupdf
import requests

from footnote.dataset import DOWNLOAD_REPORT, PDF_DIR, fetch_questions, referenced_docs

RAW = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{}.pdf"
MEDIA = "https://media.githubusercontent.com/media/patronus-ai/financebench/main/pdfs/{}.pdf"  # git-lfs fallback


def check_pdf(path) -> str | None:
    """Return an error string if the file is not a readable PDF with pages."""
    try:
        with fitz.open(path) as doc:
            if doc.page_count == 0:
                return "zero pages"
            doc[0].get_text()
    except Exception as e:  # noqa: BLE001
        return f"unreadable: {e}"
    return None


def fetch(doc_name: str) -> tuple[str, str | None]:
    dest = PDF_DIR / f"{doc_name}.pdf"
    if dest.exists() and check_pdf(dest) is None:
        return doc_name, None
    last_err = None
    for tmpl in (RAW, MEDIA):
        try:
            r = requests.get(tmpl.format(quote(doc_name)), timeout=120)
        except requests.RequestException as e:
            last_err = f"request failed: {e}"
            continue
        if r.status_code != 200:
            last_err = f"HTTP {r.status_code}"
            continue
        if not r.content.startswith(b"%PDF"):
            last_err = "not a PDF (maybe LFS pointer)"
            continue
        dest.write_bytes(r.content)
        err = check_pdf(dest)
        if err is None:
            return doc_name, None
        last_err = err
        dest.unlink(missing_ok=True)
    return doc_name, last_err


def main() -> int:
    qs = fetch_questions()
    docs = referenced_docs(qs)
    print(f"{len(qs)} questions reference {len(docs)} docs")
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(fetch, docs))
    failed = {d: e for d, e in results if e}
    ok = [d for d, e in results if not e]
    affected = [q["financebench_id"] for q in qs if q["doc_name"] in failed]
    report = {"docs_referenced": len(docs), "docs_ok": len(ok), "docs_failed": failed,
              "questions_total": len(qs), "questions_skipped": len(affected),
              "skipped_question_ids": affected}
    DOWNLOAD_REPORT.write_text(json.dumps(report, indent=2))
    print(f"ok: {len(ok)}  failed: {len(failed)}  questions skipped: {len(affected)}")
    for d, e in failed.items():
        print(f"  FAILED {d}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
