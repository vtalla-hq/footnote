"""FinanceBench open-source questions (HF: PatronusAI/financebench, CC-BY-NC-4.0)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
QUESTIONS_PATH = DATA_DIR / "financebench_questions.jsonl"
DOWNLOAD_REPORT = DATA_DIR / "download_report.json"

HF_REPO = "PatronusAI/financebench"


def fetch_questions() -> list[dict]:
    """Download the HF dataset once and cache it as jsonl."""
    if QUESTIONS_PATH.exists():
        return load_questions()
    import pandas as pd
    from huggingface_hub import snapshot_download

    local = Path(snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=["*.parquet", "*.jsonl", "*.json"]))
    files = sorted(local.rglob("*.parquet")) or sorted(local.rglob("*.jsonl"))
    if not files:
        raise RuntimeError(f"no data files found in {local}")
    f = files[0]
    df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_json(f, lines=True)
    rows = json.loads(df.to_json(orient="records"))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with QUESTIONS_PATH.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return rows


def load_questions() -> list[dict]:
    with QUESTIONS_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def referenced_docs(questions: list[dict]) -> list[str]:
    docs = set()
    for q in questions:
        docs.add(q["doc_name"])
        for ev in q.get("evidence") or []:
            if ev.get("doc_name"):
                docs.add(ev["doc_name"])
    return sorted(docs)


def doc_metadata(questions: list[dict]) -> dict[str, dict]:
    """doc_name -> {company, doc_type, fiscal_year} taken from the dataset itself."""
    meta: dict[str, dict] = {}
    for q in questions:
        meta.setdefault(
            q["doc_name"],
            {
                "company": q.get("company"),
                "doc_type": normalize_doc_type(q.get("doc_type"), q["doc_name"]),
                "fiscal_year": _int_or_none(q.get("doc_period")),
            },
        )
    return meta


def normalize_doc_type(raw: str | None, doc_name: str) -> str:
    s = (raw or doc_name).lower().replace("-", "")
    if "10k" in s:
        return "10-K"
    if "10q" in s:
        return "10-Q"
    if "8k" in s:
        return "8-K"
    if "earning" in s or "transcript" in s:
        return "earnings"  # FinanceBench "Earnings" docs are earnings releases, not call transcripts
    return raw or "unknown"


def _int_or_none(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
