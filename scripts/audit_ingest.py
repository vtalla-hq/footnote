"""Ingest audit: gold-page coverage, page text recall, table-row integrity, chunk sanity.

    python -m scripts.audit_ingest 256      # or 1024
"""
import collections
import json
import re
import sys


import fitz

from footnote.dataset import PDF_DIR, doc_metadata, load_questions, referenced_docs
from footnote.ingest import _parse_and_chunk
from footnote.parse import parse_pdf
from footnote.chunking import split_table

size = int(sys.argv[1]) if len(sys.argv) > 1 else 256
overlap, table_max = (32, 416) if size == 256 else (128, 1024)
qs = load_questions()
meta = doc_metadata(qs)
docs = referenced_docs(qs)

chunks_by_doc = {}
for d in docs:
    _, _, rows, _ = _parse_and_chunk((d, meta[d], size, overlap, table_max, False))
    chunks_by_doc[d] = rows

NUM = re.compile(r"\d[\d,]*\.?\d*")
WORD = re.compile(r"[A-Za-z]{3,}|\d[\d,.]*\d|\d")

# 1) gold evidence coverage
cov = collections.Counter()
missing_nums = []
for q in qs:
    for ev in q["evidence"]:
        d, p = ev["doc_name"], ev["evidence_page_num"] + 1
        on_page = [r for r in chunks_by_doc[d] if p in r["pages"]]
        cov["evidence"] += 1
        cov["page_has_chunk"] += bool(on_page)
        text = "\n".join(r["text"] for r in on_page)
        nums = [n.rstrip(".,") for n in NUM.findall(ev["evidence_text"]) if len(n.rstrip(".,").replace(",", "")) >= 3]
        cov["nums_total"] += len(nums)
        got = [n for n in nums if n in text]
        cov["nums_found"] += len(got)
        if len(got) < len(nums):
            missing_nums.append((q["financebench_id"], d, p, [n for n in nums if n not in text][:5]))
print(f"[{size}] evidence items {cov['evidence']}, gold page has >=1 chunk: {cov['page_has_chunk']}")
print(f"[{size}] numbers (>=3 digits) in evidence_text found on gold-page chunks: {cov['nums_found']}/{cov['nums_total']}")
for m in missing_nums[:12]:
    print("   missing", m)

# 2) word recall per page: are words on the PDF page present in that page's chunks?
low = []
tot_w = tot_hit = 0
for d in docs:
    with fitz.open(PDF_DIR / f"{d}.pdf") as doc:
        by_page = collections.defaultdict(str)
        for r in chunks_by_doc[d]:
            for p in r["pages"]:
                by_page[p] += "\n" + r["text"]
        for i, page in enumerate(doc):
            words = WORD.findall(page.get_text())
            if len(words) < 20:
                continue
            ctext = by_page.get(i + 1, "")
            cw = collections.Counter(WORD.findall(ctext))
            pw = collections.Counter(words)
            hit = sum(min(c, cw[w]) for w, c in pw.items())
            tot_w += len(words); tot_hit += hit
            rec = hit / len(words)
            if rec < 0.9:
                low.append((round(rec, 2), d, i + 1, len(words)))
print(f"[{size}] word recall page->chunks: {tot_hit / tot_w:.4f}; pages with recall<0.9: {len(low)}")
for l in sorted(low)[:10]:
    print("   low", l)

# 3) real table integrity: every row of every parsed table sits whole inside one chunk
viol = rows_total = tables_total = 0
for d in docs:
    els = parse_pdf(PDF_DIR / f"{d}.pdf")
    texts = [r["text"] for r in chunks_by_doc[d]]
    for el in els:
        if el.kind != "table":
            continue
        tables_total += 1
        for ln in el.text.splitlines()[2:]:
            if not ln.strip():
                continue
            rows_total += 1
            if not any(ln in t for t in texts):
                viol += 1
print(f"[{size}] tables {tables_total}, rows {rows_total}, rows not found whole in any chunk: {viol}")

# 4) reviewer symptoms
NUMTOK = re.compile(r"^[($]*-?\$?\d[\d,]*(\.\d+)?%?\)?$")
multi = cells = 0
max_head = 0
for d in docs:
    for el in parse_pdf(PDF_DIR / f"{d}.pdf"):
        if el.kind == "heading":
            max_head = max(max_head, len(el.text))
        if el.kind != "table":
            continue
        for ln in el.text.splitlines()[2:]:
            for c in ln.strip("|").split(" | ")[1:]:
                toks = c.split()
                if not toks:
                    continue
                cells += 1
                multi += sum(bool(NUMTOK.match(t)) for t in toks) >= 2
allc = [r for d in docs for r in chunks_by_doc[d]]
text_over = sum(r["chunk_type"] == "text" and r["n_tokens"] > size for r in allc)
text_over_slack = sum(r["chunk_type"] == "text" and r["n_tokens"] > size + 40 for r in allc)
lone_head = sum(r["n_tokens"] < 12 and "\n" not in r["text"] for r in allc)
trunc = sum(r["embed_tokens"] > 512 for r in allc)
print(f"[{size}] table cells with >=2 numbers: {multi}/{cells}; longest heading: {max_head} chars")
print(f"[{size}] chunks {len(allc)}; text chunks > chunk_size: {text_over} (> +40 tokens: {text_over_slack}); "
      f"tiny single-line chunks: {lone_head}; truncated by embedder (>512 incl. header): {trunc}")
print(f"[{size}] distinct chunk_ids: {len({r['chunk_id'] for r in allc})}")
