# Footnote

RAG over SEC filings, evaluated on the FinanceBench open-source set (150 questions, 84 filings). The eval harness is the core of the project. The chatbot comes second.

> Status: Phase 1 (ingest). Results tables will appear here once the eval harness lands. No numbers are shown until they are measured.

## Data and license

- Questions: [PatronusAI/financebench](https://huggingface.co/datasets/PatronusAI/financebench), licensed **CC-BY-NC-4.0**. **This project and anything derived from the dataset are for non-commercial use only.**
- PDFs: [patronus-ai/financebench](https://github.com/patronus-ai/financebench) `/pdfs`. Only the 84 docs the questions reference are downloaded, and they are never committed.
- Gold `evidence_page_num` is **0-indexed**. All 189 evidence items match the PDF page at that index. Chunks store 1-based page numbers.

## Ingest (Phase 1)

- Pages are parsed with pymupdf. Tables come from two sources:
  - A layout detector rebuilds rows from word boxes and anchors columns on the right edges of numbers. This handles the rule-less financial statements that pdfplumber's line finder misses.
  - pdfplumber covers ruled tables that the layout pass didn't catch.
- Tables are serialized as markdown and are never split mid-row.
  - A table that fits in the room left in the current chunk is packed in whole. Otherwise it starts a new chunk together with its caption lines.
  - A table larger than `table_max` is split on row boundaries, and the header repeats in every piece.
  - A row's values map to columns in reading order (a small DP), so values that sit close together don't merge or swap.
- Embeddings come from `BAAI/bge-small-en-v1.5` (384-dim), run locally. Each chunk is embedded with a short header (company, doc type, FY, section) and cached by content hash in `cache/embeddings.sqlite`.
- The store is Postgres + pgvector in the `footnote` schema, one table per chunk-size config. Each table has an HNSW index (cosine) and a GIN index on a generated `tsvector`.
- Re-runs are idempotent:
  - `chunk_id` is `doc::p<page>::<k>`.
  - The upsert compares a hash of every stored field.
  - Stale chunks and dropped docs are deleted.
  - `VACUUM` runs every 8 docs, and a 450 MB guard aborts the run, so a re-ingest can't push past the free tier's 500 MB cap.

### Ingest report (measured 2026-11-22: RTX 3060, 10 parse workers, Supabase free tier via session pooler)

| | 256-token | 1024-token |
|---|---|---|
| Docs / pages | 84 / 12,013 | 84 / 12,013 |
| Chunks | 42,137 | 11,719 |
| Chunks containing a table | 9,586 | 4,957 |
| Mean tokens / chunk | 228 | 808 |
| Chunks truncated by the embedder (>512 tokens incl. header) | 1 | 11,458 (98%) |
| Cold parse + chunk | 289 s | 290 s |
| Cold embed (GPU) | 235 s | 87 s |
| DB upsert | 231 s (1k changed rows) / 583 s (all 42k rows) | not loaded yet |
| Table size in Postgres | 314 MB | — |

- 0 of 84 docs failed to download or parse, so 0 of 150 questions are skipped.
- A warm re-run (both caches hit) of parse + chunk + embed takes about 25 s. Only the DB writes cost time.
- bge-small reads only 512 tokens, so 1024-token chunks are embedded on their first ~half only. Experiment 1 measures what that costs.
- The 1024 table is not loaded yet. The DB is at 346 MB with 256 loaded, and both tables together would pass the 400 MB budget. So 1024 gets loaded after the 256 retrieval eval is saved and that table is dropped.

### Ingest audit (`python -m scripts.audit_ingest 256`, `python -m scripts.check_db`)

| Check | 256 | 1024 |
|---|---|---|
| Gold evidence pages that have chunks | 189 / 189 | 189 / 189 |
| Multi-digit numbers in gold evidence text found in that page's chunks | 8,172 / 8,188* | 8,172 / 8,188* |
| Words on each PDF page found in that page's chunks | 99.82% | 99.83% |
| Table rows not found whole in a single chunk | 0 / 73,228 | 0 / 73,228 |
| Text chunks over chunk_size | 0 | 0 |
| DB vs local: ids, row hashes, sampled embeddings | identical | — |

\* All 16 misses are dates. The gold evidence text writes them without a space ("31,2016"), while the chunks keep the PDF's "31, 2016". I checked by hand that they're present.

**Known parser limitations (measured, not fixed):**
- About 1% of table cells (2,487 of 239k) contain two numbers. A random sample of 12 showed most are legitimate cells: dates, note names like "2.500% Notes due 2022", footnote markers. About 2 in 12 were real overflows, where a row has more values than detected columns.
- 76 chunks (0.2%) are a single short line, usually a section title right before a page break.

## Setup

```bash
# Python 3.11
python -m venv .venv && .venv/Scripts/activate        # or: uv venv --python 3.11
pip install -r requirements.txt                        # CUDA: install torch from the cu124 index first
cp .env.example .env                                   # fill SUPABASE_DB_URL, GROQ_API_KEY
python -m scripts.download                             # questions + 84 PDFs -> data/
python -m footnote.ingest --config configs/ingest_256.yaml
python -m footnote.ingest --config configs/ingest_1024.yaml
python -m scripts.audit_ingest 256
python -m scripts.check_db
pytest
```
