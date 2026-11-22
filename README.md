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
- Tables are serialized as markdown and are never split mid-row. A table larger than `table_max` is split on row boundaries, and the header repeats in every piece.
- Embeddings come from `BAAI/bge-small-en-v1.5` (384-dim), run locally and cached by content hash in `cache/embeddings.sqlite`.
- The store is Postgres + pgvector in the `footnote` schema, one table per chunk-size config. Each table has an HNSW index (cosine) and a GIN index on a generated `tsvector`. Upserts go on `chunk_id`, and stale chunks are deleted, so re-runs are idempotent.

### Ingest report (measured 2026-11-22: RTX 3060, 10 parse workers, Supabase free tier via session pooler)

| | 256-token | 1024-token |
|---|---|---|
| Docs / pages | 84 / 12,013 | 84 / 12,013 |
| Chunks | 42,093 | 11,676 |
| Chunks containing a table | 8,993 | 4,891 |
| Mean tokens / chunk | 227 | 806 |
| Chunks > 512 tokens (bge truncates) | 294 (0.7%) | 11,380 (97%) |
| Cold parse + chunk | 291 s | 290 s |
| Cold embed (GPU) | 250 s | 86 s |
| DB upsert (42k rows, update path) | 582 s | not loaded yet |
| Table size in Postgres | 273 MB | — |

- 0 of 84 docs failed to download or parse, so 0 of 150 questions are skipped.
- A warm re-run (both caches hit) of parse + chunk + embed takes about 25 s. Only the DB writes cost time.
- The 1024 table is not loaded yet. With 256 loaded the DB is 303 MB. Loading 1024 alongside it would pass the 400 MB budget, and re-upserting in place briefly doubled the table, peaking at 488 MB against the 500 MB free cap. So 1024 gets loaded after the 256 retrieval eval is saved and that table is dropped.

## Setup

```bash
# Python 3.11
python -m venv .venv && .venv/Scripts/activate        # or: uv venv --python 3.11
pip install -r requirements.txt                        # CUDA: install torch from the cu124 index first
cp .env.example .env                                   # fill SUPABASE_DB_URL, GROQ_API_KEY
python -m scripts.download                             # questions + 84 PDFs -> data/
python -m footnote.ingest --config configs/ingest_256.yaml
python -m footnote.ingest --config configs/ingest_1024.yaml
pytest
```
