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

## Retrieval eval (Phase 2)

`python -m eval.run --config configs/<name>.yaml` writes `results/<timestamp>_<config>.json`. The file holds the config and its hash, the git sha, per-question retrieved chunks with ranks, aggregates, breakdowns and latency. To diff runs, use `python -m eval.compare results/A.json results/B.json --by question_type`.

**Metrics.** Gold for a question is its set of (doc, page) evidence pairs. A chunk hits a gold pair when the doc matches and the chunk's pages include that page. Metrics count gold pages covered, so two chunks on one page don't double count.
- **recall@k:** the fraction of gold pages covered by the top k chunks.
- **hit@10:** whether any gold page is covered in the top 10.
- **MRR:** 1 / rank of the first hit.
- **nDCG@10:** binary gain, with ideal = min(|gold|, 10).
- **doc_hit@10:** right doc anywhere in the top 10, page ignored. Diagnostic only.

The definitions are hand-checked in `tests/test_metrics.py`. Generation metrics are kept separate (Phase 4).

**Latency.** `total` is end-to-end from this client: query embedding, network and DB. `db_exec` is server-side execution from `EXPLAIN ANALYZE`. The DB lives in us-east-1, about 35 ms away, and an identical re-run moved total p50 from 38 to 46 ms. So config comparisons use `db_exec` plus local compute, not wall time.

### First run: dense only, 256-token chunks, top 10 (`configs/dense_256.yaml`, 150 questions, 0 skipped)

| recall@1 | recall@5 | recall@10 | hit@10 | MRR | nDCG@10 | doc_hit@10 | DB exec p50 / p95 | total p50 / p95 |
|---|---|---|---|---|---|---|---|---|
| 0.197 | 0.393 | 0.542 | 0.613 | 0.315 | 0.355 | 0.953 | 1.2 / 1.8 ms | 38 / 92 ms |

| question_type | n | recall@10 | MRR | | doc_type | n | recall@10 | MRR |
|---|---|---|---|---|---|---|---|---|
| domain-relevant | 50 | 0.447 | 0.239 | | 10-K | 112 | 0.512 | 0.284 |
| metrics-generated | 50 | 0.560 | 0.313 | | 10-Q | 15 | 0.600 | 0.396 |
| novel-generated | 50 | 0.620 | 0.392 | | 8-K | 9 | 0.889 | 0.553 |
| | | | | | earnings | 14 | 0.500 | 0.317 |

The right filing is in the top 10 for 95% of questions, but the right page for only 61%. Of the 58 questions with no page hit, 33 land more than 5 pages from the gold page in the right doc, 10 land 2–5 pages away, 8 land 1 page away, and 7 never retrieve the right doc. That points to within-document ranking (hybrid and reranking) as the next lever, not doc routing. By reasoning type, information-extraction questions get recall@10 0.726 and numerical-reasoning questions 0.465.

## Retrieval experiments (Phase 3)

The full table, CIs, ingest cost and the recommendation are in [results/SUMMARY.md](results/SUMMARY.md). The headline rows:

| config | recall@10 | Δ vs dense_256 [95% CI] | MRR | compute p50 |
|---|---|---|---|---|
| dense, 256-token | 0.542 | — | 0.315 | 13 ms |
| dense, 1024-token | 0.493 | −0.049 [−0.120, +0.021] | 0.288 | 19 ms |
| hybrid (Postgres FTS + dense, RRF), 256 | 0.407 | −0.136 [−0.192, −0.083] * | 0.214 | 945 ms |
| hybrid + bge-reranker top-50, 256 | 0.570 | +0.028 [−0.031, +0.086] | 0.342 | 2.0 s (GPU) |
| **dense + bge-reranker top-50, 256** | **0.599** | +0.057 [−0.010, +0.120] | 0.339 | 0.93 s (GPU) / ~10.5 s (CPU) |

`*` means the 95% paired-bootstrap CI excludes 0. Hybrid hurts and reranking helps, but with 150 questions only the hybrid drop is significant.

## CI regression gate (Phase 5)

`.github/workflows/retrieval-gate.yml` runs on every PR. It's retrieval only (no LLM calls, $0):

1. Installs CPU torch and runs `pytest`.
2. `python -m scripts.ci_gate` **re-ingests a fixed subset with the PR's code**: 19 filings, 999 pages, covering 53 of the 150 questions across all four doc types (`configs/ci.yaml`). The data goes into a throwaway table `footnote.ci_<run id>`.
3. That table has no HNSW index, so search is exact and deterministic. The gate runs the retrieval eval on those 53 questions and drops the table afterwards.
4. It writes a markdown diff vs `results/baseline.json` to the job summary and **fails if recall@10 or MRR drops by more than 1 point.** MRR was added after recall@10 alone missed both test breaks (see "What didn't work").

The gate re-ingests rather than querying the existing table because a PR that breaks parsing or chunking would otherwise go unnoticed. The baseline is produced the same way on CPU (`python -m scripts.ci_gate --write-baseline` with `CUDA_VISIBLE_DEVICES=""`). It scores recall@10 0.660, and an unchanged re-run reproduces every metric exactly. The DB URL comes from the `SUPABASE_DB_URL` repo secret.

## What didn't work

- **recall@10 alone was blind to chunking damage, and shredding chunks even raised it.** The CI gate was first specified as "fail if recall@10 drops > 1 point". Two deliberate breaks on the 53-question CI subset both passed it:

  | break | recall@10 | MRR | recall@1 |
  |---|---|---|---|
  | baseline | 0.660 | 0.398 | 0.264 |
  | layout table detection off (statements become prose lines) | **0.670** (+0.009) | 0.376 (−0.022) | 0.245 |
  | chunk size 64 instead of 256 | **0.679** (+0.019) | 0.358 (−0.040) | 0.226 |

  A page-level hit only needs *some* chunk from the gold page in the top 10. Tiny or shredded chunks are page-local, so ten of them hit more distinct pages, even though each is a worse unit of evidence. The rank of the first hit (MRR) does degrade, so the gate now also fails on a > 1-point MRR drop. Exact search makes that threshold noise-free. A better fix is on the v2 list.
- **Postgres full-text hybrid made retrieval worse.** It cost 10–17 recall@10 points with every keyword scorer tried, including real BM25 computed in SQL. FinanceBench's instruction boilerplate ("…by relying on the details shown in…") is rare in filings, so IDF rewards exactly the wrong words. Details are in [results/SUMMARY.md](results/SUMMARY.md).
- **Page hits overstate answerability for split tables.** Large statements are split into row pieces that share a caption and look alike to the embedder. For 3M's FY2018 capex, retrieval returned the financing slice of the cash-flow statement. That counts as a hit on page 60, but the capex row was in a different piece, and the generator correctly refused.
- **pdfplumber missed most financial statements.** SEC statements have no ruling lines, so a layout detector (word positions plus right-aligned number columns) does the work, with pdfplumber only for ruled tables.
- **The reranker can't be served for free.** bge-reranker-base takes ~0.9 s/query on a GPU but ~10.5 s on CPU, so the demo serves dense top-5 only.

## v2 stretch goals

- **Token-budget recall:** recall over a fixed context budget (e.g. the first ~2,500 tokens of results) instead of top-k. It removes the small-chunk advantage at the metric level.
- **Strip question boilerplate before keyword search,** with the stripping rule fixed on a held-out split, not on the eval questions.
- **A larger embedding window,** so 1024-token chunks aren't truncated at 512.

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
