# Phase 3: retrieval experiments

FinanceBench open-source, 150 questions, 0 skipped. Every number below comes from a saved run in `results/`. Regenerate the table with `python -m scripts.summary`. Measured 2026-11-22 on an RTX 3060, with Supabase free tier in us-east-1 reached through the session pooler.

**Metric:** a hit is the right doc AND an overlapping page. recall@k is the fraction of gold evidence pages covered by the top k chunks (see the README's metric section). Δ is the paired per-question change vs `dense_256` with a 95% bootstrap CI (10k resamples). `*` means the CI excludes 0.

**Latency:**
- **compute:** query embedding + server-side DB execution (`EXPLAIN ANALYZE`) + reranker. This is the cost of the method itself.
- **e2e:** wall time from this client. It adds 20–60 ms of network per round trip and moved about ±20% between identical runs.

## Results

| exp | config | chunks | retrieval | recall@1 | recall@5 | recall@10 | Δ recall@10 vs dense_256 [95% CI] | MRR | nDCG@10 | compute p50 / p95 ms | e2e p50 / p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `dense_256` | 256 | dense | 0.197 | 0.393 | **0.542** | — | 0.315 | 0.355 | 13 / 30 | 38 / 92 |
| 1 | `dense_1024` | 1024 | dense | 0.183 | 0.370 | **0.493** | -0.049 [-0.120, +0.021] | 0.288 | 0.323 | 19 / 26 | 45 / 57 |
| 2 | `hybrid_256` | 256 | hybrid + ts_rank | 0.107 | 0.300 | **0.407** | -0.136 [-0.192, -0.083] * | 0.214 | 0.253 | 945 / 3297 | 1212 / 4277 |
| 2 | `hybrid_bm25_256` | 256 | hybrid + BM25 | 0.127 | 0.340 | **0.437** | -0.106 [-0.163, -0.051] * | 0.234 | 0.276 | 1402 / 2703 | 1625 / 3106 |
| 2 | `hybrid_1024` | 1024 | hybrid + ts_rank | 0.090 | 0.263 | **0.380** | -0.162 [-0.232, -0.093] * | 0.185 | 0.223 | 452 / 1362 | 599 / 1539 |
| 2 | `hybrid_idf_1024` | 1024 | hybrid + ts_rank+idf-terms | 0.093 | 0.263 | **0.377** | -0.166 [-0.236, -0.096] * | 0.188 | 0.224 | 43 / 90 | 81 / 149 |
| 2 | `hybrid_bm25_1024` | 1024 | hybrid + BM25 | 0.157 | 0.313 | **0.420** | -0.122 [-0.192, -0.052] * | 0.247 | 0.279 | 1154 / 1436 | 1092 / 1398 |
| 3 | `hybrid_rerank_256` | 256 | hybrid + ts_rank + rerank top-50 | 0.210 | 0.447 | **0.570** | +0.028 [-0.031, +0.086] | 0.342 | 0.386 | 2019 / 5298 | 2487 / 5913 |
| 3 | `hybrid_bm25_rerank_256` | 256 | hybrid + BM25 + rerank top-50 | 0.203 | 0.443 | **0.577** | +0.034 [-0.031, +0.098] | 0.332 | 0.382 | 2551 / 4161 | 2909 / 5505 |
| 3 | `hybrid_rerank_1024` | 1024 | hybrid + ts_rank + rerank top-50 | 0.147 | 0.417 | **0.547** | +0.004 [-0.073, +0.083] | 0.284 | 0.339 | 1396 / 2384 | 1677 / 2658 |
| 3+ | `dense_rerank_256` | 256 | dense + rerank top-50 | 0.210 | 0.413 | **0.599** | +0.057 [-0.010, +0.120] | 0.339 | 0.390 | 934 / 954 | 1152 / 1426 |
| 3+ | `dense_rerank_1024` | 1024 | dense + rerank top-50 | 0.143 | 0.360 | **0.517** | -0.026 [-0.103, +0.053] | 0.263 | 0.314 | 967 / 1022 | 1092 / 1296 |
| opt | `hybrid_rerank_company_256` | 256 | hybrid + ts_rank + rerank top-50 + company filter | 0.190 | 0.443 | **0.563** | +0.021 [-0.046, +0.087] | 0.324 | 0.373 | 999 / 1198 | 1147 / 2015 |
| opt | `hybrid_rerank_company_1024` | 1024 | hybrid + ts_rank + rerank top-50 + company filter | 0.137 | 0.390 | **0.527** | -0.016 [-0.100, +0.070] | 0.270 | 0.322 | 990 / 1107 | 1105 / 1295 |
| diag | `lexical_bm25_256` | 256 | lexical + BM25 | 0.070 | 0.113 | **0.177** | -0.366 [-0.447, -0.286] * | 0.099 | 0.115 | 1434 / 2245 | 1394 / 2213 |
| diag | `lexical_bm25_1024` | 1024 | lexical + BM25 | 0.100 | 0.147 | **0.203** | -0.339 [-0.423, -0.256] * | 0.120 | 0.139 | 1158 / 1430 | 1097 / 1416 |
| diag | `lexical_or_all_1024` | 1024 | lexical + ts_rank | 0.013 | 0.023 | **0.023** | -0.519 [-0.593, -0.446] * | 0.016 | 0.018 | 464 / 1273 | 527 / 1459 |
| diag | `lexical_idf_1024` | 1024 | lexical + ts_rank+idf-terms | 0.000 | 0.013 | **0.053** | -0.489 [-0.563, -0.413] * | 0.008 | 0.018 | 54 / 100 | 86 / 136 |

**Rows outside the brief:**
- *3+* rows add dense + rerank, which isolates what the reranker contributes on its own.
- *opt* is the optional company prefilter.
- *diag* rows measure the keyword list on its own, to explain experiment 2.

### Ingest cost per chunk size

| | 256-token | 1024-token |
|---|---|---|
| Chunks | 42,137 | 11,719 |
| Postgres table (heap + TOAST + HNSW + GIN) | 272 MB | 123 MB |
| Cold parse + chunk (shared: same parse) | 289 s | 290 s |
| Cold embed, bge-small on GPU | 235 s | 87 s |
| Upload into a fresh table via pooler | 256 s | 101 s |
| Total cold ingest | ~13 min | ~8 min |
| Chunks truncated by the 512-token embedder | 1 | 11,458 (98%) |

### Reranker cost off-GPU

The reranker is `BAAI/bge-reranker-base` scoring 50 candidates of 256-token chunks per query.
- On the RTX 3060: **0.93 s** p50.
- On CPU (8 threads, same machine): **10.5 s** median (8 questions sampled; range 8.4–12.8 s).

Free serverless hosts and GitHub Actions runners are CPU-only.

## What the experiments say

1. **Chunk size: 256 ahead of 1024, but not significantly.** Dense recall@10 is 0.542 vs 0.493, a Δ of −0.049 with CI [−0.120, +0.021]. Every pairing points the same way: dense, hybrid, and every rerank variant all favour 256. The likely cause is that bge-small embeds only the first 512 tokens, which truncates 98% of 1024-token chunks. The reranker shares the 512 cap and fares worst on 1024: `dense_rerank_1024` drops recall@1 from 0.183 to 0.143. Cost cuts the other way, since 256 needs 2.2× the storage and 2.7× the embedding time.
2. **Hybrid (Postgres FTS + dense, RRF) is significantly worse than dense alone,** by 10–17 recall@10 points on both chunk sizes. That holds for all three keyword variants:
   - **Plain `ts_rank_cd` (no IDF):** 0.023 recall@10 on its own. It matches about 90% of chunks on common words.
   - **IDF term selection:** 0.053. It cuts DB time from ~450 ms to ~45 ms but doesn't help quality.
   - **Real Okapi BM25 computed in SQL:** 0.177–0.203.

   Equal-weight RRF lets that much weaker list push good dense hits out of the top 10. One cause visible in the data: FinanceBench questions carry boilerplate ("…give a response to the question by relying on the details shown in…"), and those words are *rare* in filings, so IDF rewards them. On a plain keyword query the same BM25 finds 3M's cash-flow pages fine. I didn't strip the boilerplate or tune the RRF weights, because both would be fitting to this 150-question eval set.
3. **Reranking helps, but no gain is significant at n=150.**
   - **Hybrid + rerank:** +0.028 (ts_rank) and +0.034 (BM25) vs dense, with both CIs spanning 0. Against plain hybrid, the reranker recovers 14–16 points (0.407 → 0.570 with ts_rank, 0.437 → 0.577 with BM25).
   - **Dense + rerank:** the best point estimate overall, recall@10 **0.599** (+0.057, CI [−0.010, +0.120]). It has the right doc in the top 10 for 150/150 questions.
   - **Where the gain comes from:** novel-generated (+0.12) and metrics-generated (+0.08) questions; domain-relevant questions lose 0.03. By doc type, 10-Qs gain +0.20, earnings releases +0.14, and 10-Ks only +0.02.
   - **First-hit ranks shift both ways:** 48 questions improved and 42 got worse. The gain is real on average but noisy per question.
4. **Company prefilter: no gain.** It's 0.563 vs 0.570 for the same pipeline unfiltered (hybrid + ts_rank + rerank, 256). The matcher maps 147 of 150 questions to the right company and none to the wrong one, but retrieval was already finding the right company: doc_hit@10 is 0.98–1.00 with rerank, unfiltered. Its one benefit is latency, since the GIN scan only touches one company's rows: DB execution p50 is 47 ms vs 921 ms. Caveat: I wrote its aliases (e.g. "JnJ") after reading the question text.

## Recommendation

**Use 256-token chunks with dense retrieval. Add the bge-reranker over the dense top-50 only where a GPU is available. Don't use Postgres-FTS hybrid.**

| need | config | recall@10 | compute per query | trade-off |
|---|---|---|---|---|
| Best quality (GPU box) | `dense_rerank_256` | 0.599 | ~0.93 s GPU | +5.7 points (CI just touches 0), ~70× the compute of dense |
| Cheap / CPU-only host, CI gate | `dense_256` | 0.542 | ~13 ms | 10 s/query rerank on CPU is not viable for a demo |
| Smaller index | `dense_1024` | 0.493 | ~19 ms | 45% of the storage and ~60% of the ingest time, −4.9 points (not significant) |

- **Quality vs latency:** the reranker buys about 5–6 recall@10 points for about 0.9 s on a GPU, or about 10 s on a CPU. Hybrid buys nothing here at any latency.
- **Quality vs ingest cost:** 256 costs 2.2× the storage (272 vs 123 MB, about half the free tier's 500 MB) and about 5 more minutes of cold ingest. It's ahead on every pairing, though not significantly.
- **For Phase 4,** generation should run on `dense_rerank_256` locally. The baseline row stays `dense_256` top-5, as the brief specifies.
- **For Phase 5,** the CI gate should use `dense_256`: it's CPU-friendly and deterministic within one index build. Rebuilding the HNSW index moved recall@1 from 0.197 to 0.203 with recall@10 unchanged, so `results/baseline.json` must come from the same index the gate queries.
- **For Phase 6,** a free CPU host can't serve the reranker at interactive latency. The demo should serve dense-only, or a smaller pool (e.g. top-20) if it's measured first.

## Caveats

- **Sample size:** 150 questions is small. Only the hybrid-without-rerank and lexical-only differences are statistically significant.
- **Approximate search:** HNSW is approximate (ef_search=100). Rebuilding the index shifted recall@1/MRR by about ±0.006 for the same config.
- **Retrieval scope:** retrieval searches all 84 filings in one shared index, the harder FinanceBench "shared vector store" setting.
- **Network noise:** e2e latency includes network to us-east-1, so compare configs on the compute column.

# Phase 4: generation (measured 2026-11-23 / 2026-11-25)

Generator `openai/gpt-oss-120b` on Groq (temperature 0, reasoning_effort medium), constrained to the top-5 retrieved chunks, with every claim needing a valid `[chunk_id]` citation. A bad citation gets one regeneration, then a refusal. The judge is a different model, `qwen/qwen3.8-27b`, with 1% numeric tolerance. Refusals count as wrong. Both runs: 150/150 questions, 0 errors, 0 skipped. Reproduce with `python -m scripts.gen_summary <A.json> <B.json>`.

| config | correct | partial | incorrect | refused | accuracy | wrong or refused |
|---|---|---|---|---|---|---|
| baseline: dense top-5 (`gen_dense_256_top5`) | 48 | 9 | 18 | 75 | **0.320** | 0.620 |
| dense top-50 → bge-reranker → top-5 (`gen_dense_rerank_256_top5`) | 54 | 4 | 21 | 71 | **0.360** | 0.613 |
| FinanceBench paper, GPT-4-Turbo + shared vector store | | | | | | 0.81 |

Reranking: +0.040 accuracy, 95% paired bootstrap CI [−0.033, +0.113], so **not significant at n=150**.

The paper's figure is quoted from Islam et al. 2023 (arXiv 2311.11944): "GPT-4-Turbo used with a retrieval system incorrectly answered or refused to answer 81% of questions." It isn't a like-for-like comparison: the paper graded by hand, we grade with an LLM judge, and the models differ.

| question_type | n | baseline | rerank |
|---|---|---|---|
| domain-relevant | 50 | 0.28 | 0.28 |
| metrics-generated | 50 | 0.30 | 0.30 |
| novel-generated | 50 | 0.38 | 0.50 |

| doc_type | n | baseline | rerank |
|---|---|---|---|
| 10-K | 112 | 0.31 | 0.31 |
| 10-Q | 15 | 0.33 | 0.33 |
| 8-K | 9 | 0.56 | 0.78 |
| earnings | 14 | 0.21 | 0.50 |

**Most of the loss is retrieval, not generation.** The questions split by whether a gold page made it into the top-5 context:

| config | gold page in context | n | correct | partial | incorrect | refused |
|---|---|---|---|---|---|---|
| baseline | yes | 66 | 35 (53%) | 5 | 4 | 22 |
| baseline | no | 84 | 13 (15%) | 4 | 14 | 53 |
| rerank | yes | 69 | 35 (51%) | 2 | 11 | 21 |
| rerank | no | 81 | 19 (23%) | 2 | 10 | 50 |

- Without the gold page, the model mostly refuses (53 of 84). It answers correctly only when another page carries the same figure.
- With the gold page, about half are correct, and a third are still refused. Many of those are split statements, where the needed row is in a sibling chunk (see "What didn't work").

**Citations**

| | baseline | rerank |
|---|---|---|
| answers valid on the first attempt | 133 | 125 |
| answers regenerated once | 17 | 25 |
| refused after an invalid regeneration | 4 | 7 |
| attempts citing an id not in the context | 1 | 1 |
| mean chunks cited per answer | 1.41 | 1.52 |

**Judge spot check (by hand, 9 random verdicts from the rerank run).** 8 of the 9 match my own reading. One is too lenient: the Best Buy store count (00460) was marked "partially" with wrong numbers. Accuracy counts only "correct", so this doesn't inflate it. The judge has not been audited beyond this sample.

**Cost:** $0. Both runs ran on Groq's free tier, about 200K tokens/day for gpt-oss-120b. That cap is why they took about 2.5 days.
