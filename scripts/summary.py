"""Build the Phase 3 results table from results/*.json (latest run per config name).

    python -m scripts.summary > results/SUMMARY_table.md

Latency columns:
  compute p50/p95 = per-question embed + server-side DB execution + rerank (no network)
  e2e p50/p95     = wall time from this client (adds ~20-50 ms RTT to us-east-1 per round trip)
"""
from __future__ import annotations

import json
from pathlib import Path

from eval.run import percentile

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

ORDER = [
    ("dense_256", "1"), ("dense_1024", "1"),
    ("hybrid_256", "2"), ("hybrid_bm25_256", "2"), ("hybrid_1024", "2"), ("hybrid_idf_1024", "2"),
    ("hybrid_bm25_1024", "2"),
    ("hybrid_rerank_256", "3"), ("hybrid_bm25_rerank_256", "3"), ("hybrid_rerank_1024", "3"),
    ("dense_rerank_256", "3+"), ("dense_rerank_1024", "3+"),
    ("hybrid_rerank_company_256", "opt"), ("hybrid_rerank_company_1024", "opt"),
    ("lexical_bm25_256", "diag"), ("lexical_bm25_1024", "diag"), ("lexical_or_all_1024", "diag"),
    ("lexical_idf_1024", "diag"),
]


def latest() -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for p in sorted(RES.glob("2*_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        runs[r["name"]] = r | {"_file": p.name}
    return runs


def describe(cfg: dict) -> str:
    c = cfg["retrieval"]
    parts = [c["method"]]
    if c["method"] != "dense":
        parts.append({"or_all": "ts_rank", "idf": "ts_rank+idf-terms", "bm25": "BM25"}[c.get("lexical", "or_all")])
    if c.get("rerank"):
        parts.append("rerank top-50")
    if c.get("prefilter", "none") != "none":
        parts.append(f"{c['prefilter']} filter")
    return " + ".join(parts)


def paired_bootstrap(ref: dict, r: dict, metric: str = "recall@10", n: int = 10_000, seed: int = 0):
    """Mean per-question delta (r - ref) with a 95% percentile CI, resampling questions."""
    import random

    a = {q["id"]: q["metrics"][metric] for q in ref["per_question"]}
    b = {q["id"]: q["metrics"][metric] for q in r["per_question"]}
    ids = sorted(a.keys() & b.keys())
    d = [b[i] - a[i] for i in ids]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return sum(d) / len(d), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def compute_ms(r: dict) -> list[float]:
    return [sum(q["latency_ms"].get(k, 0.0) for k in ("embed", "db_exec", "rerank")) for q in r["per_question"]]


def main() -> int:
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    runs = latest()
    ref = runs.get("dense_256")
    print("| exp | config | chunks | retrieval | recall@1 | recall@5 | recall@10 | Δ recall@10 vs dense_256 [95% CI] "
          "| MRR | nDCG@10 | compute p50 / p95 ms | e2e p50 / p95 ms |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, exp in ORDER:
        if name not in runs:
            continue
        r = runs[name]
        a = r["retrieval"]["aggregate"]
        cm = compute_ms(r)
        tot = r["retrieval"]["latency_ms"]["total"]
        size = r["config"]["retrieval"]["table"].split("_")[1]
        if ref is None or name == "dense_256":
            delta = "—"
        else:
            m, lo, hi = paired_bootstrap(ref, r)
            sig = "" if lo <= 0 <= hi else " *"
            delta = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig}"
        print(f"| {exp} | `{name}` | {size} | {describe(r['config'])} | {a['recall@1']:.3f} | {a['recall@5']:.3f} "
              f"| **{a['recall@10']:.3f}** | {delta} | {a['mrr']:.3f} | {a['ndcg@10']:.3f} "
              f"| {percentile(cm, 50):.0f} / {percentile(cm, 95):.0f} | {tot['p50']:.0f} / {tot['p95']:.0f} |")
    print("\n`*` = 95% paired-bootstrap CI (10k resamples over questions) excludes 0.")
    missing = [n for n, _ in ORDER if n not in runs]
    if missing:
        print(f"\n_not run: {', '.join(missing)}_")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
