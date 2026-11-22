"""Retrieval eval over FinanceBench.

    python -m eval.run --config configs/dense_256.yaml [--limit N]

Writes results/<timestamp>_<config>.json: config + config hash, per-question results, aggregate
metrics, breakdowns (question_type, question_reasoning, doc_type) and retrieval latency p50/p95.
Retrieval metrics only; generation metrics live in a separate module and separate files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from eval.metrics import METRICS, aggregate, first_hit_rank, gold_pages, score
from footnote.dataset import DOWNLOAD_REPORT, load_questions, normalize_doc_type

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
BREAKDOWNS = ["question_type", "question_reasoning", "doc_type"]


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=ROOT, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def breakdown(per_q: list[dict]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for key in BREAKDOWNS:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in per_q:
            groups[str(r[key] or "n/a")].append(r["metrics"])
        out[key] = {g: aggregate(rows) for g, rows in sorted(groups.items())}
    return out


def run(cfg: dict, name: str, limit: int | None = None, warmup: int = 3) -> dict:
    from footnote.retrieve import Retriever, network_rtt_ms

    questions = load_questions()
    report = json.loads(DOWNLOAD_REPORT.read_text()) if DOWNLOAD_REPORT.exists() else {"docs_failed": {}}
    skipped = [q["financebench_id"] for q in questions if q["doc_name"] in report["docs_failed"]]
    questions = [q for q in questions if q["financebench_id"] not in skipped][:limit]

    retriever = Retriever(cfg["retrieval"])
    for q in questions[:warmup]:  # warm model + connection so latency isn't dominated by first call
        retriever.search(q["question"])

    rtt = network_rtt_ms(retriever.conn)
    per_q, lat = [], defaultdict(list)
    for q in questions:
        res = retriever.search(q["question"], measure_db=True)
        for k, v in res.timings_ms.items():
            lat[k].append(v)
        ranked = [(h.doc_name, h.pages) for h in res.hits]
        gold = gold_pages(q)
        per_q.append({
            "id": q["financebench_id"],
            "question": q["question"],
            "question_type": q.get("question_type"),
            "question_reasoning": q.get("question_reasoning"),
            "doc_type": normalize_doc_type(q.get("doc_type"), q["doc_name"]),
            "gold": sorted([list(g) for g in gold]),
            "retrieved": [{"chunk_id": h.chunk_id, "doc": h.doc_name, "pages": h.pages,
                           "type": h.chunk_type, "score": round(h.score, 4)} for h in res.hits],
            "first_hit_rank": first_hit_rank(ranked, gold),
            "metrics": score(ranked, gold),
            "latency_ms": {k: round(v, 1) for k, v in res.timings_ms.items()},
        })

    return {
        "name": name,
        "config": cfg,
        "config_hash": config_hash(cfg),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "n_questions": len(per_q),
        "skipped": skipped,
        "retrieval": {
            "aggregate": aggregate([r["metrics"] for r in per_q]),
            "breakdowns": breakdown(per_q),
            # total = end-to-end from this client (embed + network + DB); db_exec = server-side
            # execution only. network_rtt is a bare `select 1` round trip for context.
            "network_rtt_ms": rtt,
            "latency_ms": {k: {"p50": round(percentile(v, 50), 1), "p95": round(percentile(v, 95), 1),
                               "mean": round(statistics.fmean(v), 1)} for k, v in lat.items()},
        },
        "per_question": per_q,
    }


def summary_table(res: dict) -> str:
    agg = res["retrieval"]["aggregate"]
    lines = [f"### {res['name']}  (config {res['config_hash']}, n={res['n_questions']}, skipped={len(res['skipped'])})",
             "", "| " + " | ".join(METRICS) + " | p50 ms | p95 ms |",
             "|" + "---|" * (len(METRICS) + 2),
             "| " + " | ".join(f"{agg[m]:.3f}" for m in METRICS)
             + f" | {res['retrieval']['latency_ms']['total']['p50']} | {res['retrieval']['latency_ms']['total']['p95']} |",
             "", "latency ms (p50 / p95): " + ", ".join(
                 f"{k} {v['p50']} / {v['p95']}" for k, v in res["retrieval"]["latency_ms"].items())
             + f"; network RTT p50 {res['retrieval']['network_rtt_ms']['p50']}"]
    for key, groups in res["retrieval"]["breakdowns"].items():
        lines += ["", f"**by {key}**", "", "| group | n | recall@5 | recall@10 | mrr | ndcg@10 |", "|---|---|---|---|---|---|"]
        for g, a in groups.items():
            lines.append(f"| {g} | {a['n']} | {a['recall@5']:.3f} | {a['recall@10']:.3f} | {a['mrr']:.3f} | {a['ndcg@10']:.3f} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N questions only (smoke test)")
    ap.add_argument("--out", default=None, help="output path (default results/<timestamp>_<config>.json)")
    args = ap.parse_args()
    path = Path(args.config)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    name = path.stem
    t0 = time.perf_counter()
    res = run(cfg, name, args.limit)
    res["wall_seconds"] = round(time.perf_counter() - t0, 1)
    out = Path(args.out) if args.out else RESULTS / f"{datetime.now():%Y%m%d-%H%M%S}_{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(summary_table(res))
    print(f"\nwrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
