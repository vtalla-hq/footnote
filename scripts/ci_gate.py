"""CI regression gate: re-ingest a fixed doc subset with this checkout's code, run the retrieval
eval, and fail if recall@10 drops more than the allowed margin vs results/baseline.json.

    python -m scripts.ci_gate                    # gate (exit 1 on regression)
    python -m scripts.ci_gate --write-baseline   # refresh results/baseline.json (run on main)

No LLM calls. Uses a throwaway table footnote.ci_<run id> with no HNSW index (exact search,
deterministic), dropped at the end. Run on CPU for the baseline too (CUDA_VISIBLE_DEVICES="")
so embeddings match the GitHub runner.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "results" / "baseline.json"


def ensure_pdfs(docs: list[str]) -> None:
    from footnote.dataset import PDF_DIR, fetch_questions
    from scripts.download import fetch

    fetch_questions()
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    failed = {d: e for d, e in map(fetch, docs) if e}
    if failed:
        raise SystemExit(f"could not fetch CI docs: {failed}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ci.yaml")
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    run_id = re.sub(r"[^a-z0-9_]", "_", os.environ.get("GITHUB_RUN_ID", f"local_{int(time.time())}").lower())
    table = f"ci_{run_id}"

    from eval.run import run
    from footnote import db
    from footnote.ingest import ingest, load_config

    ensure_pdfs(cfg["docs"])
    icfg = load_config(str(ROOT / cfg["ingest_config"]))
    conn = db.connect()
    t0 = time.perf_counter()
    try:
        # cold: fresh parse + embeddings every time, so a local CPU baseline matches the CPU runner
        stats = ingest(icfg, cfg["docs"], table, workers=args.workers, cold=True, hnsw=False,
                       prune_missing_docs=False)
        res = run({"retrieval": cfg["retrieval"] | {"table": table}}, "ci_gate", docs=cfg["docs"])
        # hash what defines the experiment (gate config + ingest config), not the per-run table name
        from eval.run import config_hash

        res["config_hash"] = config_hash({"ci": cfg, "ingest": icfg})
    finally:
        db.drop_table(conn, table)
    res["ci"] = {"table": table, "ingest": {k: stats[k] for k in ("docs", "pages", "chunks", "table_chunks")},
                 "ingest_config": icfg, "seconds": round(time.perf_counter() - t0, 1)}
    # per_question kept: it lets the job summary show which questions moved

    if args.write_baseline:
        BASELINE.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"wrote {BASELINE.relative_to(ROOT)}: recall@10 = {res['retrieval']['aggregate']['recall@10']:.4f} "
              f"over {res['n_questions']} questions")
        return 0

    out = ROOT / "results" / "ci_latest.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    # Gate on recall@10 AND MRR. recall@10 alone is blind to chunking damage: shredding tables or
    # shrinking chunks to 64 tokens left the gold page in the top 10 (recall@10 even rose) while
    # MRR fell 2-4 points. Exact search -> identical scores every run, so 1 point is not noise.
    checks = []
    for metric, key in (("recall@10", "max_recall10_drop"), ("mrr", "max_mrr_drop")):
        b = base["retrieval"]["aggregate"][metric]
        c = res["retrieval"]["aggregate"][metric]
        limit = float(cfg[key])
        checks.append((metric, b, c, limit, (b - c) <= limit + 1e-9))
    ok = all(c[4] for c in checks)

    from eval.compare import rank_changes, table as cmp_table

    base["name"], res["name"] = "baseline (main)", "this PR"
    lines = [
        f"## Retrieval regression gate: {'PASS' if ok else 'FAIL'}",
        "",
        *[f"- {'✅' if passed else '❌'} {m} **{c:.3f}** vs baseline **{b:.3f}** "
          f"(change {c - b:+.3f}; allowed drop {lim:.3f})" for m, b, c, lim, passed in checks],
        "",
        f"{res['n_questions']} questions over {len(cfg['docs'])} filings; chunks {res['ci']['ingest']['chunks']} "
        f"(baseline {base['ci']['ingest']['chunks']}), table chunks {res['ci']['ingest']['table_chunks']} "
        f"(baseline {base['ci']['ingest']['table_chunks']}).",
        "",
        cmp_table([base, res], by="question_type"),
        "",
        rank_changes(base, res),
    ]
    report = "\n".join(lines)
    print(report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(report + "\n")
    if not ok:
        for m, b, c, lim, passed in checks:
            if not passed:
                print(f"\nFAIL: {m} dropped {b - c:.3f} (> {lim:.3f})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
