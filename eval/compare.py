"""Markdown diff table between eval runs.

    python -m eval.compare results/A.json results/B.json [more.json ...] [--by question_type]

The first file is the reference; other columns show value and delta vs the reference.
Also lists questions whose first-hit rank changed between the first two runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.metrics import METRICS


def load(p: str) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def fmt(v: float, ref: float | None) -> str:
    if ref is None:
        return f"{v:.3f}"
    d = v - ref
    return f"{v:.3f} ({'+' if d >= 0 else ''}{d:.3f})"


def table(runs: list[dict], by: str | None) -> str:
    names = [f"{r['name']} `{r['config_hash'][:6]}`" for r in runs]
    out = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(runs)]
    ref = runs[0]["retrieval"]
    for m in METRICS:
        vals = [r["retrieval"]["aggregate"][m] for r in runs]
        out.append(f"| {m} | " + " | ".join(fmt(v, None if i == 0 else vals[0]) for i, v in enumerate(vals)) + " |")
    for part in ("total", "db_exec"):
        for stat in ("p50", "p95"):
            vals = [r["retrieval"]["latency_ms"].get(part, {}).get(stat) for r in runs]
            out.append(f"| {part} latency {stat} ms | " + " | ".join("—" if v is None else f"{v:.0f}" for v in vals) + " |")
    if by:
        groups = sorted(ref["breakdowns"][by])
        out += ["", f"**recall@10 by {by}**", "", "| group | n | " + " | ".join(names) + " |",
                "|---|---|" + "---|" * len(runs)]
        for g in groups:
            vals = [r["retrieval"]["breakdowns"][by].get(g, {}).get("recall@10", float("nan")) for r in runs]
            n = ref["breakdowns"][by][g]["n"]
            out.append(f"| {g} | {n} | " + " | ".join(fmt(v, None if i == 0 else vals[0]) for i, v in enumerate(vals)) + " |")
    return "\n".join(out)


def rank_changes(a: dict, b: dict) -> str:
    ra = {q["id"]: q["first_hit_rank"] for q in a["per_question"]}
    rb = {q["id"]: q["first_hit_rank"] for q in b["per_question"]}
    better = [i for i in ra if i in rb and (rb[i] or 99) < (ra[i] or 99)]
    worse = [i for i in ra if i in rb and (rb[i] or 99) > (ra[i] or 99)]
    return f"first-hit rank vs {a['name']}: {len(better)} improved, {len(worse)} worse, " \
           f"{len(ra) - len(better) - len(worse)} unchanged"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--by", default=None, choices=["question_type", "question_reasoning", "doc_type"])
    args = ap.parse_args()
    runs = [load(p) for p in args.runs]
    print(table(runs, args.by))
    for r in runs[1:]:
        print("\n" + rank_changes(runs[0], r) + f" ({r['name']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
