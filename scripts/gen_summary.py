"""Summarize Phase 4 generation runs: accuracy with a paired bootstrap CI, breakdowns, and an
outcome split by whether a gold page was in the context.

    python -m scripts.gen_summary results/A_gen.json results/B_gen.json
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

from eval.metrics import covered, gold_pages
from footnote.dataset import QUESTIONS_PATH

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def gold_in_context(row, q):
    gold = gold_pages(q)
    return any(covered((h["doc"], h["pages"]), gold) for h in row["retrieved"])


def boot(a, b, n=10_000, seed=0):
    rnd = random.Random(seed)
    idx = range(len(a))
    diffs = sorted(sum(b[i] - a[i] for i in s) / len(a)
                   for s in ([rnd.choice(idx) for _ in idx] for _ in range(n)))
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)]


def main(paths):
    qs = {q["financebench_id"]: q for q in map(json.loads, open(QUESTIONS_PATH, encoding="utf-8"))}
    runs = [load(p) for p in paths]
    print("| config | n | correct | partial | incorrect | refused | accuracy | wrong or refused |")
    print("|---|---|---|---|---|---|---|---|")
    for r in runs:
        a = r["generation"]["aggregate"]
        print(f"| {r['name']} | {a['n']} | {a['correct']} | {a['partially']} | {a['incorrect']} | {a['refused']} "
              f"| {a['accuracy']:.3f} | {a['wrong_or_refused']:.3f} |")
    if len(runs) == 2:
        ids = [x["id"] for x in runs[0]["per_question"]]
        v = [{x["id"]: x["verdict"] == "correct" for x in r["per_question"]} for r in runs]
        a, b = [int(v[0][i]) for i in ids], [int(v[1][i]) for i in ids]
        lo, hi = boot(a, b)
        print(f"\naccuracy delta ({runs[1]['name']} - {runs[0]['name']}): {(sum(b) - sum(a)) / len(a):+.3f} "
              f"[95% paired bootstrap {lo:+.3f}, {hi:+.3f}]")
    for by in ("question_type", "question_reasoning", "doc_type"):
        print(f"\n| {by} | n | " + " | ".join(f"{r['name']} acc" for r in runs) + " |")
        print("|---|---|" + "---|" * len(runs))
        keys = runs[0]["generation"]["breakdowns"][by]
        for k in keys:
            cells = [f"{r['generation']['breakdowns'][by][k]['accuracy']:.2f}" for r in runs]
            print(f"| {k} | {keys[k]['n']} | " + " | ".join(cells) + " |")
    for r in runs:
        c = Counter()
        for x in r["per_question"]:
            c[(gold_in_context(x, qs[x["id"]]), x["verdict"])] += 1
        print(f"\n{r['name']}: outcome by whether a gold page was in the top-5 context")
        for g in (True, False):
            row = {k[1]: n for k, n in c.items() if k[0] == g}
            print(f"  gold in context={g}: n={sum(row.values())} {dict(sorted(row.items()))}")
        print("  citations:", json.dumps(r["generation"]["citations"]))


if __name__ == "__main__":
    main(sys.argv[1:])
