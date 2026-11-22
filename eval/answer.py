"""Generation + answer eval (kept separate from retrieval metrics).

    python -m eval.answer --config configs/gen_<name>.yaml [--limit N]

Pipeline per question: retrieve (config `retrieval:`) -> generate with chunk-id citations
(validated; one regeneration, then refuse) -> LLM judge vs reference answer.
Writes results/<timestamp>_<name>.json with a `generation` section only.

Free-tier aware: every LLM call is cached by prompt hash, so if a daily quota runs out the run
saves partial results (status "incomplete") and a re-run resumes where it stopped at no cost.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import yaml

from eval.judge import VERDICTS, judge
from eval.run import RESULTS, ROOT, config_hash, git_sha, percentile
from footnote.dataset import DOWNLOAD_REPORT, load_questions, normalize_doc_type
from footnote.generate import generate
from footnote.llm import Groq, QuotaExhausted


def accuracy(rows: list[dict]) -> dict:
    n = len(rows)
    c = Counter(r["verdict"] for r in rows)
    out = {v: c.get(v, 0) for v in VERDICTS} | {"n": n}
    if n:
        out["accuracy"] = round(c.get("correct", 0) / n, 4)
        out["correct_or_partial"] = round((c.get("correct", 0) + c.get("partially", 0)) / n, 4)
        out["wrong_or_refused"] = round((c.get("incorrect", 0) + c.get("refused", 0)) / n, 4)
    return out


def summarize(rows: list[dict]) -> dict:
    by = {}
    for key in ("question_type", "question_reasoning", "doc_type"):
        g = defaultdict(list)
        for r in rows:
            g[str(r[key] or "n/a")].append(r)
        by[key] = {k: accuracy(v) for k, v in sorted(g.items())}
    status = Counter(r["status"] for r in rows)
    first_try_valid = sum(r["validations"][0]["valid"] for r in rows)
    hallucinated = sum(bool(v["unknown_ids"]) for r in rows for v in r["validations"])
    return {
        "aggregate": accuracy(rows),
        "breakdowns": by,
        "citations": {
            "first_attempt_valid": first_try_valid,
            "regenerated": sum(r["attempts"] == 2 for r in rows),
            "refused_invalid_citations": status.get("refused_invalid_citations", 0),
            "attempts_citing_unretrieved_ids": hallucinated,
            "status": dict(status),
            "mean_cited_chunks_when_answered": round(
                sum(len(r["validations"][-1]["cited"]) for r in rows if r["status"] == "answered")
                / max(1, status.get("answered", 0)), 2),
        },
    }


def main() -> int:
    from footnote.retrieve import Retriever

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    path = Path(args.config)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    gen_cfg, judge_cfg = cfg["generation"], cfg["judge"]
    if gen_cfg["model"] == judge_cfg["model"]:
        raise SystemExit("judge model must differ from the generator model")

    questions = load_questions()
    report = json.loads(DOWNLOAD_REPORT.read_text()) if DOWNLOAD_REPORT.exists() else {"docs_failed": {}}
    skipped = [q["financebench_id"] for q in questions if q["doc_name"] in report["docs_failed"]]
    questions = [q for q in questions if q["financebench_id"] not in skipped][: args.limit]

    retriever = Retriever(cfg["retrieval"])
    client = Groq()
    gparams = {k: v for k, v in gen_cfg.items() if k != "model"}
    jparams = {k: v for k, v in judge_cfg.items() if k != "model"}

    rows, errors, stopped, t0 = [], [], None, time.perf_counter()
    gen_ms = []
    for i, q in enumerate(questions, 1):
        res = retriever.search(q["question"])
        try:
            tg = time.perf_counter()
            g = generate(client, q["question"], res.hits, gen_cfg["model"], **gparams)
            gen_ms.append((time.perf_counter() - tg) * 1e3)
            j = judge(client, judge_cfg["model"], q["question"], q["answer"], q.get("justification"),
                      g.answer, **jparams)
        except QuotaExhausted as e:
            stopped = str(e)
            print(f"\nSTOPPED at question {i}/{len(questions)}: {e}")
            break
        except Exception as e:  # noqa: BLE001 - one bad API response must not kill a 150-question run
            errors.append({"id": q["financebench_id"], "error": f"{type(e).__name__}: {str(e)[:300]}"})
            print(f"  [{i}/{len(questions)}] ERROR     {q['financebench_id']}: {str(e)[:120]}", flush=True)
            continue
        rows.append({
            "id": q["financebench_id"], "question": q["question"], "reference": q["answer"],
            "question_type": q.get("question_type"), "question_reasoning": q.get("question_reasoning"),
            "doc_type": normalize_doc_type(q.get("doc_type"), q["doc_name"]),
            "retrieved": [{"chunk_id": h.chunk_id, "doc": h.doc_name, "pages": h.pages} for h in res.hits],
            "answer": g.answer, "status": g.status, "attempts": g.attempts, "validations": g.validations,
            "raw_attempts": g.raw, "gen_usage": g.usage,
            "verdict": j["verdict"], "judge_reasoning": j["reasoning"], "judge_usage": j["usage"],
        })
        print(f"  [{i}/{len(questions)}] {j['verdict']:<9} {g.status:<26} {q['financebench_id']}", flush=True)

    out = {
        "name": path.stem, "kind": "generation", "config": cfg, "config_hash": config_hash(cfg),
        "created_utc": datetime.utcnow().isoformat(timespec="seconds"), "git_sha": git_sha(),
        "complete": stopped is None and not errors and len(rows) == len(questions),
        "errors": errors,
        "stopped_reason": stopped,
        "n_questions_planned": len(questions), "n_questions_done": len(rows), "skipped": skipped,
        "generation": summarize(rows) if rows else {},
        "llm_stats": client.stats,
        "generate_latency_ms": {"p50": round(percentile(gen_ms, 50)), "p95": round(percentile(gen_ms, 95))}
                               if gen_ms else {},
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "per_question": rows,
    }
    dest = RESULTS / f"{datetime.now():%Y%m%d-%H%M%S}_{path.stem}.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    a = out["generation"].get("aggregate", {})
    print(f"\n{path.stem}: done {len(rows)}/{len(questions)} complete={out['complete']}  "
          f"accuracy={a.get('accuracy')} correct_or_partial={a.get('correct_or_partial')} "
          f"wrong_or_refused={a.get('wrong_or_refused')}")
    print("verdicts:", {v: a.get(v) for v in VERDICTS})
    print("citations:", out["generation"].get("citations"))
    print("llm:", client.stats)
    print(f"wrote {dest.relative_to(ROOT)}")
    return 0 if out["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
