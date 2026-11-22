"""LLM judge: grade a generated answer against the FinanceBench reference answer.

Verdicts: correct | partially | incorrect | refused. Refusals are detected deterministically and
never sent to the judge. The judge model must differ from the generator (enforced in eval.answer).
"""
from __future__ import annotations

import json
import re

from footnote.citations import is_refusal, strip_citations
from footnote.llm import Groq

VERDICTS = ("correct", "partially", "incorrect", "refused")

JUDGE_SYSTEM = """You grade answers to financial questions against a reference answer.
Return JSON: {"reasoning": "<2-4 short sentences>", "verdict": "correct" | "partially" | "incorrect"}

Grading rules:
- Judge only against the REFERENCE ANSWER (and its justification). Do not use outside knowledge.
- Numeric tolerance: a number matches if it equals the reference after reasonable rounding, or is
  within 1% relative difference. Accept equivalent units and formats ($1,577 million = $1.577
  billion = 1577; 0.96 = 96%). Sign conventions for outflows (e.g. "(1,577)" vs "1,577") match.
- correct: the candidate's key facts / final number(s) match the reference, and any yes/no or
  directional conclusion agrees. Extra correct detail is fine.
- partially: some but not all required elements are right (e.g. right number but wrong or missing
  conclusion, one of two figures right, right direction but the number is off by more than 1%).
- incorrect: the key figure or conclusion is wrong, or the answer does not address the question.
- A candidate that hedges between several values without committing is incorrect."""


def judge_prompt(question: str, reference: str, justification: str | None, candidate: str) -> str:
    just = f"\nREFERENCE JUSTIFICATION: {justification}" if justification else ""
    return (f"QUESTION: {question}\n\nREFERENCE ANSWER: {reference}{just}\n\n"
            f"CANDIDATE ANSWER: {strip_citations(candidate)}")


def judge(client: Groq, model: str, question: str, reference: str, justification: str | None,
          candidate: str, **params) -> dict:
    if is_refusal(candidate):
        return {"verdict": "refused", "reasoning": "candidate refused", "usage": None, "cached": None}
    messages = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": judge_prompt(question, reference, justification, candidate)}]
    try:
        resp = client.chat(model, messages, response_format={"type": "json_object"}, **params)
    except RuntimeError as e:
        if "json_validate_failed" not in str(e):
            raise
        # Groq rejects JSON mode output that is cut off; retry once in plain mode, parse leniently
        resp = client.chat(model, messages, **(params | {"max_tokens": max(1024, params.get("max_tokens", 0))}))
    verdict, reasoning = parse_verdict(resp["content"])
    return {"verdict": verdict, "reasoning": reasoning, "usage": resp["usage"], "cached": resp["cached"]}


def parse_verdict(text: str) -> tuple[str, str]:
    try:
        j = json.loads(text)
        v = str(j.get("verdict", "")).strip().lower()
        r = str(j.get("reasoning", ""))
    except json.JSONDecodeError:
        m = re.search(r'"verdict"\s*:\s*"(\w+)"', text)
        v, r = (m.group(1).lower() if m else ""), text[:300]
    if v.startswith("partial"):
        v = "partially"
    if v not in ("correct", "partially", "incorrect"):
        v = "incorrect"
        r = f"[unparseable judge output, scored incorrect] {r}"
    return v, r
