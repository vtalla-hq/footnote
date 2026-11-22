"""Citation validator for generated answers.

Answers cite chunks inline as [chunk_id] (several as [id1][id2] or [id1, id2]).
An answer is VALID when:
  1. it is an explicit refusal (starts with REFUSAL_PREFIX), or
  2. it cites at least one chunk, every cited chunk_id is in the retrieved set, and every
     claim-bearing sentence (one stating a quantity: $, %, decimals, thousands; not years or
     Q/FY labels) carries a citation. Both [id] and the full-width 【id】 are accepted.
Citations written right after the sentence's full stop ("... $5.2 billion. [id]") count for
that sentence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

REFUSAL_PREFIX = "INSUFFICIENT_CONTEXT"
CHUNK_ID_RE = re.compile(r"[A-Za-z0-9_.&\-]+::p\d{4}::\d{2}")
# [id] or the full-width 【id】 that gpt-oss models emit natively
BRACKET_RE = re.compile(r"\[([^\[\]]+)\]|【([^【】]+)】")
# a quantity: $ amount, percentage, decimal, or a number with thousands separators / >= 3 digits.
# Years (1990-2039) and period labels (Q2, FY2023, FY 2023) are not claims.
NUM_RE = re.compile(r"(?<![A-Za-z0-9])(\$?\(?-?\d[\d,]*(?:\.\d+)?\)?%?)")
PERIOD_RE = re.compile(r"(?:\bQ[1-4]|\bFY|\bH[12]|\bItem|\bNote|\bForm|\bSection|\bRule)[\s\u202f]*$", re.I)  # labels, not quantities
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[$\-*•])|\n+")


@dataclass
class Validation:
    valid: bool
    refusal: bool
    cited: list[str] = field(default_factory=list)
    unknown_ids: list[str] = field(default_factory=list)      # cited but not retrieved
    uncited_claims: list[str] = field(default_factory=list)   # numeric sentences without a cite
    reason: str = ""


def is_refusal(answer: str) -> bool:
    return answer.strip().upper().startswith(REFUSAL_PREFIX)


def extract_citations(text: str) -> list[str]:
    ids: list[str] = []
    for a, b in BRACKET_RE.findall(text):
        for m in CHUNK_ID_RE.findall(a or b):
            if m not in ids:
                ids.append(m)
    return ids


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text) if p and p.strip()]
    merged: list[str] = []
    for p in parts:
        # a segment that is only citations belongs to the previous sentence
        if merged and not BRACKET_RE.sub("", p).strip(" .,;:"):
            merged[-1] += " " + p
        else:
            merged.append(p)
    return merged


def has_claim(sentence: str) -> bool:
    """True if the sentence states a quantity (see NUM_RE); years and Q/FY labels don't count."""
    body = BRACKET_RE.sub("", sentence)
    for m in NUM_RE.finditer(body):
        tok = m.group(1)
        bare = tok.strip("$()%-")
        if PERIOD_RE.search(body[: m.start()]):
            continue  # "Q2", "FY2023", "FY 2023"
        if re.fullmatch(r"(19[9]\d|20[0-3]\d)", tok):
            continue  # a bare year
        if "$" in tok or "%" in tok or "," in bare or "." in bare or len(bare) >= 3 or body[m.end():m.end() + 2].strip().startswith("%"):
            return True
    return False


def strip_citations(text: str) -> str:
    text = re.sub(r"\s*\[[^\[\]]*::p\d{4}::\d{2}[^\[\]]*\]", "", text)
    return re.sub(r"\s*【[^【】]*::p\d{4}::\d{2}[^【】]*】", "", text).strip()


def validate(answer: str, retrieved_ids: list[str] | set[str]) -> Validation:
    if is_refusal(answer):
        return Validation(valid=True, refusal=True, reason="refusal")
    retrieved = set(retrieved_ids)
    cited = extract_citations(answer)
    unknown = [c for c in cited if c not in retrieved]
    uncited = []
    for s in _sentences(answer):
        if has_claim(s) and not extract_citations(s):
            uncited.append(s[:200])
    reasons = []
    if not cited:
        reasons.append("no citations")
    if unknown:
        reasons.append(f"cites chunk_ids not in the retrieved set: {', '.join(unknown)}")
    if uncited:
        reasons.append(f"{len(uncited)} numeric claim(s) without a citation")
    return Validation(valid=not reasons, refusal=False, cited=cited, unknown_ids=unknown,
                      uncited_claims=uncited, reason="; ".join(reasons) or "ok")
