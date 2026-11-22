"""Grounded answer generation with chunk-id citations.

Flow: build context blocks labelled with chunk_id -> generate -> validate citations
(footnote.citations) -> if invalid, regenerate ONCE with the validator's complaint -> if still
invalid, refuse. The final answer is always either validly cited or a refusal.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from footnote.citations import REFUSAL_PREFIX, Validation, validate
from footnote.llm import Groq
from footnote.retrieve import Hit

SYSTEM = f"""You are a financial analyst answering questions about SEC filings.

Rules:
- Use ONLY the numbered context blocks. Never use outside knowledge.
- Cite the chunk_id of every block you rely on, in square brackets, right after each sentence
  that uses it, e.g. "Capex was $1,577 million [3M_2018_10K::p0060::00]." Copy chunk_ids exactly.
  Every sentence that contains a number needs a citation.
- If the answer needs a calculation, show it briefly (inputs, formula, result) with citations.
- If the context does not contain what is needed, reply exactly:
  {REFUSAL_PREFIX}: <one short sentence on what is missing>
- Be concise: at most 150 words. Lead with the direct answer."""


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for h in hits:
        pages = f"p.{h.pages[0]}" if len(h.pages) == 1 else f"pp.{h.pages[0]}-{h.pages[-1]}"
        head = f"[chunk_id: {h.chunk_id}] {h.company} {h.doc_type} FY{h.fiscal_year}, {pages}"
        if h.section:
            head += f", section: {h.section}"
        blocks.append(f"{head}\n{h.text}")
    return "\n\n---\n\n".join(blocks)


def user_prompt(question: str, hits: list[Hit]) -> str:
    return f"Context blocks:\n\n{format_context(hits)}\n\n===\n\nQuestion: {question}"


@dataclass
class Generation:
    answer: str
    status: str  # answered | refused | refused_invalid_citations
    attempts: int
    validations: list[dict] = field(default_factory=list)
    usage: list[dict] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)


def generate(client: Groq, question: str, hits: list[Hit], model: str, **params) -> Generation:
    ids = [h.chunk_id for h in hits]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(question, hits)}]
    out = Generation(answer="", status="", attempts=0)
    for attempt in (1, 2):
        resp = client.chat(model, messages, **params)
        text = resp["content"].strip()
        v: Validation = validate(text, ids)
        out.attempts = attempt
        out.raw.append(text)
        out.usage.append(resp["usage"] | {"cached": resp["cached"]})
        out.validations.append(asdict(v))
        if v.valid:
            out.answer = text
            out.status = "refused" if v.refusal else "answered"
            return out
        # one regeneration with the validator's complaint
        messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": f"Your answer failed citation validation: {v.reason}. Valid chunk_ids are: "
                                        f"{', '.join(ids)}. Rewrite the answer following the rules exactly."},
        ]
    out.answer = f"{REFUSAL_PREFIX}: could not produce an answer with valid citations."
    out.status = "refused_invalid_citations"
    return out
