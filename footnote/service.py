"""The demo's question-answering service (used by the Vercel function and local dev).

Retrieval = the Phase 3/4 baseline (`dense_256`, top-5): the reranker is not served because it
needs ~10 s/query on CPU. Generation = the eval's generator (gpt-oss-120b, same prompt and
citation validation); if a 120b call is rate-limited the request falls back to gpt-oss-20b and
the response says so.
"""
from __future__ import annotations

import time
from functools import lru_cache

from footnote.citations import extract_citations
from footnote.generate import generate
from footnote.llm import Groq, QuotaExhausted
from footnote.retrieve import Retriever

RETRIEVAL = {"method": "dense", "table": "chunks_256", "k": 5, "ef_search": 100}
PRIMARY = "openai/gpt-oss-120b"
FALLBACK = "openai/gpt-oss-20b"
GEN_PARAMS = {"temperature": 0, "reasoning_effort": "medium", "include_reasoning": False, "max_tokens": 2048}
MAX_QUESTION_CHARS = 600


@lru_cache(maxsize=1)
def _retriever() -> Retriever:
    return Retriever(RETRIEVAL)


@lru_cache(maxsize=1)
def _client() -> Groq:
    # max_wait_s=0: any 429 raises immediately so we can fall back instead of waiting
    return Groq(max_retries=2, max_wait_s=0)


def answer(question: str) -> dict:
    question = question.strip()
    if not question:
        raise ValueError("empty question")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"question longer than {MAX_QUESTION_CHARS} characters")
    t0 = time.perf_counter()
    try:
        res = _retriever().search(question)
    except Exception:
        _retriever.cache_clear()  # stale connection after a cold period: reconnect once
        res = _retriever().search(question)
    t1 = time.perf_counter()
    model, fallback_reason = PRIMARY, None
    try:
        g = generate(_client(), question, res.hits, PRIMARY, **GEN_PARAMS)
    except QuotaExhausted as e:
        model, fallback_reason = FALLBACK, f"{PRIMARY} rate-limited: {str(e)[:160]}"
        g = generate(_client(), question, res.hits, FALLBACK, **GEN_PARAMS)
    t2 = time.perf_counter()
    cited = set(extract_citations(g.answer))
    return {
        "question": question,
        "answer": g.answer,
        "status": g.status,  # answered | refused | refused_invalid_citations
        "model": model,
        "fallback_reason": fallback_reason,
        "citations": [{"chunk_id": h.chunk_id, "doc_name": h.doc_name, "company": h.company,
                       "doc_type": h.doc_type, "fiscal_year": h.fiscal_year, "pages": h.pages}
                      for h in res.hits if h.chunk_id in cited],
        "chunks": [{"chunk_id": h.chunk_id, "doc_name": h.doc_name, "company": h.company,
                    "doc_type": h.doc_type, "fiscal_year": h.fiscal_year, "pages": h.pages,
                    "section": h.section, "chunk_type": h.chunk_type, "score": round(h.score, 4),
                    "text": h.text, "cited": h.chunk_id in cited} for h in res.hits],
        "attempts": g.attempts,
        "latency_ms": {"retrieval": round((t1 - t0) * 1e3), "generation": round((t2 - t1) * 1e3)},
    }
