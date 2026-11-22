"""Retrieval metrics against FinanceBench gold evidence pages.

Gold for a question is a set of (doc_name, page) pairs, 1-based pages (FinanceBench's
evidence_page_num is 0-based; see gold_pages()). A retrieved chunk *hits* a gold pair when the
doc matches AND the chunk's page list contains that page.

Because several chunks can land on the same gold page, metrics are computed over *gold pages
covered*, not raw chunk counts:

* recall@k  = |gold pairs covered by the top-k chunks| / |gold pairs|
* hit@k     = 1 if any of the top-k chunks hits any gold pair, else 0
* MRR       = 1 / rank of the first hitting chunk (0 if none in the retrieved list)
* nDCG@k    = binary gain 1 for a chunk that covers a gold pair not already covered by a
              higher-ranked chunk, log2 discount; ideal = min(|gold|, k) gains at ranks 1..n.

For single-evidence questions recall@k == hit@k.
"""
from __future__ import annotations

import math

Gold = set[tuple[str, int]]
Retrieved = list[tuple[str, list[int]]]  # ranked [(doc_name, pages)]


def gold_pages(question: dict) -> Gold:
    return {(ev["doc_name"], int(ev["evidence_page_num"]) + 1) for ev in question["evidence"]}


def covered(chunk: tuple[str, list[int]], gold: Gold) -> set[tuple[str, int]]:
    doc, pages = chunk
    return {(d, p) for d, p in gold if d == doc and p in pages}


def first_hit_rank(retrieved: Retrieved, gold: Gold) -> int | None:
    for i, ch in enumerate(retrieved, 1):
        if covered(ch, gold):
            return i
    return None


def recall_at_k(retrieved: Retrieved, gold: Gold, k: int) -> float:
    if not gold:
        return 0.0
    got: set = set()
    for ch in retrieved[:k]:
        got |= covered(ch, gold)
    return len(got) / len(gold)


def hit_at_k(retrieved: Retrieved, gold: Gold, k: int) -> float:
    r = first_hit_rank(retrieved[:k], gold)
    return 1.0 if r is not None else 0.0


def mrr(retrieved: Retrieved, gold: Gold) -> float:
    r = first_hit_rank(retrieved, gold)
    return 1.0 / r if r else 0.0


def ndcg_at_k(retrieved: Retrieved, gold: Gold, k: int = 10) -> float:
    if not gold:
        return 0.0
    seen: set = set()
    dcg = 0.0
    for i, ch in enumerate(retrieved[:k], 1):
        new = covered(ch, gold) - seen
        if new:
            dcg += 1.0 / math.log2(i + 1)
            seen |= new
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal


def score(retrieved: Retrieved, gold: Gold) -> dict[str, float]:
    return {
        "recall@1": recall_at_k(retrieved, gold, 1),
        "recall@5": recall_at_k(retrieved, gold, 5),
        "recall@10": recall_at_k(retrieved, gold, 10),
        "hit@10": hit_at_k(retrieved, gold, 10),
        "mrr": mrr(retrieved, gold),
        "ndcg@10": ndcg_at_k(retrieved, gold, 10),
        # diagnostic: right document anywhere in the top 10, regardless of page
        "doc_hit@10": 1.0 if any(d in {g[0] for g in gold} for d, _ in retrieved[:10]) else 0.0,
    }


METRICS = ["recall@1", "recall@5", "recall@10", "hit@10", "mrr", "ndcg@10", "doc_hit@10"]


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    n = len(rows)
    return {m: (sum(r[m] for r in rows) / n if n else 0.0) for m in METRICS} | {"n": n}
