"""Hand-computed examples for the retrieval metrics."""
import math

import pytest

from eval.metrics import gold_pages, hit_at_k, mrr, ndcg_at_k, recall_at_k, score

G1 = {("3M_2018_10K", 60)}                              # one gold page
G2 = {("3M_2018_10K", 60), ("3M_2018_10K", 62)}         # two gold pages


def test_gold_pages_converts_zero_based():
    q = {"evidence": [{"doc_name": "A", "evidence_page_num": 59}, {"doc_name": "A", "evidence_page_num": 59}]}
    assert gold_pages(q) == {("A", 60)}


def test_hit_requires_right_doc_and_overlapping_page():
    wrong_doc = [("3M_2017_10K", [60])]
    wrong_page = [("3M_2018_10K", [59])]
    multi_page = [("3M_2018_10K", [59, 60])]
    assert recall_at_k(wrong_doc, G1, 1) == 0
    assert recall_at_k(wrong_page, G1, 1) == 0
    assert recall_at_k(multi_page, G1, 1) == 1


def test_mrr_hand_computed():
    ranked = [("X", [1]), ("X", [2]), ("3M_2018_10K", [60])]
    assert mrr(ranked, G1) == pytest.approx(1 / 3)
    assert mrr([("X", [1])], G1) == 0.0


def test_recall_counts_gold_pages_not_chunks():
    # two chunks on page 60 do not count twice; page 62 found at rank 4
    ranked = [("3M_2018_10K", [60]), ("3M_2018_10K", [60]), ("X", [1]), ("3M_2018_10K", [62])]
    assert recall_at_k(ranked, G2, 1) == 0.5
    assert recall_at_k(ranked, G2, 3) == 0.5
    assert recall_at_k(ranked, G2, 4) == 1.0
    assert hit_at_k(ranked, G2, 1) == 1.0


def test_ndcg_hand_computed():
    # gains at ranks 1 and 4 (rank 2 is a duplicate of page 60 -> no gain)
    ranked = [("3M_2018_10K", [60]), ("3M_2018_10K", [60]), ("X", [1]), ("3M_2018_10K", [62])]
    dcg = 1 / math.log2(2) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(ranked, G2, 10) == pytest.approx(dcg / idcg)  # = 1.4307 / 1.6309 = 0.8772
    assert ndcg_at_k(ranked, G2, 10) == pytest.approx(0.8772, abs=1e-4)
    # single gold page found at rank 3 -> 1/log2(4) = 0.5
    assert ndcg_at_k([("X", [1]), ("X", [2]), ("3M_2018_10K", [60])], G1) == pytest.approx(0.5)
    # beyond k does not count
    assert ndcg_at_k([("X", [1])] * 10 + [("3M_2018_10K", [60])], G1, 10) == 0.0


def test_score_perfect_and_empty():
    s = score([("3M_2018_10K", [60])], G1)
    assert s["recall@1"] == s["mrr"] == s["ndcg@10"] == 1.0
    s = score([], G1)
    assert all(v == 0.0 for v in s.values())


def test_doc_hit_ignores_page():
    s = score([("3M_2018_10K", [5])], G1)
    assert s["doc_hit@10"] == 1.0 and s["recall@10"] == 0.0
