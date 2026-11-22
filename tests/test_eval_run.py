from eval.run import breakdown, config_hash, percentile


def test_percentile_hand_computed():
    xs = [10, 20, 30, 40, 50]
    assert percentile(xs, 50) == 30
    assert percentile(xs, 95) == 48  # 40 + 0.8 * (50 - 40)
    assert percentile([], 50) == 0.0


def test_config_hash_is_order_independent():
    assert config_hash({"a": 1, "b": {"c": 2}}) == config_hash({"b": {"c": 2}, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_breakdown_groups_and_null_reasoning():
    m1 = {k: 1.0 for k in ["recall@1", "recall@5", "recall@10", "hit@10", "mrr", "ndcg@10", "doc_hit@10"]}
    m0 = {k: 0.0 for k in m1}
    rows = [{"question_type": "a", "question_reasoning": None, "doc_type": "10-K", "metrics": m1},
            {"question_type": "a", "question_reasoning": "x", "doc_type": "10-K", "metrics": m0},
            {"question_type": "b", "question_reasoning": "x", "doc_type": "8-K", "metrics": m1}]
    b = breakdown(rows)
    assert b["question_type"]["a"]["recall@10"] == 0.5 and b["question_type"]["a"]["n"] == 2
    assert b["question_reasoning"]["n/a"]["n"] == 1
    assert b["doc_type"]["8-K"]["mrr"] == 1.0
