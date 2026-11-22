"""Generation flow + judge parsing, with a fake LLM client (no network)."""
from eval.judge import judge, parse_verdict
from footnote.generate import generate
from footnote.retrieve import Hit

A = "3M_2018_10K::p0060::00"


def hit(cid=A):
    return Hit(cid, "3M_2018_10K", "3M", "10-K", 2018, [60], "Cash flows", "table", "| Capex | (1,577) |", 1.0)


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, model, messages, **params):
        self.calls.append(messages)
        return {"content": self.replies.pop(0), "usage": {"prompt_tokens": 1, "completion_tokens": 1}, "cached": False}


def test_valid_first_answer_is_kept():
    c = FakeClient([f"Capex was $1,577 million [{A}]."])
    g = generate(c, "q", [hit()], "m")
    assert g.status == "answered" and g.attempts == 1 and len(c.calls) == 1


def test_invalid_citation_regenerates_once_then_accepts():
    c = FakeClient(["Capex was $1,577 million [3M_2018_10K::p0099::00].", f"Capex was $1,577 million [{A}]."])
    g = generate(c, "q", [hit()], "m")
    assert g.status == "answered" and g.attempts == 2
    assert "failed citation validation" in c.calls[1][-1]["content"]  # complaint sent back
    assert A in c.calls[1][-1]["content"]                             # with the valid ids


def test_invalid_twice_refuses():
    c = FakeClient(["Capex was $1,577 million.", "Capex was $1,577 million [BAD::p0001::00]."])
    g = generate(c, "q", [hit()], "m")
    assert g.status == "refused_invalid_citations" and g.answer.startswith("INSUFFICIENT_CONTEXT")
    assert len(c.calls) == 2  # never a third attempt


def test_model_refusal_is_passed_through():
    g = generate(FakeClient(["INSUFFICIENT_CONTEXT: no cash flow data."]), "q", [hit()], "m")
    assert g.status == "refused"


def test_judge_skips_llm_for_refusals():
    c = FakeClient([])
    assert judge(c, "j", "q", "ref", None, "INSUFFICIENT_CONTEXT: nothing")["verdict"] == "refused"
    assert c.calls == []


def test_judge_strips_citations_before_grading():
    c = FakeClient(['{"reasoning": "matches", "verdict": "correct"}'])
    out = judge(c, "j", "q", "$1577.00", None, f"Capex was $1,577 million [{A}].")
    assert out["verdict"] == "correct"
    assert "::p0060::" not in c.calls[0][-1]["content"]


def test_parse_verdict_variants():
    assert parse_verdict('{"verdict": "Partially correct", "reasoning": "x"}')[0] == "partially"
    assert parse_verdict('{"verdict": "INCORRECT"}')[0] == "incorrect"
    assert parse_verdict('garbage "verdict": "correct" more')[0] == "correct"
    v, r = parse_verdict("not json at all")
    assert v == "incorrect" and r.startswith("[unparseable")
