from footnote.citations import extract_citations, is_refusal, strip_citations, validate

A = "3M_2018_10K::p0060::00"
B = "3M_2018_10K::p0061::01"
RETRIEVED = [A, B]


def test_valid_answer_with_citations_per_claim():
    ans = f"3M's FY2018 capital expenditure was $1,577 million [{A}]. That is up from $1,373 million in 2017 [{B}]."
    v = validate(ans, RETRIEVED)
    assert v.valid and v.cited == [A, B] and not v.unknown_ids


def test_citation_after_full_stop_counts_for_that_sentence():
    ans = f"Capex was $1,577 million. [{A}]\nIt rose 15%. [{A}][{B}]"
    assert validate(ans, RETRIEVED).valid


def test_hallucinated_chunk_id_is_invalid():
    ans = f"Capex was $1,577 million [3M_2018_10K::p0099::00]."
    v = validate(ans, RETRIEVED)
    assert not v.valid and v.unknown_ids == ["3M_2018_10K::p0099::00"]


def test_numeric_claim_without_citation_is_invalid():
    ans = f"Capex was $1,577 million [{A}]. Free cash flow was $4,862 million."
    v = validate(ans, RETRIEVED)
    assert not v.valid and len(v.uncited_claims) == 1 and "4,862" in v.uncited_claims[0]


def test_no_citations_is_invalid():
    v = validate("The company is doing well overall.", RETRIEVED)
    assert not v.valid and v.reason == "no citations"


def test_refusal_is_valid_without_citations():
    v = validate("INSUFFICIENT_CONTEXT: the cash flow statement is not in the context.", RETRIEVED)
    assert v.valid and v.refusal and is_refusal("insufficient_context: x")


def test_multiple_ids_in_one_bracket_and_dedup():
    assert extract_citations(f"x [{A}, {B}] y [{A}]") == [A, B]


def test_ids_with_ampersand_and_dashes():
    cid = "JOHNSON_JOHNSON_2023_8K_dated-2023-08-30::p0003::00"
    assert extract_citations(f"Gain was $21.0 billion [{cid}].") == [cid]
    assert validate(f"Gain was $21.0 billion [{cid}].", [cid]).valid


def test_decimal_numbers_do_not_split_sentences():
    ans = f"EPS was 1.5 dollars and margin 20.2% [{A}]."
    assert validate(ans, RETRIEVED).valid


def test_strip_citations_for_judge():
    assert strip_citations(f"Capex was $1,577 million [{A}]. Up [{A}, {B}].") == "Capex was $1,577 million. Up."


def test_fullwidth_brackets_from_gpt_oss_are_accepted():
    ans = f"Restructuring cost was $0【{A}】. It rose 3%【{A}】【{B}】."
    v = validate(ans, RETRIEVED)
    assert v.valid and v.cited == [A, B]
    assert strip_citations(ans) == "Restructuring cost was $0. It rose 3%."


def test_years_and_period_labels_are_not_claims():
    ans = f"The quick ratio for Q2 FY 2023 cannot be computed from these excerpts. Revenue was $5 billion [{A}]."
    assert validate(ans, RETRIEVED).valid
    assert validate(f"In 2022 the company changed its CFO [{A}]. FY2023 guidance was withdrawn.", RETRIEVED).valid


def test_quantities_still_need_citations():
    for claim in ["Margin was 20.2%.", "Capex was $1,577 million.", "It had 4,500 employees.", "Ratio was 0.96."]:
        assert not validate(f"Intro sentence [{A}]. {claim}", RETRIEVED).valid, claim


def test_item_and_note_numbers_are_labels():
    assert validate(f"The 8-K was filed under Item 8.01 Other Events [{A}]. See Note 12 for details.", RETRIEVED).valid
