"""Regression tests for bugs found in the Phase 1 review."""
from footnote.chunking import Element, chunk_elements
from footnote.parse import Row, build_rows, rows_to_markdown


def wc(s: str) -> int:
    return len(s.split())


def w(x0, x1, text, y=10):
    return (x0, y, x1, y + 9, text)


def test_overlap_tail_never_pushes_chunk_over_size():
    lines = ["a b c d"] * 5 + [" ".join(["x"] * 20)]
    chunks = chunk_elements([Element("text", "\n".join(lines), 1)], chunk_size=20, count=wc, overlap=8)
    assert all(c.n_tokens <= 20 for c in chunks), [c.n_tokens for c in chunks]


def test_heading_never_alone_in_a_chunk():
    els = [Element("text", " ".join(["p"] * 15), 1), Element("heading", "Liquidity", 1),
           Element("text", " ".join(["y"] * 20), 1)]
    chunks = chunk_elements(els, chunk_size=20, count=wc, overlap=0)
    assert not any(c.text.strip() == "Liquidity" for c in chunks)
    assert any(c.text.startswith("Liquidity\n") for c in chunks)


def test_caption_moved_into_table_chunk_not_orphaned():
    table = "| | 2023 |\n| --- | --- |\n" + "\n".join(f"| row {i} | {i} |" for i in range(6))
    els = [Element("text", " ".join(["p"] * 15) + ".", 1), Element("text", "Years ended December 31", 1),
           Element("table", table, 1)]
    chunks = chunk_elements(els, chunk_size=30, count=wc, overlap=0)
    assert [c.text.count("Years ended December 31") for c in chunks].count(1) == 1
    assert not any(c.text == "Years ended December 31" for c in chunks)
    assert any(c.chunk_type == "table" and c.text.startswith("Years ended December 31\n|") for c in chunks)


def test_previous_page_footer_is_not_a_caption():
    table = "| | 2023 |\n| --- | --- |\n" + "\n".join(f"| row {i} | {i} |" for i in range(6))
    els = [Element("text", "Last prose sentence.\n59", 59), Element("heading", "Statement of Cash Flows", 60),
           Element("table", table, 60)]
    chunks = chunk_elements(els, chunk_size=45, count=wc, overlap=0)  # table must start a new chunk
    t = next(c for c in chunks if c.chunk_type == "table")
    assert t.pages == [60] and t.text.startswith("Statement of Cash Flows\n|")


def test_currency_of_next_column_starts_new_cell():
    # real layout from 3M_2022_10K p.24: "25.5%" then "$" 5pt later, then "1,701"
    rows = build_rows([w(25, 52, "Margin"), w(175, 192, "25.5%"), w(197, 200, "$"), w(223, 238, "1,701"),
                       w(268, 285, "20.2%")])
    assert [c[2] for c in rows[0].cells] == ["Margin", "25.5%", "$1,701", "20.2%"]


def test_number_after_number_starts_new_cell():
    rows = build_rows([w(20, 60, "Total"), w(300, 320, "1,234"), w(324, 344, "5,678")])
    assert [c[2] for c in rows[0].cells] == ["Total", "1,234", "5,678"]


def test_colliding_cells_keep_order_in_free_columns():
    head = Row(0, 9, [(390, 400, "2023"), (470, 480, "2022"), (550, 560, "2021")])
    data = Row(10, 19, [(20, 60, "A"), (380, 400, "1"), (460, 480, "2"), (540, 560, "3")])
    # both "4" and "5" snap to the 480 column; col 400 is free on the left, col 560 taken
    squeezed = Row(20, 29, [(20, 60, "B"), (455, 470, "4"), (471, 482, "5"), (540, 560, "6")])
    md = rows_to_markdown([head, data, squeezed]).splitlines()
    assert md[-1] == "| B | 4 | 5 | 6 |"
    # right side free: overflow moves right
    squeezed2 = Row(20, 29, [(20, 60, "C"), (380, 400, "7"), (455, 470, "8"), (471, 482, "9")])
    md = rows_to_markdown([head, data, squeezed2]).splitlines()
    assert md[-1] == "| C | 7 | 8 | 9 |"


def test_data_row_is_not_used_as_header():
    rows = [Row(0, 9, [(20, 60, "Revenue"), (380, 400, "1,000"), (460, 480, "900")]),
            Row(10, 19, [(20, 60, "Costs"), (380, 400, "(500)"), (460, 480, "(450)")])]
    md = rows_to_markdown(rows).splitlines()
    assert md[0].replace("|", "").strip() == ""  # blank header
    assert md[2].startswith("| Revenue | 1,000 |")


def test_year_row_is_header():
    rows = [Row(0, 9, [(390, 400, "2023"), (470, 480, "2022")]),
            Row(10, 19, [(20, 60, "Revenue"), (380, 400, "1,000"), (460, 480, "900")])]
    md = rows_to_markdown(rows).splitlines()
    assert md[0] == "|  | 2023 | 2022 |"
