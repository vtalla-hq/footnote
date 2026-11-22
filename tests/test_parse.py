"""Layout table detector on synthetic pymupdf-style words (no PDFs needed)."""
from footnote.chunking import chunk_elements
from footnote.parse import build_rows, find_layout_tables, rows_to_markdown


def w(x0, y, text, width=None):
    width = width or 5.0 * len(text)
    return (x0, y, x0 + width, y + 9, text)


def statement_words():
    words = [w(20, 10, "Consolidated"), w(85, 10, "Statement"), w(140, 10, "of"), w(152, 10, "Cash"), w(178, 10, "Flows")]
    # centred header years
    words += [w(395, 30, "2018"), w(465, 30, "2017"), w(535, 30, "2016")]
    y = 45
    for label, vals in [("Net income", ["$5,363", "$4,869", "$5,058"]),
                        ("Depreciation", ["1,488", "1,544", "1,474"]),
                        ("Purchases of PP&E", ["(1,577)", "(1,373)", "(1,420)"]),
                        ("Dividends paid", ["(3,193)", "(2,803)", "(2,678)"])]:
        x = 20
        for tok in label.split():
            words.append(w(x, y, tok))
            x += 5 * len(tok) + 3
        for right, v in zip((420, 490, 560), vals):
            words.append(w(right - 5 * len(v), y, v))
        y += 15
    # a prose paragraph after the table
    words += [w(20, y + 20, "The"), w(40, y + 20, "accompanying"), w(108, y + 20, "notes"),
              w(140, y + 20, "are"), w(160, y + 20, "an"), w(175, y + 20, "integral"),
              w(220, y + 20, "part"), w(245, y + 20, "of"), w(260, y + 20, "this"), w(285, y + 20, "statement.")]
    return words


def test_layout_table_detected_with_aligned_columns():
    rows = build_rows(statement_words())
    regions = find_layout_tables(rows)
    assert len(regions) == 1
    s, e = regions[0]
    md = rows_to_markdown(rows[s:e] if rows[s].n_num else rows[s + 1:e])
    lines = md.splitlines()
    assert lines[0] == "|  | 2018 | 2017 | 2016 |"  # centred years land in the right columns
    assert "| Purchases of PP&E | (1,577) | (1,373) | (1,420) |" in lines
    assert "| Net income | $5,363 | $4,869 | $5,058 |" in lines
    assert all("accompanying" not in ln for ln in lines)  # prose stays out


def test_currency_sign_glued_to_number():
    rows = build_rows([w(20, 10, "Revenue"), w(380, 10, "$", 5), w(420, 10, "1,234", 25)])
    assert [c[2] for c in rows[0].cells] == ["Revenue", "$1,234"]


def test_parsed_table_survives_chunking_whole():
    from footnote.chunking import Element

    rows = build_rows(statement_words())
    s, e = find_layout_tables(rows)[0]
    md = rows_to_markdown(rows[s + 1:e])
    els = [Element("text", "Consolidated Statement of Cash Flows", 60), Element("table", md, 60)]
    chunks = chunk_elements(els, chunk_size=16, count=lambda t: len(t.split()), table_max=512)
    tables = [c for c in chunks if c.chunk_type == "table"]
    assert len(tables) == 1 and "(1,577)" in tables[0].text and "Dividends paid" in tables[0].text
