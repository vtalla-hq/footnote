from footnote.chunking import Element, chunk_elements, split_table


def wc(s: str) -> int:  # deterministic stand-in for the real tokenizer
    return len(s.split())


def make_table(n_rows: int) -> str:
    lines = ["| Item | 2023 | 2022 |", "| --- | --- | --- |"]
    lines += [f"| Revenue line {i} | {i * 10} | {i * 9} |" for i in range(n_rows)]
    return "\n".join(lines)


def rows_of(md: str) -> list[str]:
    return [ln for ln in md.splitlines() if ln.startswith("| Revenue line")]


def test_small_table_is_one_chunk_and_never_split():
    els = [
        Element("text", "Some prose before the table. " * 3, page=4),
        Element("text", "CONSOLIDATED STATEMENTS OF INCOME", page=4),
        Element("table", make_table(5), page=4),
        Element("text", "Prose after the table.", page=4),
    ]
    for size in (40, 60, 200):  # table doesn't fit / fits after flush / packs with prose
        chunks = chunk_elements(els, chunk_size=size, count=wc, overlap=0)
        holders = [c for c in chunks if rows_of(c.text)]
        assert len(holders) == 1, size  # every row lives in exactly one chunk
        t = holders[0]
        assert rows_of(t.text) == rows_of(make_table(5))
        assert t.chunk_type == "table" and t.pages == [4]
        assert "CONSOLIDATED STATEMENTS OF INCOME\n| Item" in t.text  # caption kept with table
        assert all(c.chunk_type == "text" for c in chunks if c is not t)


def test_table_bigger_than_chunk_is_own_chunk():
    els = [Element("text", "intro line here", 1), Element("table", make_table(30), 1)]
    chunks = chunk_elements(els, chunk_size=50, count=wc, table_max=512)
    tables = [c for c in chunks if c.chunk_type == "table"]
    assert len(tables) == 1 and len(rows_of(tables[0].text)) == 30
    assert tables[0].text.startswith("intro line here\n| Item")


def test_large_table_split_on_row_boundaries_with_header_repeated():
    md = make_table(60)
    pieces = split_table(md, table_max=60, count=wc)
    assert len(pieces) > 1
    all_rows = []
    for p in pieces:
        lines = p.splitlines()
        assert lines[0] == "| Item | 2023 | 2022 |"
        assert lines[1] == "| --- | --- | --- |"
        for ln in lines[2:]:
            assert ln.startswith("| ") and ln.endswith(" |")  # every row whole
        all_rows += rows_of(p)
    assert all_rows == rows_of(md)  # nothing lost, nothing duplicated, order kept


def test_table_under_limit_not_split_even_if_bigger_than_chunk_size():
    md = make_table(20)  # ~ 20 rows * 9 words = ~190 words
    els = [Element("table", md, page=1)]
    chunks = chunk_elements(els, chunk_size=50, count=wc, table_max=512)
    assert len(chunks) == 1
    assert rows_of(chunks[0].text) == rows_of(md)


def test_giant_single_row_is_not_split():
    md = "| a | b |\n| --- | --- |\n| " + "x " * 300 + "| 1 |"
    pieces = split_table(md, table_max=50, count=wc)
    assert len(pieces) == 1


def test_text_respects_chunk_size_and_tracks_pages_and_sections():
    els = [Element("heading", "Item 7. Management's Discussion", page=10)]
    els += [Element("heading", "Liquidity", page=10)]  # heading on a near-empty chunk: no flush
    els += [Element("text", "\n".join(f"line {p}-{i} with five words" for i in range(10)), page=p)
            for p in (10, 11)]
    chunks = chunk_elements(els, chunk_size=30, count=wc, overlap=0)
    assert all(c.n_tokens <= 30 for c in chunks)
    assert chunks[0].text.startswith("Item 7.")
    assert chunks[0].section == "Item 7. Management's Discussion"  # heading in effect at chunk start
    assert all(c.section == "Liquidity" for c in chunks[1:])
    assert any(c.pages == [10, 11] for c in chunks)  # chunks may cross a page break
    joined = "\n".join(c.text for c in chunks)
    assert joined.count("line 11-9 with five words") == 1


def test_overlap_carries_tail_lines():
    els = [Element("text", "\n".join(f"l{i} a b c d" for i in range(20)), page=1)]
    chunks = chunk_elements(els, chunk_size=20, count=wc, overlap=5)
    assert len(chunks) > 1
    for a, b in zip(chunks, chunks[1:]):
        assert b.text.splitlines()[0] == a.text.splitlines()[-1]
