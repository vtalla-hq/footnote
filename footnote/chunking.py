"""Table-aware chunker.

Input is a document as an ordered list of Elements (text / heading / table), each tagged with
its 1-based page number. Rules:

* Text is packed into chunks of ~chunk_size tokens with a small overlap, splitting on line
  boundaries (then words, for a single over-long line).
* A table is an atomic unit: never split mid-row. If it fits in the room left in the current
  chunk it is packed in whole; otherwise it starts a new chunk (with its caption lines). A table
  bigger than table_max tokens is split on row boundaries and every piece repeats the header
  rows, so each piece is still a readable markdown table.
* chunk_type is "table" for any chunk that contains a table (or table piece).
* Headings update the running `section`; they start a new chunk only once the current one is
  at least half full.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

TokenCounter = Callable[[str], int]


@dataclass
class Element:
    kind: str  # "text" | "heading" | "table"
    text: str
    page: int
    header_rows: int = 2  # tables: markdown header line + separator line


@dataclass
class Chunk:
    text: str
    pages: list[int]
    section: str | None
    chunk_type: str  # "text" | "table"
    n_tokens: int = 0
    meta: dict = field(default_factory=dict)


def _split_long(line: str, limit: int, count: TokenCounter) -> list[str]:
    """Split one over-long line on word boundaries into pieces of <= limit tokens."""
    words, out, cur = line.split(), [], []
    for w in words:
        cand = " ".join(cur + [w])
        if cur and count(cand) > limit:
            out.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        out.append(" ".join(cur))
    return out


def split_table(md: str, table_max: int, count: TokenCounter, header_rows: int = 2) -> list[str]:
    """Split a markdown table on row boundaries, repeating the header in every piece."""
    lines = [ln for ln in md.splitlines() if ln.strip()]
    if count(md) <= table_max or len(lines) <= header_rows + 1:
        return [md]
    header, rows = lines[:header_rows], lines[header_rows:]
    pieces, cur = [], []
    for row in rows:
        cand = "\n".join(header + cur + [row])
        if cur and count(cand) > table_max:
            pieces.append("\n".join(header + cur))
            cur = [row]
        else:
            cur.append(row)  # a single giant row stays whole: rows are never split
    if cur:
        pieces.append("\n".join(header + cur))
    return pieces


def chunk_elements(
    elements: list[Element],
    chunk_size: int,
    count: TokenCounter,
    overlap: int | None = None,
    table_max: int | None = None,
) -> list[Chunk]:
    overlap = chunk_size // 8 if overlap is None else overlap
    table_max = max(chunk_size, 512) if table_max is None else table_max
    chunks: list[Chunk] = []
    section: str | None = None
    buf_section: str | None = None  # section in effect where the current chunk starts
    buf: list[tuple[str, int, int, str]] = []  # (text, page, tokens, kind) atomic units

    def tokens(items) -> int:
        return sum(b[2] for b in items)

    def flush(keep_overlap: bool, incoming: int = 0) -> None:
        nonlocal buf, buf_section
        text = "\n".join(b[0] for b in buf).strip()
        if text:
            chunks.append(Chunk(text=text, pages=sorted({b[1] for b in buf}), section=buf_section,
                                chunk_type="table" if any(b[3] == "table" for b in buf) else "text",
                                n_tokens=count(text)))
        tail: list[tuple[str, int, int, str]] = []
        if keep_overlap and overlap > 0:
            for item in reversed(buf):
                if item[3] == "table" or tokens(tail) + item[2] > overlap:  # tables never overlap
                    break
                tail.insert(0, item)
            if len(tail) == len(buf) or tokens(tail) + incoming > chunk_size:
                tail = []  # never carry everything, and never let overlap push past chunk_size
        buf = tail
        buf_section = section

    def add(item: tuple[str, int, int, str]) -> None:
        nonlocal buf_section
        if not buf:
            buf_section = section
        buf.append(item)

    def only_headings() -> bool:
        return bool(buf) and all(b[3] == "heading" for b in buf)

    recent: list[tuple[str, int]] = []  # trailing short lines (with page) -> table caption

    def note(line: str, page: int) -> None:
        recent.append((line, page))
        del recent[:-3]

    for el in elements:
        if el.kind == "heading":
            # a heading starts a new chunk only if the current one is already reasonably full;
            # otherwise tiny heading-bounded chunks would make chunk_size meaningless
            if tokens(buf) >= chunk_size // 2:
                flush(keep_overlap=False)
            section = el.text.strip()[:200]
            n = count(el.text)
            if buf and tokens(buf) + n > chunk_size:
                flush(keep_overlap=False)
            if not buf:
                buf_section = section
            add((el.text.strip(), el.page, n, "heading"))
            note(el.text.strip(), el.page)
        elif el.kind == "text":
            for line in el.text.splitlines():
                line = line.strip()
                if not line:
                    continue
                n = count(line)
                parts = [line] if n <= chunk_size else _split_long(line, chunk_size, count)
                for p in parts:
                    pn = n if len(parts) == 1 else count(p)
                    # a heading is never left alone in a chunk: it may push the chunk over
                    # chunk_size by its own length (headings are <= 100 chars)
                    if buf and tokens(buf) + pn > chunk_size and not only_headings():
                        flush(keep_overlap=True, incoming=pn)
                    add((p, el.page, pn, "text"))
                note(line, el.page)
        elif el.kind == "table":
            cap: list[tuple[str, int]] = []
            for ln, pg in reversed(recent):  # up to 3 short lines, nearest first, <= 200 chars
                if len(ln) > 100 or ln.endswith(".") or sum(len(c[0]) for c in cap) + len(ln) > 200:
                    break
                if not any(ch.isalpha() for ch in ln):  # page-number footer ("59"), not a caption
                    break
                cap.insert(0, (ln, pg))
            for piece in split_table(el.text, table_max, count, el.header_rows):
                pn = count(piece)
                if buf and tokens(buf) + pn <= chunk_size:
                    add((piece, el.page, pn, "table"))  # whole table/piece packed into current chunk
                    continue
                # caption lines travel with the table (and their page counts as the chunk's page).
                # MOVE them out of the pending buffer rather than copy, so they aren't also left
                # behind as a tiny orphan chunk ("Years ended December 31")
                j = 0  # longest suffix of the buffer that equals a suffix of the caption
                while j < len(cap) and j < len(buf) and buf[-1 - j][0] == cap[-1 - j][0]:
                    j += 1
                if j:
                    del buf[-j:]
                flush(keep_overlap=False)
                cap_items = [(ln, pg, count(ln), "text") for ln, pg in cap]
                if tokens(cap_items) + pn <= chunk_size:
                    for c in cap_items:
                        add(c)
                    add((piece, el.page, pn, "table"))
                else:  # bigger than a chunk: its own chunk, still whole
                    text = "\n".join([c[0] for c in cap_items] + [piece])
                    chunks.append(Chunk(text=text, pages=sorted({el.page, *(c[1] for c in cap_items)}),
                                        section=section, chunk_type="table", n_tokens=count(text)))
            recent.clear()
        else:
            raise ValueError(f"unknown element kind {el.kind!r}")
    flush(keep_overlap=False)
    return chunks
