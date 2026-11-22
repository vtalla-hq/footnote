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
    buf: list[tuple[str, int, int, bool]] = []  # (text, page, tokens, is_table): atomic units

    def flush(keep_overlap: bool) -> None:
        nonlocal buf, buf_section
        if buf:
            text = "\n".join(b[0] for b in buf).strip()
            if text:
                chunks.append(Chunk(text=text, pages=sorted({b[1] for b in buf}), section=buf_section,
                                    chunk_type="table" if any(b[3] for b in buf) else "text",
                                    n_tokens=count(text)))
        tail: list[tuple[str, int, int, bool]] = []
        if keep_overlap and overlap > 0:
            t = 0
            for item in reversed(buf):
                if item[3] or t + item[2] > overlap:  # tables are never carried as overlap
                    break
                tail.insert(0, item)
                t += item[2]
            if len(tail) == len(buf):  # never carry the whole buffer forward
                tail = []
        buf = tail
        buf_section = section

    def buf_tokens() -> int:
        return sum(b[2] for b in buf)

    def add(item: tuple[str, int, int, bool]) -> None:
        nonlocal buf_section
        if not buf:
            buf_section = section
        buf.append(item)

    recent: list[str] = []  # trailing short lines before a table become its caption

    def note(line: str) -> None:
        recent.append(line)
        del recent[:-3]

    for el in elements:
        if el.kind == "heading":
            # a heading starts a new chunk only if the current one is already reasonably full;
            # otherwise tiny heading-bounded chunks would make chunk_size meaningless
            if buf_tokens() >= chunk_size // 2:
                flush(keep_overlap=False)
            section = el.text.strip()[:200]
            n = count(el.text)
            if buf and buf_tokens() + n > chunk_size:
                flush(keep_overlap=False)
            if not buf:
                buf_section = section
            add((el.text.strip(), el.page, n, False))
            note(el.text.strip())
        elif el.kind == "text":
            for line in el.text.splitlines():
                line = line.strip()
                if not line:
                    continue
                n = count(line)
                parts = [line] if n <= chunk_size else _split_long(line, chunk_size, count)
                for p in parts:
                    pn = n if len(parts) == 1 else count(p)
                    if buf and buf_tokens() + pn > chunk_size:
                        flush(keep_overlap=True)
                    add((p, el.page, pn, False))
                note(line)
        elif el.kind == "table":
            cap: list[str] = []
            for ln in reversed(recent):  # up to 3 short lines, nearest first, <= 200 chars total
                if len(ln) > 100 or ln.endswith(".") or sum(map(len, cap)) + len(ln) > 200:
                    break
                cap.insert(0, ln)
            caption = "\n".join(cap)
            for piece in split_table(el.text, table_max, count, el.header_rows):
                pn = count(piece)
                if buf and buf_tokens() + pn <= chunk_size:
                    add((piece, el.page, pn, True))  # whole table/piece packed into current chunk
                    continue
                flush(keep_overlap=False)
                text = f"{caption}\n{piece}" if caption else piece  # caption travels with the table
                tn = count(text)
                if tn <= chunk_size:
                    add((text, el.page, tn, True))
                else:  # bigger than a chunk: its own chunk, still whole
                    chunks.append(Chunk(text=text, pages=[el.page], section=section,
                                        chunk_type="table", n_tokens=tn))
            recent.clear()
        else:
            raise ValueError(f"unknown element kind {el.kind!r}")
    flush(keep_overlap=False)
    return chunks
