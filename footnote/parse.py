"""PDF -> ordered Elements (text / heading / table), page by page.

SEC financial statements are mostly *rule-less* tables, which pdfplumber's line-based finder
misses (verified on 3M_2018_10K p.60, the cash-flow statement). So tables come from two sources:

1. pdfplumber `find_tables` (lines strategy) for ruled tables.
2. A layout detector on pymupdf words: rebuild visual rows by y, split each row into cells on
   wide horizontal gaps, and mark runs of rows that carry numeric cells as a table region.
   Columns are anchored on the right edges of numeric cells (numbers are right-aligned).

Everything outside table regions becomes prose; bold / large / "Item N." lines become headings.
"""
from __future__ import annotations

import json
import os
import re
import statistics
from pathlib import Path

import fitz  # pymupdf
import pdfplumber

from footnote.chunking import Element

ROOT = Path(__file__).resolve().parent.parent
PARSE_CACHE = ROOT / "cache" / "parsed"
PARSER_VERSION = "v6"

NUM_RE = re.compile(r"^[($€£]*-?\$?\d[\d,]*(\.\d+)?%?\)?%?\*?$|^\$?[—–-]+$|^n/?m$|^N/?M$")
ITEM_RE = re.compile(r"^(item|ITEM|Item)\s+\d{1,2}[A-Za-z]?\b")
PART_RE = re.compile(r"^PART\s+[IV]+\b")
NAV_LINES = {"table of contents", "index to financial statements"}
YEAR_RE = re.compile(r"^(19|20)\d\d$")
CURRENCY = {"$", "€", "£", "$(", "($"}
ROW_TOL = 3.0      # pt: words whose y-centres are this close are on one visual row
CELL_GAP = 7.0     # pt: a horizontal gap wider than this starts a new cell
COL_TOL = 12.0     # pt: numeric right edges this close belong to one column
LINE_SETTINGS = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}


def is_num(s: str) -> bool:
    return bool(NUM_RE.match(s.replace(" ", "")))


# ---------------------------------------------------------------- rows & cells
class Row:
    __slots__ = ("y0", "y1", "cells", "size", "bold")

    def __init__(self, y0, y1, cells, size=0.0, bold=False):
        self.y0, self.y1, self.cells, self.size, self.bold = y0, y1, cells, size, bold

    @property
    def text(self) -> str:
        return " ".join(c[2] for c in self.cells)

    @property
    def n_num(self) -> int:
        return sum(is_num(c[2]) for c in self.cells)


def build_rows(words: list[tuple]) -> list[Row]:
    """words: pymupdf (x0, y0, x1, y1, text, ...). Returns rows of cells (x0, x1, text)."""
    words = sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    groups: list[list[tuple]] = []
    for w in words:
        yc = (w[1] + w[3]) / 2
        if groups and abs(yc - groups[-1][0]) <= ROW_TOL:
            groups[-1][1].append(w)
        else:
            groups.append([yc, [w]])
    rows = []
    for _, ws in groups:
        ws.sort(key=lambda w: w[0])
        cells: list[list] = []
        for w in ws:
            # a currency sign, or a number following a number, always starts a new cell: column
            # values can sit closer than CELL_GAP ("25.5%  $ 1,701" -> "25.5%" | "$1,701")
            prev_last = cells[-1][2].split()[-1] if cells else ""
            breaks = w[4] in CURRENCY or (is_num(w[4]) and is_num(prev_last))
            if cells and not breaks and w[0] - cells[-1][1] <= CELL_GAP:
                cells[-1][1] = w[2]
                cells[-1][2] += " " + w[4]
            else:
                cells.append([w[0], w[2], w[4]])
        merged: list[list] = []  # glue lone "$" / "(" onto the number that follows
        for c in cells:
            if merged and merged[-1][2] in CURRENCY and is_num(c[2]):
                merged[-1][1] = c[1]
                merged[-1][2] += c[2]
            elif merged and c[2] in {"%", ")", "%)"} and is_num(merged[-1][2] + c[2]):
                merged[-1][1] = c[1]
                merged[-1][2] += c[2]
            else:
                merged.append(c)
        rows.append(Row(min(w[1] for w in ws), max(w[3] for w in ws), [tuple(c) for c in merged]))
    return rows


def _row_kind(r: Row) -> str:
    """T = tabular (label + numeric cells), L = short label line, O = other (prose etc.)."""
    n = len(r.cells)
    if n >= 2 and r.n_num >= 1 and (r.n_num >= 2 or not is_num(r.cells[0][2])):
        if r.n_num >= n - 1 or r.n_num >= 2:
            return "T"
    if n >= 3 and r.n_num == 0 and max(len(c[2]) for c in r.cells) <= 40:
        return "L"  # multi-cell text row: column headers like "Three Months Ended | Nine Months Ended"
    if len(r.text) <= 90 and not r.text.rstrip().endswith((".", ";", ":")):
        return "L"
    return "O"


def find_layout_tables(rows: list[Row]) -> list[tuple[int, int]]:
    """Return [start, end) row-index ranges that form tables."""
    kinds = [_row_kind(r) for r in rows]
    regions, i = [], 0
    while i < len(rows):
        if kinds[i] != "T":
            i += 1
            continue
        start = i
        while start > 0 and kinds[start - 1] == "L" and i - start < 3:  # pull in header lines
            start -= 1
        j, gap, last_t = i, 0, i
        while j < len(rows) and kinds[j] in "TL":
            if kinds[j] == "T":
                last_t, gap = j, 0
            else:
                gap += 1
                if gap > 3:
                    break
            j += 1
        end = last_t + 1
        if sum(k == "T" for k in kinds[start:end]) >= 3:
            # drop leading label rows that are really prose captions (kept as caption by chunker)
            while start < i and len(rows[start].cells) == 1 and len(rows[start].text) > 60:
                start += 1
            regions.append((start, end))
        i = end
    return regions


def _assign_columns(cells: list[tuple[float, str]], cols: list[float]) -> list[tuple[int, str]]:
    """Map a row's value cells (left->right, by right edge) to columns.

    Order-preserving and one value per column: pick strictly increasing column indices that
    minimise total |right_edge - column_anchor| (small DP). If a row has more values than columns,
    fall back to nearest-column (values sharing a column get joined).
    """
    n, m = len(cells), len(cols)
    if n == 0:
        return []
    if n > m:
        return [(min(range(m), key=lambda k: abs(cols[k] - x)), t) for x, t in cells]
    inf = float("inf")
    cost = [[inf] * m for _ in range(n)]  # cost[i][j]: best for cells[0..i] with cell i in col j
    back = [[-1] * m for _ in range(n)]
    for j in range(m):
        cost[0][j] = abs(cols[j] - cells[0][0])
    for i in range(1, n):
        best, arg = inf, -1
        for j in range(m):
            if j - 1 >= 0 and cost[i - 1][j - 1] < best:
                best, arg = cost[i - 1][j - 1], j - 1
            if arg >= 0:
                cost[i][j] = best + abs(cols[j] - cells[i][0])
                back[i][j] = arg
    j = min(range(m), key=lambda j: cost[n - 1][j])
    out = []
    for i in range(n - 1, -1, -1):
        out.append((j, cells[i][1]))
        j = back[i][j]
    return out[::-1]


def rows_to_markdown(rows: list[Row]) -> str:
    # anchor columns on data rows only: header years ("2018  2017") are centred, not right-aligned
    data = [r for r in rows if r.n_num and not all(YEAR_RE.match(c[2]) for c in r.cells if is_num(c[2]))]
    rights = sorted(c[1] for r in (data or rows) for c in r.cells if is_num(c[2]))
    anchors: list[list[float]] = []
    for x in rights:
        if anchors and x - anchors[-1][-1] <= COL_TOL:
            anchors[-1].append(x)
        else:
            anchors.append([x])
    cols = [statistics.median(a) for a in anchors]
    if not cols:
        return ""
    label_edge = min(min(c[0] for c in r.cells if is_num(c[2])) for r in rows if r.n_num) - 1
    table = []
    for r in rows:
        label, cells = [], []
        for x0, x1, t in r.cells:
            if (x1 <= label_edge or x1 < cols[0] - 3 * COL_TOL) and not is_num(t) and not cells:
                label.append(t)
            else:
                cells.append((x1, t))
        vals = [""] * len(cols)
        for k, t in _assign_columns(cells, cols):
            vals[k] = f"{vals[k]} {t}".strip()
        table.append([" ".join(label)] + vals)
    esc = lambda s: s.replace("|", "\\|")
    # the first row is a header only if it carries no real data (just years / column titles);
    # otherwise use a blank header so a data row isn't repeated as "header" when the table splits
    if any(is_num(c[2]) and not YEAR_RE.match(c[2]) for c in rows[0].cells):
        head, body = [""] * len(table[0]), table
    else:
        head, body = table[0], table[1:]
    out = ["| " + " | ".join(esc(c) for c in head) + " |", "|" + " --- |" * len(head)]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in body]
    return "\n".join(out)


# ---------------------------------------------------------------- pdfplumber (ruled tables)
def _clean_cell(c) -> str:
    return re.sub(r"\s+", " ", str(c)).strip() if c is not None else ""


def plumber_table_to_markdown(rows: list[list]) -> str | None:
    rows = [[_clean_cell(c) for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if len(rows) < 2:
        return None
    for r in rows:
        for j in range(len(r) - 1):
            if r[j] in CURRENCY and r[j + 1]:
                r[j], r[j + 1] = "", r[j] + r[j + 1]
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    keep = [j for j in range(width) if any(r[j] for r in rows)]
    rows = [[r[j] for j in keep] for r in rows]
    cells = [c for r in rows for c in r if c]
    if len(keep) < 2 or len(cells) < 4:
        return None
    esc = lambda s: s.replace("|", "\\|")
    out = ["| " + " | ".join(esc(c) for c in rows[0]) + " |", "|" + " --- |" * len(keep)]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _plumber_tables(ppage) -> list[tuple[tuple, str]]:
    if ppage is None:
        return []
    try:
        found = ppage.find_tables(LINE_SETTINGS)
    except Exception:  # noqa: BLE001 - pdfplumber can choke on odd pages
        return []
    out = []
    for t in found:
        md = plumber_table_to_markdown(t.extract())
        if md:
            out.append((t.bbox, md))
    return out


# ---------------------------------------------------------------- page assembly
def _inside(bbox, boxes) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return any(b[0] - 2 <= cx <= b[2] + 2 and b[1] - 2 <= cy <= b[3] + 2 for b in boxes)


def _font_info(fpage) -> list[tuple[float, float, float, bool]]:
    spans = []
    for b in fpage.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                if s["text"].strip():
                    bold = bool(s["flags"] & 16) or "bold" in s["font"].lower()
                    spans.append(((s["bbox"][1] + s["bbox"][3]) / 2, s["bbox"][0], s["size"], bold))
    return spans


def _overlap(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    area = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return (x1 - x0) * (y1 - y0) / area if area > 0 else 0.0


def parse_page(fpage, ppage, page_no: int) -> list[Element]:
    words = [w for w in fpage.get_text("words") if w[4].strip()]
    rows = [r for r in build_rows(words) if r.text.strip().lower() not in NAV_LINES]
    spans = _font_info(fpage)
    body = statistics.median([s[2] for s in spans]) if spans else 10.0
    for r in rows:
        mine = [s for s in spans if r.y0 - 1 <= s[0] <= r.y1 + 1]
        r.size = max((s[2] for s in mine), default=body)
        r.bold = bool(mine) and all(s[3] for s in mine)

    # 1) layout tables (primary: handles rule-less financial statements)
    items: list[tuple[float, Element]] = []
    in_table: set[int] = set()
    layout_boxes = []
    for s, e in []:  # DEMO: deliberately disable layout table detection (should fail the CI gate)
        first_t = next(k for k in range(s, e) if _row_kind(rows[k]) == "T")
        md = rows_to_markdown(rows[first_t:e])
        if md:
            # title rows above the header stay prose; the chunker attaches them as the caption
            items.append((rows[first_t].y0, Element("table", md, page_no)))
            in_table.update(range(first_t, e))
            xs = [c for r in rows[first_t:e] for c in r.cells]
            layout_boxes.append((min(c[0] for c in xs), rows[first_t].y0, max(c[1] for c in xs), rows[e - 1].y1))
    # 2) ruled tables from pdfplumber that the layout pass did not already cover (e.g. text grids)
    for bbox, md in _plumber_tables(ppage):
        if any(_overlap(bbox, b) > 0.2 for b in layout_boxes):
            continue
        idx = [i for i, r in enumerate(rows) if i not in in_table
               and _inside((r.cells[0][0], r.y0, r.cells[-1][1], r.y1), [bbox])]
        if not idx:
            continue
        items.append((bbox[1], Element("table", md, page_no)))
        in_table.update(idx)

    for i, r in enumerate(rows):
        if i in in_table:
            continue
        t = r.text.strip()
        heading = (len(t) <= 100 and bool(ITEM_RE.match(t) or PART_RE.match(t))) or (
            4 <= len(t) <= 100 and any(ch.isalpha() for ch in t) and not t.endswith((".", ",", ";"))
            and (r.size >= body * 1.15 or r.bold))
        items.append((r.y0, Element("heading" if heading else "text", t, page_no)))
    items.sort(key=lambda x: x[0])
    out: list[Element] = []
    for _, el in items:
        if (el.kind == "heading" and out and out[-1].kind == "heading"
                and len(out[-1].text) + len(el.text) < 100 and not ITEM_RE.match(el.text)):
            out[-1].text += " " + el.text  # multi-line headings (TOC lines stay separate)
        elif el.kind == "text" and out and out[-1].kind == "text":
            out[-1].text += "\n" + el.text
        else:
            out.append(el)
    return out


def _cache_key() -> str:
    """Parser version + hash of this file, chunking.Element and library versions."""
    import hashlib
    import inspect

    from footnote import chunking

    src = Path(__file__).read_bytes() + inspect.getsource(chunking.Element).encode()
    src += f"{fitz.VersionBind}|{pdfplumber.__version__}".encode()
    return f"{PARSER_VERSION}-{hashlib.sha256(src).hexdigest()[:10]}"


def parse_pdf(path: Path, use_cache: bool = True) -> list[Element]:
    cache = PARSE_CACHE / f"{path.stem}.{_cache_key()}.json"
    if use_cache and cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return [Element(**e) for e in json.loads(cache.read_text(encoding="utf-8"))]
    elements: list[Element] = []
    with fitz.open(path) as fdoc, pdfplumber.open(path) as pdoc:
        for i, fpage in enumerate(fdoc):
            ppage = pdoc.pages[i] if i < len(pdoc.pages) else None
            try:
                elements += parse_page(fpage, ppage, i + 1)
            finally:
                if ppage is not None:
                    ppage.flush_cache()
    PARSE_CACHE.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps([e.__dict__ for e in elements]), encoding="utf-8")
    os.replace(tmp, cache)  # atomic: concurrent ingests never read a half-written file
    return elements


def page_count(path: Path) -> int:
    with fitz.open(path) as d:
        return d.page_count
