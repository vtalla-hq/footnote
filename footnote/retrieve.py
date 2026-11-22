"""Retrievers over a footnote.chunks_<size> table.

Config (the `retrieval:` block of an eval config):
    method: dense            # dense | (hybrid, rerank: Phase 3)
    table: chunks_256
    k: 10                    # chunks returned
    ef_search: 100           # HNSW search breadth (pgvector default is 40)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from footnote import db
from footnote.embed import embed_query

COLS = "chunk_id, doc_name, company, doc_type, fiscal_year, pages, section, chunk_type, text"


@dataclass
class Hit:
    chunk_id: str
    doc_name: str
    company: str
    doc_type: str
    fiscal_year: int | None
    pages: list[int]
    section: str | None
    chunk_type: str
    text: str
    score: float


@dataclass
class Result:
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)


class Retriever:
    def __init__(self, cfg: dict, conn=None):
        self.cfg = {"method": "dense", "k": 10, "ef_search": 100} | cfg
        self.table = db.table_name(self.cfg["table"])
        self.conn = conn or db.connect()
        self.conn.execute(f"set hnsw.ef_search = {int(self.cfg['ef_search'])}")
        if self.cfg["method"] != "dense":
            raise NotImplementedError(f"method {self.cfg['method']!r} not implemented yet")

    def search(self, question: str, measure_db: bool = False) -> Result:
        """measure_db: also run EXPLAIN ANALYZE to record server-side execution time, which
        separates DB compute from network round-trips (the client is far from us-east-1)."""
        t0 = time.perf_counter()
        v = embed_query(question)
        t1 = time.perf_counter()
        sql = (f"select {COLS}, 1 - (embedding <=> %s) as score from {self.table} "
               f"order by embedding <=> %s limit %s")
        params = (v, v, int(self.cfg["k"]))
        rows = self.conn.execute(sql, params).fetchall()
        t2 = time.perf_counter()
        hits = [Hit(*r[:9], score=float(r[9])) for r in rows]
        timings = {"embed": (t1 - t0) * 1e3, "search": (t2 - t1) * 1e3, "total": (t2 - t0) * 1e3}
        if measure_db:
            plan = self.conn.execute("explain (analyze, format json) " + sql, params).fetchone()[0]
            timings["db_exec"] = float(plan[0]["Execution Time"])
        return Result(hits, timings)


def network_rtt_ms(conn, n: int = 20) -> dict[str, float]:
    xs = []
    for _ in range(n):
        t = time.perf_counter()
        conn.execute("select 1").fetchone()
        xs.append((time.perf_counter() - t) * 1e3)
    xs.sort()
    return {"p50": round(xs[n // 2], 1), "min": round(xs[0], 1), "max": round(xs[-1], 1)}
