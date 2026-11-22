"""Retrievers over a footnote.chunks_<size> table.

Config (the `retrieval:` block of an eval config):
    method: dense | hybrid
    table: chunks_256
    k: 10                  # chunks returned
    ef_search: 100         # HNSW search breadth (pgvector default 40)
    candidates: 50         # hybrid: depth of each ranked list fed to RRF; rerank: pool size
    rrf_k: 60              # reciprocal rank fusion constant
    rerank: false          # true -> cross-encoder (BAAI/bge-reranker-base) re-scores the pool
    prefilter: none        # none | company: restrict to the company named in the question

Hybrid = dense (cosine over bge-small) + Postgres full-text (ts_rank_cd over an OR-of-lexemes
tsquery against the GIN-indexed tsvector; normalization 1 = rank / (1 + log(length)) to damp
the long-chunk bias), fused with RRF: score = sum 1 / (rrf_k + rank).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from functools import lru_cache

from footnote import db
from footnote.embed import embed_query

COLS = "chunk_id, doc_name, company, doc_type, fiscal_year, pages, section, chunk_type, text"
RERANK_MODEL = "BAAI/bge-reranker-base"


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
    info: dict = field(default_factory=dict)


# ------------------------------------------------------------------ company prefilter
# Aliases seen in FinanceBench questions; matched case-insensitively on word boundaries.
ALIASES = {
    "AMERICANEXPRESS": ["american express", "amex"],
    "JOHNSONJOHNSON": ["johnson & johnson", "johnson and johnson", "j&j", "jnj"],
    "MGMRESORTS": ["mgm resorts", "mgm"],
    "COCACOLA": ["coca-cola", "coca cola", "coke"],
    "JPMORGAN": ["jpmorgan", "jp morgan", "jpm"],
    "AMERICANWATERWORKS": ["american water works", "american water"],
    "ULTABEAUTY": ["ulta beauty", "ulta"],
    "FOOTLOCKER": ["foot locker", "footlocker"],
    "BESTBUY": ["best buy", "bestbuy"],
    "GENERALMILLS": ["general mills"],
    "KRAFTHEINZ": ["kraft heinz", "kraft-heinz", "kraftheinz"],
    "LOCKHEEDMARTIN": ["lockheed martin", "lockheed"],
    "CVSHEALTH": ["cvs health", "cvs"],
    "ACTIVISIONBLIZZARD": ["activision blizzard", "activision"],
    "PEPSICO": ["pepsico", "pepsi"],
    "3M": ["3m"],
    "AESCORPORATION": ["aes corporation", "aes"],
    "BLOCK": ["block", "square"],
}


def company_matcher(companies: list[str]):
    """Build question -> company (doc-name prefix style key) using names + aliases.

    `companies` are the `company` values stored on chunks (e.g. "3M", "American Express").
    Returns the stored company value, or None if zero or several companies match.
    """
    patterns: dict[str, list[re.Pattern]] = {}
    for c in companies:
        key = re.sub(r"[^A-Z0-9]", "", c.upper())
        names = {c.lower()} | set(ALIASES.get(key, []))
        patterns[c] = [re.compile(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])", re.I) for n in names]

    def match(question: str) -> str | None:
        found = {c for c, ps in patterns.items() if any(p.search(question) for p in ps)}
        if len(found) > 1:  # prefer the longest name ("American Water Works" over "American ...")
            best = max(found, key=lambda c: max(len(p.pattern) for p in patterns[c] if p.search(question)))
            return best
        return next(iter(found), None)

    return match


@lru_cache(maxsize=1)
def get_reranker():
    import torch
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL, max_length=512, device="cuda" if torch.cuda.is_available() else "cpu")


def chunk_header(h: Hit) -> str:
    head = f"{h.company} {h.doc_type} FY{h.fiscal_year}"
    return f"{head} | {h.section}" if h.section else head


class Retriever:
    def __init__(self, cfg: dict, conn=None):
        self.cfg = {"method": "dense", "k": 10, "ef_search": 100, "candidates": 50, "rrf_k": 60,
                    "rerank": False, "prefilter": "none", "lexical": "or_all",
                    "max_df": 0.10, "max_terms": 8} | cfg
        if self.cfg["method"] not in ("dense", "hybrid", "lexical"):
            raise ValueError(f"unknown method {self.cfg['method']!r}")
        if self.cfg["lexical"] not in ("or_all", "idf", "bm25"):
            raise ValueError(f"unknown lexical {self.cfg['lexical']!r}")
        self.table = db.table_name(self.cfg["table"])
        self.conn = conn or db.connect()
        if self.cfg["method"] in ("hybrid", "lexical") and self.cfg["lexical"] in ("idf", "bm25"):
            self._ensure_lexdf()
        self.conn.execute(f"set hnsw.ef_search = {int(self.cfg['ef_search'])}")
        self.match_company = None
        if self.cfg["prefilter"] == "company":
            # filtered HNSW scans can return < k rows; iterative scan keeps searching (pgvector 0.8)
            self.conn.execute("set hnsw.iterative_scan = relaxed_order")
            companies = [r[0] for r in self.conn.execute(f"select distinct company from {self.table}")]
            self.match_company = company_matcher(companies)
        elif self.cfg["prefilter"] != "none":
            raise ValueError(f"unknown prefilter {self.cfg['prefilter']!r}")
        if self.cfg["rerank"]:
            get_reranker()

    # -------------------------------------------------------------- SQL
    def _sql(self, n: int, company: str | None) -> tuple[str, dict]:
        where = "where company = %(company)s" if company else ""
        and_where = "and company = %(company)s" if company else ""
        if self.cfg["method"] == "dense":
            sql = (f"select {COLS}, 1 - (embedding <=> %(v)s) as score from {self.table} {where} "
                   f"order by embedding <=> %(v)s limit %(n)s")
        elif self.cfg["method"] == "lexical":  # diagnostic: the keyword list on its own
            sql = f"""with {self._tsq_cte()}, lexr as ({self._lex_ranked(and_where)})
            select {', '.join('t.' + c.strip() for c in COLS.split(','))}, lexr.rk as score
            from lexr join {self.table} t using (chunk_id) order by lexr.rk desc"""
        else:
            sql = f"""
            with dense as (
                select chunk_id, row_number() over (order by d) as r from (
                    select chunk_id, embedding <=> %(v)s as d from {self.table} {where}
                    order by embedding <=> %(v)s limit %(n)s) s),
            {self._tsq_cte()},
            lex as (
                select chunk_id, row_number() over (order by rk desc) as r from (
                    {self._lex_ranked(and_where)}) s),
            fused as (
                select chunk_id, sum(1.0 / (%(rrf_k)s + r)) as s,
                       min(r) filter (where src = 'd') as dr, min(r) filter (where src = 'l') as lr
                from (select chunk_id, r, 'd' as src from dense
                      union all select chunk_id, r, 'l' from lex) u
                group by chunk_id)
            select {', '.join('t.' + c.strip() for c in COLS.split(','))}, f.s as score
            from fused f join {self.table} t using (chunk_id)
            order by f.s desc limit %(n)s"""
        return sql, {"company": company, "n": n, "rrf_k": int(self.cfg["rrf_k"]),
                     "max_df": float(self.cfg["max_df"]), "max_terms": int(self.cfg["max_terms"])}

    def _lex_ranked(self, and_where: str) -> str:
        """Top-n (chunk_id, rk) by the keyword scorer."""
        if self.cfg["lexical"] != "bm25":
            return (f"select chunk_id, ts_rank_cd(tsv, tsq.q, 1) as rk from {self.table}, tsq "
                    f"where tsv @@ tsq.q {and_where} order by rk desc limit %(n)s")
        # Okapi BM25 (k1=1.2, b=0.75): tf = number of positions of the lexeme in the chunk's
        # tsvector, doc length = n_tokens, avgdl over the table, idf from the lexdf table
        k1, b = 1.2, 0.75
        return f"""select c.chunk_id, sum(qt.idf * (cardinality(u.positions) * {k1 + 1}) /
                       (cardinality(u.positions) + {k1} * (1 - {b} + {b} * c.n_tokens / {self.avgdl}))) as rk
                   from (select chunk_id, tsv, n_tokens from {self.table}, tsq
                         where tsv @@ tsq.q {and_where}) c
                   cross join lateral unnest(c.tsv) as u(lexeme, positions, weights)
                   join qt on qt.word = u.lexeme
                   group by c.chunk_id order by rk desc limit %(n)s"""

    def _tsq_cte(self) -> str:
        """The keyword query.

        or_all: OR of every question lexeme (plain Postgres FTS; no IDF, so common words dominate).
        idf:    OR of the question's `max_terms` rarest lexemes whose document frequency is at most
                `max_df` of chunks (df from footnote.<table>_lexdf). This adds BM25's missing IDF
                ingredient as term selection. Lexemes are already stemmed -> 'simple' config.
        """
        if self.cfg["lexical"] == "or_all":
            return ("tsq as (select to_tsquery('english', "
                    "replace(plainto_tsquery('english', %(q)s)::text, ' & ', ' | ')) as q)")
        if self.cfg["lexical"] == "bm25":
            # query terms with their BM25 idf; candidates are chunks matching any of them
            return f"""qt as (
                select distinct l.word, ln(1 + ({self.n_chunks} - l.ndoc + 0.5) / (l.ndoc + 0.5)) as idf
                from unnest(tsvector_to_array(to_tsvector('english', %(q)s))) as w(word)
                join {self.lexdf} l on l.word = w.word
                where l.ndoc <= %(max_df)s * {self.n_chunks}),
            tsq as (select to_tsquery('simple', coalesce(string_agg(quote_literal(word), ' | '), '')) as q from qt)"""
        return f"""tsq as (
                select to_tsquery('simple', coalesce(string_agg(quote_literal(word), ' | '), '')) as q
                from (select l.word from unnest(tsvector_to_array(to_tsvector('english', %(q)s))) as w(word)
                      join {self.lexdf} l on l.word = w.word
                      where l.ndoc <= %(max_df)s * {self.n_chunks}
                      order by l.ndoc limit %(max_terms)s) x)"""

    def _ensure_lexdf(self) -> None:
        """Document-frequency table for the idf keyword query (built once per chunk table)."""
        name = self.cfg["table"] + "_lexdf"
        self.lexdf = db.table_name(name)
        self.n_chunks, avgdl = self.conn.execute(f"select count(*), avg(n_tokens) from {self.table}").fetchone()
        self.n_chunks, self.avgdl = int(self.n_chunks), float(avgdl)
        exists = self.conn.execute("select to_regclass(%s)", (self.lexdf,)).fetchone()[0]
        if not exists:
            self.conn.execute(f"create table {self.lexdf} as select word, ndoc from "
                              f"ts_stat('select tsv from {self.table}')")
            self.conn.execute(f"alter table {self.lexdf} add primary key (word)")

    # -------------------------------------------------------------- search
    def search(self, question: str, measure_db: bool = False) -> Result:
        """measure_db: also run EXPLAIN ANALYZE to record server-side execution time, which
        separates DB compute from network round-trips (the client is far from us-east-1)."""
        k = int(self.cfg["k"])
        rerank = bool(self.cfg["rerank"])
        n = int(self.cfg["candidates"]) if rerank else k
        company = self.match_company(question) if self.match_company else None
        t0 = time.perf_counter()
        v = embed_query(question)
        t1 = time.perf_counter()
        sql, params = self._sql(n, company)
        params |= {"v": v, "q": question}
        rows = self.conn.execute(sql, params).fetchall()
        t2 = time.perf_counter()
        hits = [Hit(*r[:9], score=float(r[9])) for r in rows]
        timings = {"embed": (t1 - t0) * 1e3, "search": (t2 - t1) * 1e3}
        if rerank and hits:
            scores = get_reranker().predict([(question, chunk_header(h) + "\n" + h.text) for h in hits],
                                            batch_size=64, show_progress_bar=False)
            for h, s in zip(hits, scores):
                h.score = float(s)
            hits = sorted(hits, key=lambda h: -h.score)
            timings["rerank"] = (time.perf_counter() - t2) * 1e3
        hits = hits[:k]
        timings["total"] = (time.perf_counter() - t0) * 1e3
        if measure_db:
            plan = self.conn.execute("explain (analyze, format json) " + sql, params).fetchone()[0]
            timings["db_exec"] = float(plan[0]["Execution Time"])
        return Result(hits, timings, {"company_filter": company})


def network_rtt_ms(conn, n: int = 20) -> dict[str, float]:
    xs = []
    for _ in range(n):
        t = time.perf_counter()
        conn.execute("select 1").fetchone()
        xs.append((time.perf_counter() - t) * 1e3)
    xs.sort()
    return {"p50": round(xs[n // 2], 1), "min": round(xs[0], 1), "max": round(xs[-1], 1)}
