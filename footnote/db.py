"""Postgres/pgvector access. All tables live in the `footnote` schema."""
from __future__ import annotations

import os
import re

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

SCHEMA = "footnote"


def connect() -> psycopg.Connection:
    load_dotenv()
    url = os.environ.get("SUPABASE_DB_URL")
    if not url or "<password>" in url:
        raise RuntimeError("SUPABASE_DB_URL is not set in .env (see .env.example)")
    conn = psycopg.connect(url, autocommit=True, prepare_threshold=None)
    conn.execute("set search_path to footnote, extensions, public")
    register_vector(conn)
    return conn


def table_name(name: str) -> str:
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError(f"bad table name {name!r}")
    return f"{SCHEMA}.{name}"


def create_table(conn: psycopg.Connection, name: str, hnsw: bool = True) -> None:
    t = table_name(name)
    short = name
    conn.execute(f"""
        create table if not exists {t} (
            chunk_id     text primary key,
            doc_name     text not null,
            company      text,
            doc_type     text,
            fiscal_year  int,
            page_start   int not null,
            page_end     int not null,
            pages        int[] not null,
            section      text,
            chunk_type   text not null,
            n_tokens     int,
            content_hash text not null,
            text         text not null,
            embedding    extensions.vector(384) not null,
            tsv          tsvector generated always as (
                             to_tsvector('english', coalesce(section, '') || ' ' || text)) stored
        )""")
    if hnsw:
        conn.execute(f"create index if not exists {short}_hnsw on {t} using hnsw (embedding extensions.vector_cosine_ops)")
    conn.execute(f"create index if not exists {short}_tsv on {t} using gin (tsv)")
    conn.execute(f"create index if not exists {short}_doc on {t} (doc_name)")
    conn.execute(f"create index if not exists {short}_co_fy on {t} (company, fiscal_year)")


COLS = ["chunk_id", "doc_name", "company", "doc_type", "fiscal_year", "page_start", "page_end",
        "pages", "section", "chunk_type", "n_tokens", "content_hash", "text", "embedding"]


def upsert_doc(conn: psycopg.Connection, name: str, doc_name: str, rows: list[dict], embs: np.ndarray) -> None:
    """Idempotent: upsert every chunk of one doc, then delete that doc's stale chunk_ids."""
    t = table_name(name)
    with conn.transaction():
        # generated tsv column can't be copied into; stage only the base columns
        conn.execute("drop table if exists _s")
        conn.execute("create temp table if not exists _s as select " + ", ".join(COLS) + f" from {t} limit 0")
        conn.execute("truncate _s")
        with conn.cursor().copy(f"copy _s ({', '.join(COLS)}) from stdin with (format binary)") as cp:
            cp.set_types(["text", "text", "text", "text", "int4", "int4", "int4", "int4[]", "text",
                          "text", "int4", "text", "text", "vector"])
            for r, e in zip(rows, embs):
                cp.write_row([r[c] for c in COLS[:-1]] + [e])
        updates = ", ".join(f"{c} = excluded.{c}" for c in COLS[1:])
        conn.execute(f"""insert into {t} ({', '.join(COLS)}) select {', '.join(COLS)} from _s
                         on conflict (chunk_id) do update set {updates}
                         where {t}.content_hash is distinct from excluded.content_hash""")
        conn.execute(f"delete from {t} where doc_name = %s and chunk_id not in (select chunk_id from _s)",
                     (doc_name,))


def drop_table(conn: psycopg.Connection, name: str) -> None:
    """Only for throwaway tables (CI); refuses the experiment tables."""
    if not name.startswith("ci_"):
        raise ValueError(f"refusing to drop non-CI table {name!r}")
    conn.execute(f"drop table if exists {table_name(name)}")


def db_size_bytes(conn: psycopg.Connection) -> int:
    return conn.execute("select pg_database_size(current_database())").fetchone()[0]


def table_size_bytes(conn: psycopg.Connection, name: str) -> int:
    return conn.execute("select pg_total_relation_size(%s::regclass)", (table_name(name),)).fetchone()[0]
