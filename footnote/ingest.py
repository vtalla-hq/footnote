"""Ingest FinanceBench PDFs into a pgvector table.

    python -m footnote.ingest --config configs/ingest_256.yaml

Idempotent: chunk_ids are deterministic, rows are upserted, stale chunks of a doc are deleted.
Parsing is cached per PDF (cache/parsed/) and embeddings per content hash (cache/embeddings.sqlite),
so a re-run only pays for DB writes.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

from footnote.dataset import DOWNLOAD_REPORT, PDF_DIR, doc_metadata, load_questions, referenced_docs

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("overlap", cfg["chunk_size"] // 8)
    cfg.setdefault("table_max", max(cfg["chunk_size"], 512))
    return cfg


def embed_header(meta: dict, section: str | None) -> str:
    h = f"{meta['company']} {meta['doc_type']} FY{meta['fiscal_year']}"
    return f"{h} | {section}" if section else h


def _parse_and_chunk(args) -> tuple[str, int, list[dict], float]:
    doc_name, meta, chunk_size, overlap, table_max, cold = args
    from footnote.chunking import chunk_elements
    from footnote.embed import content_hash, count_tokens
    from footnote.parse import page_count, parse_pdf

    t0 = time.perf_counter()
    path = PDF_DIR / f"{doc_name}.pdf"
    elements = parse_pdf(path, use_cache=not cold)
    chunks = chunk_elements(elements, chunk_size, count_tokens, overlap=overlap, table_max=table_max)
    rows = []
    for i, c in enumerate(chunks):
        embed_text = embed_header(meta, c.section) + "\n" + c.text
        rows.append({
            "chunk_id": f"{doc_name}::{i:05d}",
            "doc_name": doc_name,
            "company": meta["company"],
            "doc_type": meta["doc_type"],
            "fiscal_year": meta["fiscal_year"],
            "page_start": min(c.pages),
            "page_end": max(c.pages),
            "pages": c.pages,
            "section": c.section,
            "chunk_type": c.chunk_type,
            "n_tokens": c.n_tokens,
            "content_hash": content_hash(embed_text),
            "text": c.text,
            "embed_text": embed_text,
        })
    return doc_name, page_count(path), rows, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first N docs (smoke test)")
    ap.add_argument("--no-db", action="store_true", help="parse/chunk/embed only")
    ap.add_argument("--cold", action="store_true", help="ignore parse + embedding caches (true ingest cost)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    table = cfg["table"]

    t_start = time.perf_counter()
    qs = load_questions()
    meta = doc_metadata(qs)
    report = json.loads(DOWNLOAD_REPORT.read_text()) if DOWNLOAD_REPORT.exists() else {"docs_failed": {}}
    docs = [d for d in referenced_docs(qs) if d not in report["docs_failed"] and (PDF_DIR / f"{d}.pdf").exists()]
    docs = docs[: args.limit] if args.limit else docs
    print(f"[{table}] ingesting {len(docs)} docs, chunk_size={cfg['chunk_size']} overlap={cfg['overlap']} "
          f"table_max={cfg['table_max']}")

    jobs = [(d, meta[d], cfg["chunk_size"], cfg["overlap"], cfg["table_max"], args.cold) for d in docs]
    per_doc: dict[str, tuple[int, list[dict]]] = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        for n, (d, pages, rows, secs) in enumerate(ex.map(_parse_and_chunk, jobs), 1):
            per_doc[d] = (pages, rows)
            print(f"  [{n}/{len(docs)}] {d}: {pages} pages, {len(rows)} chunks ({secs:.1f}s)", flush=True)
    t_parse = time.perf_counter() - t0

    from footnote.embed import EmbeddingCache, embed_passages

    all_rows = [r for d in docs for r in per_doc[d][1]]
    t0 = time.perf_counter()
    cache = EmbeddingCache(ROOT / "cache" / "embeddings_cold.sqlite") if args.cold else EmbeddingCache()
    if args.cold:
        cache.db.execute("delete from emb"); cache.db.commit()
    embs, computed = embed_passages([r["embed_text"] for r in all_rows], cache)
    t_embed = time.perf_counter() - t0
    print(f"embedded {len(all_rows)} chunks ({computed} new, {len(all_rows) - computed} from cache) in {t_embed:.1f}s")

    stats = {
        "table": table, "config": cfg, "docs": len(docs),
        "pages": sum(per_doc[d][0] for d in docs),
        "chunks": len(all_rows),
        "table_chunks": sum(r["chunk_type"] == "table" for r in all_rows),
        "tokens_mean": round(sum(r["n_tokens"] for r in all_rows) / max(len(all_rows), 1), 1),
        "tokens_max": max((r["n_tokens"] for r in all_rows), default=0),
        "chunks_over_512_tokens": sum(r["n_tokens"] > 512 for r in all_rows),
        "embeddings_computed": computed,
        "seconds": {"parse_chunk": round(t_parse, 1), "embed": round(t_embed, 1)},
    }

    if not args.no_db:
        from footnote import db

        conn = db.connect()
        stats["db_bytes_before"] = db.db_size_bytes(conn)
        db.create_table(conn, table)
        t0 = time.perf_counter()
        i = 0
        for d in docs:
            rows = per_doc[d][1]
            db.upsert_doc(conn, table, d, rows, embs[i:i + len(rows)])
            i += len(rows)
        conn.execute(f"analyze {db.table_name(table)}")
        stats["seconds"]["upsert"] = round(time.perf_counter() - t0, 1)
        stats["db_rows"] = conn.execute(f"select count(*) from {db.table_name(table)}").fetchone()[0]
        stats["db_bytes_after"] = db.db_size_bytes(conn)
        stats["table_bytes"] = db.table_size_bytes(conn, table)

    stats["cold"] = args.cold
    stats["seconds"]["total"] = round(time.perf_counter() - t_start, 1)
    out = ROOT / "results" / f"ingest_{table}{'_cold' if args.cold else ''}{'_nodb' if args.no_db else ''}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "config"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
