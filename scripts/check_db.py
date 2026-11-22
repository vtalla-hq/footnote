"""Check the DB table against locally rebuilt chunks (ids, hashes, embeddings, indexes) + smoke query.

    python -m scripts.check_db
"""
import random
import sys


import numpy as np

from footnote import db
from footnote.dataset import doc_metadata, load_questions, referenced_docs
from footnote.embed import EmbeddingCache, embed_passages, embed_query
from footnote.ingest import _parse_and_chunk

table = "chunks_256"
qs = load_questions()
meta = doc_metadata(qs)
local = {}
for d in referenced_docs(qs):
    for r in _parse_and_chunk((d, meta[d], 256, 32, 416, False))[2]:
        local[r["chunk_id"]] = r

conn = db.connect()
rows = conn.execute(f"select chunk_id, content_hash, pages, page_start, page_end, doc_name from footnote.{table}").fetchall()
remote = {r[0]: r for r in rows}
print("local rows", len(local), "| db rows", len(remote))
print("ids only local:", len(local.keys() - remote.keys()), "| ids only in db:", len(remote.keys() - local.keys()))
mism = [k for k in local.keys() & remote.keys() if local[k]["content_hash"] != remote[k][1]]
print("hash mismatches:", len(mism))
bad_pages = [k for k, r in remote.items() if not (r[3] == min(r[2]) and r[4] == max(r[2]))]
print("page_start/end inconsistent with pages[]:", len(bad_pages))

random.seed(0)
sample = random.sample(sorted(local.keys() & remote.keys()), 50)
dbv = dict(conn.execute(f"select chunk_id, embedding from footnote.{table} where chunk_id = any(%s)", (sample,)).fetchall())
lv, _ = embed_passages([local[k]["embed_text"] for k in sample], EmbeddingCache())
diff = max(float(np.abs(dbv[k].to_numpy() - lv[i]).max()) for i, k in enumerate(sample))
print("max |db - local| embedding diff over 50 sampled rows:", diff)

idx = conn.execute("select indexname, indexdef from pg_indexes where schemaname='footnote' and tablename=%s", (table,)).fetchall()
for name, d in idx:
    print("index", name, "|", d.split(" USING ")[1][:60])
print("db size MB", round(db.db_size_bytes(conn) / 1e6, 1), "| table MB", round(db.table_size_bytes(conn, table) / 1e6, 1))

q = next(q for q in qs if q["doc_name"] == "3M_2018_10K" and "capital expenditure" in q["question"])
v = embed_query(q["question"])
top = conn.execute(f"select doc_name, pages, chunk_type, left(text, 80) from footnote.{table} "
                   "order by embedding <=> %s limit 5", (v,)).fetchall()
gold = q["evidence"][0]["evidence_page_num"] + 1
print(f"smoke: {q['question'][:70]}... gold {q['doc_name']} p{gold}")
for t in top:
    print("  ", t[0], t[1], t[2], t[3].replace("\n", " / "))
