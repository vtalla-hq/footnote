"""Local embeddings (BAAI/bge-small-en-v1.5, 384-dim) with an on-disk cache keyed by content hash."""
from __future__ import annotations

import hashlib
import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def content_hash(text: str, model: str = MODEL_NAME) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()


@lru_cache(maxsize=1)
def get_model(name: str = MODEL_NAME):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(name, device=device)


@lru_cache(maxsize=1)
def get_tokenizer(name: str = MODEL_NAME):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False))


class EmbeddingCache:
    def __init__(self, path: Path = CACHE_DIR / "embeddings.sqlite"):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("create table if not exists emb (h text primary key, v blob)")

    def get_many(self, hashes: list[str]) -> dict[str, np.ndarray]:
        out = {}
        for i in range(0, len(hashes), 900):
            part = hashes[i:i + 900]
            q = f"select h, v from emb where h in ({','.join('?' * len(part))})"
            for h, v in self.db.execute(q, part):
                out[h] = np.frombuffer(v, dtype=np.float32)
        return out

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        self.db.executemany("insert or replace into emb values (?, ?)",
                            [(h, v.astype(np.float32).tobytes()) for h, v in items.items()])
        self.db.commit()


def embed_passages(texts: list[str], cache: EmbeddingCache | None = None, batch_size: int = 64) -> tuple[np.ndarray, int]:
    """Returns (embeddings [n, 384], number actually computed). Cached by content hash."""
    cache = cache or EmbeddingCache()
    hashes = [content_hash(t) for t in texts]
    have = cache.get_many(list(set(hashes)))
    todo = sorted({h: t for h, t in zip(hashes, texts) if h not in have}.items())
    if todo:
        vecs = get_model().encode([t for _, t in todo], batch_size=batch_size,
                                  normalize_embeddings=True, show_progress_bar=len(todo) > 2000)
        new = {h: v for (h, _), v in zip(todo, vecs)}
        cache.put_many(new)
        have.update(new)
    return np.stack([have[h] for h in hashes]) if texts else np.zeros((0, DIM), np.float32), len(todo)


@lru_cache(maxsize=1)
def get_fastembed(name: str = MODEL_NAME):
    """ONNX build of the same model (no torch) for serverless hosts. Measured against the torch
    model on all 150 FinanceBench questions: query-embedding cosine 1.00000 (min and mean)."""
    import os

    from fastembed import TextEmbedding

    return TextEmbedding(name, cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"))


def embed_query(q: str) -> np.ndarray:
    import os

    if os.environ.get("FOOTNOTE_EMBEDDER") == "fastembed":
        v = next(iter(get_fastembed().embed([QUERY_INSTRUCTION + q])))
        return (v / np.linalg.norm(v)).astype(np.float32)
    return get_model().encode([QUERY_INSTRUCTION + q], normalize_embeddings=True)[0]
