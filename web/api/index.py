"""Vercel Python function (FastAPI, ASGI). Served at /api/* next to the Next.js app.

Needs SUPABASE_DB_URL and GROQ_API_KEY as Vercel environment variables.
"""
import os

# serverless: no torch (fastembed ONNX build of the same bge-small), writable dirs only under /tmp
os.environ.setdefault("FOOTNOTE_EMBEDDER", "fastembed")
os.environ.setdefault("FOOTNOTE_CACHE_DIR", "/tmp/footnote")
os.environ.setdefault("FASTEMBED_CACHE_PATH", "/tmp/fastembed")

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

app = FastAPI(title="Footnote API")


class Ask(BaseModel):
    question: str


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/ask")
def ask(body: Ask) -> dict:
    from footnote.service import answer

    try:
        return answer(body.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {str(e)[:300]}") from e
