"""Vercel Python function (FastAPI, ASGI). Served at /api/* next to the Next.js app.

Needs SUPABASE_DB_URL and GROQ_API_KEY as Vercel environment variables.
"""
import os

# serverless: no torch (fastembed ONNX build of the same bge-small), writable dirs only under /tmp
os.environ.setdefault("FOOTNOTE_EMBEDDER", "fastembed")
os.environ.setdefault("FOOTNOTE_CACHE_DIR", "/tmp/footnote")
os.environ.setdefault("FASTEMBED_CACHE_PATH", "/tmp/fastembed")
# huggingface_hub (used by fastembed to download the model) writes to ~/.cache, which is read-only on Vercel
os.environ.setdefault("HF_HOME", "/tmp/huggingface")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

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
