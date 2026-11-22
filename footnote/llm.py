"""Groq chat client (OpenAI-compatible) with on-disk cache, retry and backoff.

* Cache: every successful response is stored in cache/llm.sqlite keyed by sha256 of
  (model, messages, params). Re-running an eval replays cached calls for free, which also makes
  long runs resumable across days on the free tier.
* 429: honours `retry-after`. Per-minute limits (TPM/RPM) are waited out; if the error is a
  per-day limit (TPD/RPD) with a long wait, QuotaExhausted is raised so the caller can save
  partial results and resume later instead of sleeping for hours.
* 5xx / timeouts: exponential backoff with jitter.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import time
from pathlib import Path

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
URL = "https://api.groq.com/openai/v1/chat/completions"


class QuotaExhausted(RuntimeError):
    def __init__(self, model: str, retry_after: float, message: str):
        super().__init__(f"{model}: daily quota exhausted (retry after ~{retry_after / 60:.0f} min): {message}")
        self.model, self.retry_after = model, retry_after


def _api_key() -> str:
    key = os.environ.get("GROQ_API_KEY") or dotenv_values(ROOT / ".env").get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set (see .env.example)")
    return key


def prompt_hash(model: str, messages: list[dict], params: dict) -> str:
    return hashlib.sha256(json.dumps({"model": model, "messages": messages, "params": params},
                                     sort_keys=True).encode()).hexdigest()


CACHE_DIR = Path(os.environ.get("FOOTNOTE_CACHE_DIR") or ROOT / "cache")  # /tmp on serverless


class Groq:
    def __init__(self, cache_path: Path = CACHE_DIR / "llm.sqlite", max_retries: int = 8,
                 max_wait_s: float = 120.0, session: requests.Session | None = None):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_path)
        self.db.execute("create table if not exists llm (h text primary key, model text, response text, created real)")
        self.max_retries, self.max_wait_s = max_retries, max_wait_s
        self.http = session or requests.Session()
        self.stats = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0, "retries": 0}

    def chat(self, model: str, messages: list[dict], **params) -> dict:
        """Returns {"content", "usage", "cached"}. Params are passed through (temperature, ...)."""
        h = prompt_hash(model, messages, params)
        row = self.db.execute("select response from llm where h = ?", (h,)).fetchone()
        if row:
            self.stats["cache_hits"] += 1
            return json.loads(row[0]) | {"cached": True}
        out = self._call(model, messages, params)
        self.db.execute("insert or replace into llm values (?, ?, ?, ?)", (h, model, json.dumps(out), time.time()))
        self.db.commit()
        return out | {"cached": False}

    def _call(self, model: str, messages: list[dict], params: dict) -> dict:
        body = {"model": model, "messages": messages} | params
        headers = {"Authorization": f"Bearer {_api_key()}"}
        for attempt in range(self.max_retries + 1):
            try:
                r = self.http.post(URL, headers=headers, json=body, timeout=120)
            except requests.RequestException:
                if attempt == self.max_retries:
                    raise
                self._sleep(2 ** attempt)
                continue
            if r.status_code == 429:
                wait = float(r.headers.get("retry-after") or 2 ** attempt)
                msg = r.text[:300]
                if wait > self.max_wait_s or "per day" in msg or "(TPD)" in msg or "(RPD)" in msg:
                    raise QuotaExhausted(model, wait, msg)
                self._sleep(wait + random.uniform(0.2, 1.5))
                continue
            if r.status_code >= 500:
                if attempt == self.max_retries:
                    r.raise_for_status()
                self._sleep(2 ** attempt)
                continue
            if not r.ok:
                raise RuntimeError(f"Groq {r.status_code}: {r.text[:300]}")
            j = r.json()
            usage = j.get("usage", {})
            self.stats["calls"] += 1
            self.stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self.stats["completion_tokens"] += usage.get("completion_tokens", 0)
            return {"content": j["choices"][0]["message"].get("content") or "",
                    "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
                    "finish_reason": j["choices"][0].get("finish_reason")}
        raise RuntimeError("unreachable")

    def _sleep(self, s: float) -> None:
        self.stats["retries"] += 1
        time.sleep(s)
