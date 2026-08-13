from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel

from .router_chat import RouterChatClient


class ExpandedQuery(BaseModel):
    english_translation: str
    visual_queries: list[str]
    keywords_vi: list[str]
    keywords_en: list[str]


class GeminiTextServices:
    def __init__(self, root: Path, index: Path,
                 embedding_model: str = "gemini-embedding-001",
                 expansion_model: str | None = None) -> None:
        self.root, self.index = root.resolve(), index.resolve()
        self.embedding_model = embedding_model
        load_dotenv(self.root / ".env")
        self.client = RouterChatClient(self.root)
        self.expansion_model = expansion_model or self.client.model
        self.database = self.index / "gemini_text_cache.sqlite3"
        with sqlite3.connect(self.database) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, kind TEXT, value TEXT)")

    def _get(self, key: str) -> str | None:
        with sqlite3.connect(self.database) as connection:
            row = connection.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return str(row[0]) if row else None

    def _put(self, key: str, kind: str, value: str) -> None:
        with sqlite3.connect(self.database) as connection:
            connection.execute("INSERT OR REPLACE INTO cache VALUES(?,?,?)", (key, kind, value))

    def expand(self, query: str) -> ExpandedQuery:
        from google.genai import types
        key = hashlib.sha256(f"expand:{self.expansion_model}:{query}".encode("utf-8")).hexdigest()
        cached = self._get(key)
        if cached:
            return ExpandedQuery.model_validate_json(cached)
        prompt = f"""Expand this Vietnamese AIC video retrieval query without inventing details: {query}
Return a faithful English translation, up to 3 concise visual paraphrases, and short Vietnamese/English keywords.
Preserve exact counts, left/right/top/bottom positions, names, OCR strings and event order."""
        response = self.client.json_response(
            prompt, ExpandedQuery, model=self.expansion_model,
            temperature=0, max_tokens=1024)
        value = ExpandedQuery.model_validate_json(response.text)
        self._put(key, "expansion", value.model_dump_json())
        return value

    def embed_query(self, query: str, dimension: int) -> np.ndarray:
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError(
                "Legacy Gemini embedding is disabled: GEMINI_API_KEY is empty. "
                "Use the local BGE-M3/SigLIP/OpenCLIP retrieval branches."
            )
        from google.genai import types
        key = hashlib.sha256(
            f"embed:{self.embedding_model}:{dimension}:{query}".encode("utf-8")).hexdigest()
        cached = self._get(key)
        if cached:
            vector = np.asarray(json.loads(cached), dtype=np.float32)
        else:
            from google import genai
            embedding_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            response = embedding_client.models.embed_content(
                model=self.embedding_model, contents=query,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY",
                                                output_dimensionality=dimension))
            vector = np.asarray(response.embeddings[0].values, dtype=np.float32)
            self._put(key, "embedding", json.dumps(vector.tolist()))
        return vector / max(float(np.linalg.norm(vector)), 1e-12)
