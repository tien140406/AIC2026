from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import replace
from pathlib import Path

from pydantic import BaseModel, Field

from .gemini_online import GeminiOnlineClient
from .query_understanding import QueryPlan


class StructuredQuery(BaseModel):
    query_vi: str
    query_en: str
    visual_queries_en: list[str] = Field(default_factory=list, max_length=3)
    exact_terms: list[str] = Field(default_factory=list)
    objects_en: list[str] = Field(default_factory=list)
    answer_target: str = ""


QUERY_EXPANSION_SYSTEM_PROMPT = (
    "Parse Vietnamese and English video-retrieval queries. Translate faithfully and "
    "never invent colors, clothing, actions, objects, names, numbers, spatial relations, "
    "or event order. Return exact_terms only when they are literal spans from the query. "
    "visual_queries_en must be concise descriptions suitable for visual retrieval."
)


class GeminiQueryTransformer:
    """Cached direct-Gemini expansion used only after a user submits a query."""

    def __init__(self, root: Path, index: Path,
                 model: str | None = None) -> None:
        self.root, self.index = root.resolve(), index.resolve()
        self.client = GeminiOnlineClient(self.root, model=model)
        self.model = self.client.model
        self.database = self.index / "query_transformer_cache.sqlite3"
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )

    @property
    def available(self) -> bool:
        return self.client.available

    def _cached(self, key: str) -> StructuredQuery | None:
        with sqlite3.connect(self.database) as connection:
            row = connection.execute(
                "SELECT value FROM cache WHERE key=?", (key,)
            ).fetchone()
        return StructuredQuery.model_validate_json(row[0]) if row else None

    def expand(self, query: str) -> StructuredQuery:
        key = hashlib.sha256(
            f"gemini-query-expand:v1:{self.model}:{query}".encode("utf-8")
        ).hexdigest()
        cached = self._cached(key)
        if cached:
            return cached
        response = self.client.json_response(
            f"Expand this AIC video retrieval query:\n{query}",
            StructuredQuery,
            system_prompt=QUERY_EXPANSION_SYSTEM_PROMPT,
            model=self.model,
            temperature=0,
            max_tokens=2048,
        )
        value = StructuredQuery.model_validate_json(response.text)
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO cache VALUES(?,?)",
                (key, value.model_dump_json()),
            )
        return value

    def enhance(self, plan: QueryPlan) -> QueryPlan:
        value = self.expand(plan.normalized)
        source_folded = plan.normalized.casefold()
        safe_exact = []
        for term in value.exact_terms:
            cleaned = " ".join(str(term).strip().split())
            if cleaned and (cleaned.casefold() in source_folded or
                            (cleaned.isdigit() and re.search(
                                rf"(?<!\d){re.escape(cleaned)}(?!\d)",
                                plan.normalized))):
                safe_exact.append(cleaned)
        object_vocabulary = {
            "person", "bicycle", "motorcycle", "car", "bus", "truck", "boat",
            "airplane", "train", "cell phone", "laptop", "dog", "cat", "microphone",
        }
        safe_objects = [item.casefold().strip() for item in value.objects_en
                        if item.casefold().strip() in object_vocabulary]
        generated_variants = [value.query_en, *value.visual_queries_en]
        if (any(cue in plan.normalized.casefold() for cue in ("màu xanh", "áo xanh"))
                and "xanh lá" not in plan.normalized.casefold()):
            generated_variants = [
                re.sub(r"\bgreen\b", "blue", text, flags=re.IGNORECASE)
                for text in generated_variants
            ]
        variants = tuple(dict.fromkeys([*plan.variants, *generated_variants]))
        return replace(
            plan,
            variants=variants,
            exact_terms=tuple(dict.fromkeys([*plan.exact_terms, *safe_exact])),
            objects=tuple(dict.fromkeys([*plan.objects, *safe_objects])),
            answer_target=plan.answer_target,
            transformer_used=True,
        )


# Compatibility for callers that imported the previous class name.
LocalQueryTransformer = GeminiQueryTransformer
