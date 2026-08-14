from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Type
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel


@dataclass
class GeminiResponse:
    text: str
    usage: dict[str, Any] | None
    raw: dict[str, Any]
    latency_seconds: float


class GeminiOnlineClient:
    """Direct Gemini REST client reserved for user-triggered query work."""

    def __init__(self, root: Path, model: str | None = None,
                 timeout_seconds: float = 30.0) -> None:
        load_dotenv(root.resolve() / ".env")
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.api_base = os.getenv(
            "GEMINI_API_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta",
        ).rstrip("/")
        self.model = (model or os.getenv("GEMINI_QUERY_MODEL")
                      or os.getenv("GEMINI_MODEL")
                      or "gemini-3.1-flash-lite").strip()
        self.timeout_seconds = timeout_seconds

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _image_part(image: Image.Image) -> dict[str, Any]:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=90, optimize=True)
        return {"inlineData": {
            "mimeType": "image/jpeg",
            "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }}

    @staticmethod
    def _response_text(raw: dict[str, Any]) -> str:
        candidates = raw.get("candidates") or []
        if not candidates:
            feedback = raw.get("promptFeedback") or raw.get("prompt_feedback")
            raise ValueError(f"Gemini response has no candidates: {feedback}")
        parts = ((candidates[0].get("content") or {}).get("parts") or [])
        text = "".join(str(part.get("text", "")) for part in parts).strip()
        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text,
                          re.DOTALL | re.IGNORECASE)
        if fence:
            return fence.group(1).strip()
        start, end = text.find("{"), text.rfind("}")
        return text[start:end + 1] if start >= 0 and end > start else text

    def json_response(
        self,
        prompt: str,
        schema: Type[BaseModel],
        image: Image.Image | None = None,
        *,
        system_prompt: str | None = None,
        video_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> GeminiResponse:
        if not self.api_key:
            raise RuntimeError(
                "Online query expansion and QA require GEMINI_API_KEY in .env"
            )
        parts: list[dict[str, Any]] = [{"text": prompt}]
        if video_url:
            parts.append({"fileData": {
                "mimeType": "video/mp4", "fileUri": video_url,
            }})
        if image is not None:
            parts.append(self._image_part(image))
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
                "responseJsonSchema": schema.model_json_schema(),
            },
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        selected_model = model or self.model
        endpoint = f"{self.api_base}/models/{quote(selected_model, safe='')}:generateContent"
        request = Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "AIC2026-Online-Query/1.0",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Gemini HTTP {exc.code}: {body.strip() or exc.reason}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            raise RuntimeError(f"Gemini request failed: {reason}") from exc
        text = self._response_text(raw)
        validated = schema.model_validate_json(text)
        return GeminiResponse(
            text=validated.model_dump_json(),
            usage=raw.get("usageMetadata") or raw.get("usage_metadata"),
            raw=raw,
            latency_seconds=time.perf_counter() - started,
        )
