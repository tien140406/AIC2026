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
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel


@dataclass
class RouterResponse:
    text: str
    usage: dict[str, Any] | None
    raw: dict[str, Any]
    latency_seconds: float


class RouterChatClient:
    """Minimal OpenAI-compatible client for the configured 9Router gateway."""

    def __init__(self, root: Path, timeout_seconds: float = 30.0) -> None:
        load_dotenv(root.resolve() / ".env")
        self.url = os.getenv(
            "NINEROUTER_CHAT_URL", "https://router.studev.net/v1/chat/completions"
        ).strip()
        self.key = os.getenv("NINEROUTER_KEY", "").strip()
        self.model = os.getenv("NINEROUTER_MODEL", "ag/gemini-3-flash").strip()
        self.max_output_tokens = int(os.getenv(
            "NINEROUTER_MAX_OUTPUT_TOKENS", "64000"
        ))
        if self.max_output_tokens <= 0:
            raise ValueError("NINEROUTER_MAX_OUTPUT_TOKENS must be positive")
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _image_part(image: Image.Image) -> dict[str, Any]:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=88, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{encoded}", "detail": "high"}}

    @staticmethod
    def _message_text(raw: dict[str, Any]) -> str:
        content = raw["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict)
            ).strip()
        raise ValueError("9Router response message content is not text")

    @classmethod
    def _json_text(cls, raw: dict[str, Any]) -> str:
        text = cls._message_text(raw)
        fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text,
                             re.DOTALL | re.IGNORECASE)
        if fence:
            text = fence.group(1).strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                finish_reason = (raw.get("choices") or [{}])[0].get("finish_reason")
                detail = f" (finish_reason={finish_reason})" if finish_reason else ""
                raise ValueError(
                    f"9Router response does not contain a JSON object{detail}") from exc
            try:
                value = json.loads(text[start:end + 1])
            except json.JSONDecodeError as nested:
                finish_reason = (raw.get("choices") or [{}])[0].get("finish_reason")
                detail = f" (finish_reason={finish_reason})" if finish_reason else ""
                raise ValueError(
                    f"9Router response JSON is incomplete{detail}") from nested
        if not isinstance(value, dict):
            raise ValueError("9Router response JSON must be an object")
        return json.dumps(value, ensure_ascii=False)

    def json_schema_response(
        self,
        prompt: str,
        schema: dict[str, Any],
        image: Image.Image | None = None,
        *,
        system_prompt: str | None = None,
        video_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> RouterResponse:
        instruction = (
            prompt + "\nReturn only one JSON object matching this JSON Schema:\n"
            + json.dumps(schema, ensure_ascii=False)
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
        if video_url:
            content.append({"type": "video_url", "video_url": {"url": video_url}})
        if image is not None:
            content.append(self._image_part(image))
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_output_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AIC2026-Pipeline/1.0",
        }
        # 9Router docs say Authorization must be omitted when gateway auth is disabled.
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        request = Request(self.url, data=json.dumps(payload).encode("utf-8"),
                          headers=headers, method="POST")
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"9Router HTTP {exc.code}: {body.strip() or exc.reason}. "
                "If remote API auth is enabled, set NINEROUTER_KEY in .env."
            ) from exc
        except (URLError, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            raise RuntimeError(f"9Router request failed: {reason}") from exc
        return RouterResponse(
            self._json_text(raw), raw.get("usage"), raw,
            time.perf_counter() - started)

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
    ) -> RouterResponse:
        response = self.json_schema_response(
            prompt, schema.model_json_schema(), image,
            system_prompt=system_prompt, video_url=video_url, model=model,
            temperature=temperature, max_tokens=max_tokens)
        # Validate at the boundary so malformed router/provider output never
        # reaches retrieval or QA state.
        validated = schema.model_validate_json(response.text)
        return RouterResponse(
            validated.model_dump_json(), response.usage, response.raw,
            response.latency_seconds)
