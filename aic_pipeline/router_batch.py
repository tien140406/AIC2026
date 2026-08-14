from __future__ import annotations

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .router_chat import RouterChatClient, RouterResponse


@dataclass
class PreparedRouterRequest:
    key: str
    prompt: str
    schema: dict[str, Any]
    image: Image.Image | None
    temperature: float
    max_tokens: int


def decode_batch_request(envelope: dict[str, Any], default_max_tokens: int) -> PreparedRouterRequest:
    key = str(envelope.get("key", "")).strip()
    if not key:
        raise ValueError("batch request is missing key")
    request = envelope.get("request")
    if not isinstance(request, dict):
        raise ValueError(f"batch request {key} has no request object")

    text_parts: list[str] = []
    images: list[Image.Image] = []
    for content in request.get("contents") or []:
        for part in content.get("parts") or []:
            if "text" in part:
                text_parts.append(str(part["text"]))
            inline = part.get("inlineData") or part.get("inline_data")
            if inline:
                data = base64.b64decode(str(inline["data"]), validate=True)
                with Image.open(BytesIO(data)) as source:
                    images.append(source.convert("RGB"))
    if not text_parts:
        raise ValueError(f"batch request {key} has no text prompt")
    if len(images) > 1:
        raise ValueError(f"batch request {key} has more than one image")

    config = request.get("generationConfig") or request.get("generation_config") or {}
    schema = config.get("responseJsonSchema") or config.get("response_json_schema")
    if not isinstance(schema, dict):
        raise ValueError(f"batch request {key} has no response JSON schema")
    return PreparedRouterRequest(
        key=key,
        prompt="\n".join(text_parts),
        schema=schema,
        image=images[0] if images else None,
        temperature=float(config.get("temperature", 0.1)),
        max_tokens=int(config.get("maxOutputTokens", default_max_tokens)),
    )


def _successful(envelope: dict[str, Any]) -> bool:
    response = envelope.get("response")
    return isinstance(response, dict) and bool(response.get("candidates"))


def _success_envelope(key: str, response: RouterResponse, model: str,
                      attempt_latencies: list[float]) -> dict[str, Any]:
    choice = (response.raw.get("choices") or [{}])[0]
    return {
        "key": key,
        "response": {
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": response.text}]},
                "finishReason": choice.get("finish_reason", "STOP"),
            }],
            "usageMetadata": response.usage,
        },
        "router": {
            "provider": "9router",
            "model": response.raw.get("model", model),
            "response_id": response.raw.get("id"),
            "usage": response.usage,
            "latency_seconds": round(response.latency_seconds, 6),
            "attempts": len(attempt_latencies),
            "attempt_latency_seconds": [round(value, 6) for value in attempt_latencies],
        },
    }


def _error_envelope(key: str, exc: Exception, attempts: int,
                    attempt_latencies: list[float] | None = None) -> dict[str, Any]:
    return {
        "key": key,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "attempts": attempts,
        },
        "router": {
            "provider": "9router",
            "attempts": attempts,
            "attempt_latency_seconds": [round(value, 6)
                                        for value in (attempt_latencies or [])],
        },
    }


def _load_output(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    output: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        envelope = json.loads(line)
        key = str(envelope.get("key", "")).strip()
        if not key:
            raise ValueError(f"response line {number} is missing key")
        output[key] = envelope
    return output


def _write_output(path: Path, keys: list[str], output: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(output[key], ensure_ascii=False) + "\n"
                for key in keys if key in output),
        encoding="utf-8",
    )
    temporary.replace(path)


def router_batch_status(input_path: Path, output_path: Path) -> dict[str, int]:
    requests = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    output = _load_output(output_path)
    keys = [str(envelope.get("key", "")).strip() for envelope in requests]
    succeeded = sum(_successful(output[key]) for key in keys if key in output)
    failed = sum(not _successful(output[key]) for key in keys if key in output)
    return {"requests": len(keys), "succeeded": succeeded, "failed": failed,
            "pending": len(keys) - succeeded - failed}


def _latency_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "total": 0.0, "mean": 0.0,
                "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "count": len(values),
        "total": round(sum(values), 6),
        "mean": round(sum(values) / len(values), 6),
        "p50": round(percentile(0.50), 6),
        "p95": round(percentile(0.95), 6),
        "max": round(max(values), 6),
    }


def run_router_batch(
    root: Path,
    input_path: Path,
    output_path: Path,
    *,
    workers: int = 4,
    max_retries: int = 3,
    timeout_seconds: float = 120,
    default_max_tokens: int = 64000,
    client: RouterChatClient | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    batch_started = time.perf_counter()
    requests = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    keys = [str(envelope.get("key", "")).strip() for envelope in requests]
    if any(not key for key in keys) or len(keys) != len(set(keys)):
        raise ValueError("batch input contains missing or duplicate keys")

    output = _load_output(output_path)
    pending = [envelope for envelope, key in zip(requests, keys)
               if not _successful(output.get(key, {}))]
    router = client or RouterChatClient(root, timeout_seconds=timeout_seconds)

    def execute(envelope: dict[str, Any]) -> dict[str, Any]:
        key = str(envelope.get("key", "")).strip()
        try:
            prepared = decode_batch_request(envelope, default_max_tokens)
        except Exception as exc:
            return _error_envelope(key, exc, 0)
        last_error: Exception | None = None
        attempt_latencies: list[float] = []
        try:
            for attempt in range(1, max_retries + 2):
                attempt_started = time.perf_counter()
                try:
                    response = router.json_schema_response(
                        prepared.prompt, prepared.schema, prepared.image,
                        temperature=prepared.temperature,
                        max_tokens=prepared.max_tokens,
                    )
                    attempt_latencies.append(time.perf_counter() - attempt_started)
                    return _success_envelope(
                        prepared.key, response, router.model, attempt_latencies)
                except Exception as exc:
                    attempt_latencies.append(time.perf_counter() - attempt_started)
                    last_error = exc
                    if attempt <= max_retries:
                        time.sleep(min(2 ** (attempt - 1), 8))
            assert last_error is not None
            return _error_envelope(
                prepared.key, last_error, max_retries + 1, attempt_latencies)
        finally:
            if prepared.image is not None:
                prepared.image.close()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(execute, envelope): str(envelope["key"])
                   for envelope in pending}
        for future in as_completed(futures):
            result = future.result()
            output[str(result["key"])] = result
            _write_output(output_path, keys, output)
            if progress:
                progress(result)

    status = router_batch_status(input_path, output_path)
    total_tokens = sum(
        int(((output.get(key, {}).get("router") or {}).get("usage") or {}).get("total_tokens", 0))
        for key in keys
    )
    operation_latencies = [
        float((output.get(key, {}).get("router") or {}).get("latency_seconds", 0))
        for key in keys
        if float((output.get(key, {}).get("router") or {}).get("latency_seconds", 0)) > 0
    ]
    attempt_latencies = [
        float(value)
        for key in keys
        for value in ((output.get(key, {}).get("router") or {}).get(
            "attempt_latency_seconds", []) or [])
    ]
    return {**status, "model": router.model, "output": str(output_path),
            "total_tokens": total_tokens,
            "wall_seconds": round(time.perf_counter() - batch_started, 6),
            "latency_seconds": _latency_summary(operation_latencies),
            "attempt_latency_seconds": _latency_summary(attempt_latencies)}
