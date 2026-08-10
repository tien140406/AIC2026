from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from ..common.io import atomic_write_parquet

PROMPT_VERSION = "gemini_frame_ocr_v5_batched_screened"
Coordinate = Annotated[int, Field(ge=0, le=1000)]
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class ErrorDisposition:
    category: str
    retryable: bool
    terminal: bool = False
    retry_after_seconds: float | None = None
    status_code: int | None = None


class OcrBlock(BaseModel):
    text: str = Field(
        min_length=1,
        description="Exact visible text with original spelling, casing, numbers, and diacritics",
    )
    bbox_2d: list[Coordinate] = Field(
        min_length=4,
        max_length=4,
        description="Normalized [y_min, x_min, y_max, x_max] coordinates from 0 to 1000",
    )
    legibility: Literal["clear", "partial"]


class FrameOcr(BaseModel):
    frame_id: int
    has_text: bool
    blocks: list[OcrBlock]
    uncertainties: list[str]

    @model_validator(mode="after")
    def validate_text_state(self) -> FrameOcr:
        if self.has_text != bool(self.blocks):
            raise ValueError("has_text must match whether OCR blocks are present")
        return self


class FrameOcrBatch(BaseModel):
    frames: list[FrameOcr] = Field(
        min_length=1,
        description="One independent OCR result for every supplied keyframe",
    )

    @model_validator(mode="after")
    def validate_unique_frame_ids(self) -> FrameOcrBatch:
        frame_ids = [frame.frame_id for frame in self.frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("Batch OCR response contains duplicate frame IDs")
        return self


def _region_lines(regions: list[dict[str, Any]]) -> str:
    if not regions:
        return "- none"
    lines = []
    for region in regions:
        name = region.get("name")
        bbox = region.get("bbox")
        prefix = f"{name}: " if name else ""
        lines.append(f"- {prefix}{bbox}")
    return "\n".join(lines)


def build_prompt(
    frame_id: int,
    *,
    candidate_regions: list[dict[str, Any]] | None = None,
) -> str:
    candidates = candidate_regions or []
    return f"""Extract OCR from this single video keyframe.
The authoritative frame_id is {frame_id}; copy that exact integer into the response.

PP-OCR candidate regions, normalized as [y_min, x_min, y_max, x_max]:
{_region_lines(candidates)}

Rules:
- Scan the entire image from top to bottom and left to right before responding.
- Use candidate regions as strong hints, but include other genuine scene text outside them.
- Transcribe only text that is visibly present. Never infer, translate, repair, or complete text.
- Preserve original spelling, capitalization, punctuation, numbers, and Vietnamese diacritics.
- Include title cards, logos, watermarks, clocks, scoreboards, signs, subtitles, lower thirds, tickers, and stylized text.
- Do not omit isolated digits or words that visibly belong to a larger text region.
- Use one block per visually distinct line or text region.
- bbox_2d is [y_min, x_min, y_max, x_max], normalized to integers from 0 to 1000.
- Use legibility=partial when only a visible fragment can be read; include only that fragment.
- If no text can be read, return has_text=false, blocks=[], and explain nothing unless useful.
- Put genuine ambiguity in uncertainties. Do not invent confidence scores.
"""


def build_batch_prompt(frame_ids: list[int]) -> str:
    ids = ", ".join(str(frame_id) for frame_id in frame_ids)
    return f"""Extract OCR independently from each of these video keyframes: {ids}.

The images are supplied in the order identified by their frame labels. Return one
object per image in a top-level `frames` array, and copy each authoritative
frame_id exactly. Include every supplied frame exactly once; never combine text
from different images. Follow the OCR rules in the per-frame instructions.
"""


def build_frame_context(
    frame_id: int,
    *,
    candidate_regions: list[dict[str, Any]] | None = None,
) -> str:
    candidates = candidate_regions or []
    return f"""IMAGE FOR FRAME_ID={frame_id}
PP-OCR candidate regions, normalized as [y_min, x_min, y_max, x_max]:
{_region_lines(candidates)}

Transcribe only readable text from this image. Scan the entire image, use the
candidate regions as hints, preserve spelling/case/punctuation/numbers/diacritics,
and return has_text=false with no blocks when no text is readable. Use one block
per distinct line or text region, with bbox_2d=[y_min, x_min, y_max, x_max] in
0..1000 and legibility=clear or partial. Put genuine ambiguity in uncertainties.
"""


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _nested_values(value: Any) -> list[Any]:
    values = [value]
    if isinstance(value, dict):
        for child in value.values():
            values.extend(_nested_values(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            values.extend(_nested_values(child))
    return values


def _retry_delay_seconds(error: Exception) -> float | None:
    details = getattr(error, "details", None)
    for value in _nested_values(details):
        if not isinstance(value, str):
            continue
        match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)s\s*", value)
        if match:
            return float(match.group(1))
    match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", str(error), re.IGNORECASE)
    return float(match.group(1)) if match else None


def classify_error(error: Exception) -> ErrorDisposition:
    status_code = getattr(error, "code", None)
    if not isinstance(status_code, int):
        status_code = None
    details = getattr(error, "details", None)
    searchable = " ".join(str(value) for value in _nested_values(details)).lower()
    searchable = f"{searchable} {error}".lower()
    retry_after = _retry_delay_seconds(error)

    daily_quota_markers = (
        "generaterequestsperdayperprojectpermodel",
        "perdayperprojectpermodel",
        "requests_per_day",
    )
    if status_code == 429 and any(
        marker in searchable for marker in daily_quota_markers
    ):
        return ErrorDisposition(
            category="daily_quota_exhausted",
            retryable=False,
            terminal=True,
            retry_after_seconds=retry_after,
            status_code=status_code,
        )
    if status_code == 429:
        return ErrorDisposition(
            category="rate_limited",
            retryable=True,
            retry_after_seconds=retry_after,
            status_code=status_code,
        )
    if status_code in RETRYABLE_STATUS_CODES:
        return ErrorDisposition(
            category="transient_api_error",
            retryable=True,
            retry_after_seconds=retry_after,
            status_code=status_code,
        )
    if status_code is not None:
        return ErrorDisposition(
            category="fatal_api_error",
            retryable=False,
            status_code=status_code,
        )
    if isinstance(error, (TimeoutError, ConnectionError, ValueError)):
        return ErrorDisposition(category="transient_local_error", retryable=True)
    if (
        "timeout" in type(error).__name__.lower()
        or "connection" in type(error).__name__.lower()
    ):
        return ErrorDisposition(category="transient_local_error", retryable=True)
    return ErrorDisposition(category="fatal_local_error", retryable=False)


def normalize_search_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", normalized).strip()


def success_path(output_dir: Path, frame_id: int) -> Path:
    return output_dir / "raw" / f"frame_{frame_id:09d}.json"


def failure_path(output_dir: Path, frame_id: int) -> Path:
    return output_dir / "failures" / f"frame_{frame_id:09d}.json"


def reusable_success(
    path: Path,
    *,
    frame_id: int,
    image_sha256: str,
    model: str,
    prompt_version: str = PROMPT_VERSION,
    screening_signature: str | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    metadata_matches = (
        payload.get("image_sha256") == image_sha256
        and payload.get("model") == model
        and payload.get("prompt_version") == prompt_version
        and isinstance(payload.get("ocr"), dict)
    )
    if screening_signature is not None:
        metadata_matches = (
            metadata_matches
            and payload.get("screening_signature") == screening_signature
        )
    if not metadata_matches:
        return False
    try:
        ocr = FrameOcr.model_validate(payload["ocr"])
    except ValueError:
        return False
    return payload.get("frame_id") == frame_id and ocr.frame_id == frame_id


def frame_record(payload: dict[str, Any]) -> dict[str, Any]:
    ocr = FrameOcr.model_validate(payload["ocr"])
    texts = [block.text for block in ocr.blocks]
    raw_text = "\n".join(texts)
    return {
        "video_id": payload["video_id"],
        "shot_id": payload.get("shot_id"),
        "frame_id": payload["frame_id"],
        "pts_time": payload.get("pts_time"),
        "image_path": payload["image_path"],
        "image_sha256": payload["image_sha256"],
        "has_text": ocr.has_text,
        "raw_text": raw_text,
        "normalized_text": normalize_search_text(raw_text),
        "blocks_json": json.dumps(
            [block.model_dump(mode="json") for block in ocr.blocks],
            ensure_ascii=False,
        ),
        "uncertainties_json": json.dumps(ocr.uncertainties, ensure_ascii=False),
        "model": payload["model"],
        "prompt_version": payload["prompt_version"],
        "attempts": payload["attempts"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "processed_at": payload["processed_at"],
        "usage_json": json.dumps(payload.get("usage"), ensure_ascii=False),
        "request_id": payload.get("request_id"),
        "batch_size": payload.get("batch_size"),
        "screening_signature": payload.get("screening_signature"),
        "screening_boxes_json": json.dumps(
            payload.get("screening_boxes") or [], ensure_ascii=False
        ),
    }


def _payload_matches_selection(
    payload: dict[str, Any],
    *,
    selected_signatures: dict[int, str] | None,
    model: str | None,
    prompt_version: str | None,
) -> bool:
    if model is not None and payload.get("model") != model:
        return False
    if prompt_version is not None and payload.get("prompt_version") != prompt_version:
        return False
    if selected_signatures is None:
        return True
    frame_id = int(payload.get("frame_id", -1))
    return payload.get("screening_signature") == selected_signatures.get(frame_id)


def write_summary_tables(
    output_dir: Path,
    *,
    selected_signatures: dict[int, str] | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    successful_frames: set[int] = set()
    for path in sorted((output_dir / "raw").glob("frame_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not _payload_matches_selection(
                payload,
                selected_signatures=selected_signatures,
                model=model,
                prompt_version=prompt_version,
            ):
                continue
            record = frame_record(payload)
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            continue
        records.append(record)
        successful_frames.add(int(record["frame_id"]))

    failures: list[dict[str, Any]] = []
    for path in sorted((output_dir / "failures").glob("frame_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not _payload_matches_selection(
            payload,
            selected_signatures=selected_signatures,
            model=model,
            prompt_version=prompt_version,
        ):
            continue
        if int(payload.get("frame_id", -1)) not in successful_frames:
            failures.append(payload)

    frame_columns = [
        "video_id",
        "shot_id",
        "frame_id",
        "pts_time",
        "image_path",
        "image_sha256",
        "has_text",
        "raw_text",
        "normalized_text",
        "blocks_json",
        "uncertainties_json",
        "model",
        "prompt_version",
        "attempts",
        "elapsed_seconds",
        "processed_at",
        "usage_json",
        "request_id",
        "batch_size",
        "screening_signature",
        "screening_boxes_json",
    ]
    failure_columns = [
        "video_id",
        "shot_id",
        "frame_id",
        "pts_time",
        "image_path",
        "image_sha256",
        "model",
        "prompt_version",
        "attempts",
        "error_category",
        "status_code",
        "retry_after_seconds",
        "terminal",
        "error_type",
        "error",
        "failed_at",
        "screening_signature",
    ]
    frames = pd.DataFrame(records, columns=frame_columns)
    if not frames.empty:
        frames = frames.sort_values(["video_id", "frame_id"]).reset_index(drop=True)
    failure_frame = pd.DataFrame(failures, columns=failure_columns)
    if not failure_frame.empty:
        failure_frame = failure_frame.sort_values(["video_id", "frame_id"]).reset_index(
            drop=True
        )

    atomic_write_parquet(frames, output_dir / "frames.parquet")
    atomic_write_parquet(failure_frame, output_dir / "failures.parquet")
    return frames, failure_frame
