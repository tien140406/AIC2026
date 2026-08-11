from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ..common.identity import seconds_to_ms

TRANSCRIPT_COLUMNS = [
    "segment_id",
    "video_id",
    "level",
    "start",
    "end",
    "start_ms",
    "end_ms",
    "language",
    "source",
    "source_type",
    "provider",
    "model",
    "raw_text",
    "clean_text",
    "shot_ids",
    "primary_shot",
    "keyframe_ids",
    "confidence",
    "youtube_id",
    "chunk_id",
    "quality_score",
    "created_at",
    "metadata_json",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TranscriptSegment:
    segment_id: str
    video_id: str
    start: float
    end: float
    raw_text: str
    clean_text: str = ""
    level: str = "L0"
    language: str | None = None
    source: str = ""
    source_type: str = ""
    provider: str = ""
    model: str | None = None
    shot_ids: tuple[int, ...] = ()
    primary_shot: int | None = None
    keyframe_ids: tuple[int, ...] = ()
    confidence: float | None = None
    youtube_id: str | None = None
    chunk_id: str | None = None
    quality_score: float | None = None
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.level not in {"L0", "L1"}:
            raise ValueError(f"Unsupported transcript level: {self.level}")
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"Invalid segment interval [{self.start}, {self.end})")
        if not self.raw_text.strip():
            raise ValueError("raw_text must not be empty")

    def with_updates(self, **changes: Any) -> TranscriptSegment:
        return replace(self, **changes)

    def to_record(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "video_id": self.video_id,
            "level": self.level,
            "start": float(self.start),
            "end": float(self.end),
            "start_ms": seconds_to_ms(self.start),
            "end_ms": seconds_to_ms(self.end),
            "language": self.language,
            "source": self.source,
            "source_type": self.source_type,
            "provider": self.provider,
            "model": self.model,
            "raw_text": self.raw_text,
            "clean_text": self.clean_text,
            "shot_ids": list(self.shot_ids),
            "primary_shot": self.primary_shot,
            "keyframe_ids": list(self.keyframe_ids),
            "confidence": self.confidence,
            "youtube_id": self.youtube_id,
            "chunk_id": self.chunk_id,
            "quality_score": self.quality_score,
            "created_at": self.created_at,
            "metadata_json": json.dumps(
                self.metadata, ensure_ascii=False, sort_keys=True
            ),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> TranscriptSegment:
        metadata_value = record.get("metadata_json")
        if isinstance(metadata_value, str) and metadata_value:
            metadata = json.loads(metadata_value)
        elif isinstance(record.get("metadata"), dict):
            metadata = dict(record["metadata"])
        else:
            metadata = {}

        def optional(value: Any) -> Any:
            if value is None:
                return None
            missing = pd.isna(value)
            if not hasattr(missing, "__len__") and bool(missing):
                return None
            return value

        def values(value: Any) -> tuple[Any, ...]:
            value = optional(value)
            if value is None:
                return ()
            return tuple(value)

        return cls(
            segment_id=str(record["segment_id"]),
            video_id=str(record["video_id"]),
            level=str(record.get("level", "L0")),
            start=float(record["start"]),
            end=float(record["end"]),
            language=optional(record.get("language")),
            source=str(record.get("source") or ""),
            source_type=str(record.get("source_type") or ""),
            provider=str(record.get("provider") or ""),
            model=optional(record.get("model")),
            raw_text=str(record["raw_text"]),
            clean_text=str(record.get("clean_text") or ""),
            shot_ids=tuple(int(value) for value in values(record.get("shot_ids"))),
            primary_shot=(
                int(record["primary_shot"])
                if optional(record.get("primary_shot")) is not None
                else None
            ),
            keyframe_ids=tuple(
                int(value) for value in values(record.get("keyframe_ids"))
            ),
            confidence=(
                float(record["confidence"])
                if optional(record.get("confidence")) is not None
                else None
            ),
            youtube_id=optional(record.get("youtube_id")),
            chunk_id=optional(record.get("chunk_id")),
            quality_score=(
                float(record["quality_score"])
                if optional(record.get("quality_score")) is not None
                else None
            ),
            created_at=str(record.get("created_at") or utc_now_iso()),
            metadata=metadata,
        )


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    segments: tuple[TranscriptSegment, ...]
    language: str | None = None
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QualityResult:
    accepted: bool
    score: float
    reasons: tuple[str, ...] = ()


def segments_to_frame(segments: Iterable[TranscriptSegment]) -> pd.DataFrame:
    records = [segment.to_record() for segment in segments]
    return pd.DataFrame(records, columns=TRANSCRIPT_COLUMNS)


def frame_to_segments(frame: pd.DataFrame) -> list[TranscriptSegment]:
    return [
        TranscriptSegment.from_record(record) for record in frame.to_dict("records")
    ]
