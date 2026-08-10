from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from itertools import pairwise

from .models import QualityResult, TranscriptSegment

ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
WHITESPACE_RE = re.compile(r"\s+")
SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([,.;:!?%])")
SPACE_AFTER_PUNCTUATION_RE = re.compile(r"([,.;:!?])(?=[^\s\d])")


def normalize_transcript_text(text: str) -> str:
    """Apply conservative formatting cleanup without changing wording."""
    normalized = unicodedata.normalize("NFC", text)
    normalized = ZERO_WIDTH_RE.sub("", normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    normalized = SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", normalized)
    normalized = SPACE_AFTER_PUNCTUATION_RE.sub(r"\1 ", normalized)
    return normalized


def normalize_segments(
    segments: Iterable[TranscriptSegment],
) -> list[TranscriptSegment]:
    return [
        segment.with_updates(clean_text=normalize_transcript_text(segment.raw_text))
        for segment in segments
    ]


def evaluate_transcript_quality(
    segments: Iterable[TranscriptSegment],
    *,
    video_duration: float | None = None,
    expected_language: str | None = None,
    min_total_characters: int = 20,
    min_characters_per_minute: float = 5.0,
) -> QualityResult:
    materialized = list(segments)
    ordered = sorted(materialized, key=lambda item: (item.start, item.end))
    reasons: list[str] = []
    if not ordered:
        return QualityResult(False, 0.0, ("no_segments",))

    clean_texts = [normalize_transcript_text(item.raw_text) for item in ordered]
    character_count = sum(len(text) for text in clean_texts)
    if character_count < min_total_characters:
        reasons.append("too_few_characters")

    if video_duration and video_duration > 0:
        required = max(
            min_total_characters,
            int((video_duration / 60.0) * min_characters_per_minute),
        )
        if character_count < required:
            reasons.append("insufficient_text_for_duration")
        outside = sum(
            item.start > video_duration + 2.0 or item.end > video_duration + 10.0
            for item in ordered
        )
        if outside / len(ordered) > 0.1:
            reasons.append("timestamps_outside_video")

    non_monotonic = sum(
        current.start < previous.start for previous, current in pairwise(materialized)
    )
    if non_monotonic:
        reasons.append("non_monotonic_timestamps")

    if expected_language:
        expected = expected_language.lower().split("-")[0]
        declared = {
            item.language.lower().split("-")[0] for item in ordered if item.language
        }
        if declared and expected not in declared:
            reasons.append("language_mismatch")

    text_factor = min(1.0, character_count / max(min_total_characters * 4, 1))
    segment_factor = min(1.0, len(ordered) / 8.0)
    score = max(0.0, min(1.0, 0.75 * text_factor + 0.25 * segment_factor))
    if reasons:
        score *= 0.25
    return QualityResult(not reasons, round(score, 6), tuple(reasons))


def build_semantic_chunks(
    segments: Iterable[TranscriptSegment],
    *,
    max_duration: float = 30.0,
    max_characters: int = 600,
    max_gap: float = 2.0,
    target_duration: float = 12.0,
) -> list[TranscriptSegment]:
    """Group adjacent L0 segments deterministically while preserving their IDs."""
    if max_duration <= 0 or max_characters <= 0 or max_gap < 0:
        raise ValueError("Invalid semantic chunking limits")
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    if not ordered:
        return []

    groups: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []
    for segment in ordered:
        if not current:
            current = [segment]
            continue
        proposed_duration = segment.end - current[0].start
        proposed_characters = sum(
            len(item.clean_text or item.raw_text) for item in current
        )
        proposed_characters += len(segment.clean_text or segment.raw_text) + 1
        gap = segment.start - current[-1].end
        boundary = (
            (current[-1].clean_text or current[-1].raw_text)
            .rstrip()
            .endswith((".", "!", "?", ":", ";"))
        )
        should_split = (
            gap > max_gap
            or proposed_duration > max_duration
            or proposed_characters > max_characters
            or (boundary and current[-1].end - current[0].start >= target_duration)
        )
        if should_split:
            groups.append(current)
            current = [segment]
        else:
            current.append(segment)
    groups.append(current)

    chunks: list[TranscriptSegment] = []
    for index, group in enumerate(groups):
        first = group[0]
        chunk_id = f"{first.video_id}:L1:{index:06d}"
        languages = {item.language for item in group if item.language}
        sources = {item.source for item in group}
        source_types = {item.source_type for item in group}
        providers = {item.provider for item in group}
        models = {item.model for item in group}
        confidences = [item.confidence for item in group if item.confidence is not None]
        quality_scores = [
            item.quality_score for item in group if item.quality_score is not None
        ]
        chunks.append(
            TranscriptSegment(
                segment_id=chunk_id,
                video_id=first.video_id,
                level="L1",
                start=first.start,
                end=group[-1].end,
                language=next(iter(languages)) if len(languages) == 1 else None,
                source=next(iter(sources)) if len(sources) == 1 else "mixed",
                source_type=(
                    next(iter(source_types)) if len(source_types) == 1 else "mixed"
                ),
                provider=next(iter(providers)) if len(providers) == 1 else "mixed",
                model=next(iter(models)) if len(models) == 1 else None,
                raw_text=" ".join(item.raw_text.strip() for item in group),
                clean_text=" ".join(
                    (item.clean_text or item.raw_text).strip() for item in group
                ),
                confidence=(
                    sum(confidences) / len(confidences) if confidences else None
                ),
                youtube_id=first.youtube_id,
                chunk_id=chunk_id,
                quality_score=(
                    sum(quality_scores) / len(quality_scores)
                    if quality_scores
                    else None
                ),
                metadata={
                    "constituent_segment_ids": [item.segment_id for item in group]
                },
            )
        )
    return chunks
