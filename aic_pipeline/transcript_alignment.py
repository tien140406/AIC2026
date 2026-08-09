from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .transcript_models import TranscriptSegment


@dataclass(frozen=True)
class ShotInterval:
    shot_id: int
    start: float
    end: float
    start_frame: int
    end_frame_exclusive: int


def load_shot_intervals(path: Path, fps: float) -> list[ShotInterval]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("ranges_end_exclusive", False):
        raise ValueError("Expected half-open shot ranges [start, end)")
    intervals: list[ShotInterval] = []
    previous_end = 0
    for shot_id, pair in enumerate(payload.get("ranges") or []):
        if len(pair) != 2:
            raise ValueError(f"Invalid shot range at index {shot_id}: {pair}")
        start_frame, end_frame = map(int, pair)
        if start_frame < previous_end or end_frame <= start_frame:
            raise ValueError(f"Invalid or unsorted shot range: {pair}")
        intervals.append(
            ShotInterval(
                shot_id=shot_id,
                start=start_frame / fps,
                end=end_frame / fps,
                start_frame=start_frame,
                end_frame_exclusive=end_frame,
            )
        )
        previous_end = end_frame
    return intervals


def load_keyframe_mapping(
    path: Path | None,
    *,
    video_id: str,
    shots: Iterable[ShotInterval],
) -> dict[int, tuple[int, ...]]:
    shot_list = list(shots)
    mapping: dict[int, list[int]] = {shot.shot_id: [] for shot in shot_list}
    if path is None or not path.exists():
        return {shot_id: () for shot_id in mapping}

    frame = pd.read_parquet(path)
    if "video_id" in frame.columns:
        frame = frame.loc[frame["video_id"] == video_id]
    if "selected" in frame.columns:
        frame = frame.loc[frame["selected"].astype(bool)]
    if "frame_id" not in frame.columns:
        raise ValueError(f"Keyframe manifest has no frame_id column: {path}")

    if "shot_id" in frame.columns:
        for row in frame[["shot_id", "frame_id"]].itertuples(index=False):
            if pd.isna(row.shot_id):
                continue
            shot_id = int(row.shot_id)
            if shot_id in mapping:
                mapping[shot_id].append(int(row.frame_id))
    else:
        ordered_frames = sorted(int(value) for value in frame["frame_id"].dropna())
        shot_index = 0
        for frame_id in ordered_frames:
            while (
                shot_index < len(shot_list)
                and frame_id >= shot_list[shot_index].end_frame_exclusive
            ):
                shot_index += 1
            if shot_index >= len(shot_list):
                break
            shot = shot_list[shot_index]
            if shot.start_frame <= frame_id < shot.end_frame_exclusive:
                mapping[shot.shot_id].append(frame_id)

    return {
        shot_id: tuple(sorted(set(frame_ids))) for shot_id, frame_ids in mapping.items()
    }


def align_segments(
    segments: Iterable[TranscriptSegment],
    shots: Iterable[ShotInterval],
    keyframes_by_shot: dict[int, tuple[int, ...]] | None = None,
) -> list[TranscriptSegment]:
    """Align sorted transcript and shot intervals in O(n + m + overlaps)."""
    ordered_segments = sorted(segments, key=lambda item: (item.start, item.end))
    ordered_shots = sorted(shots, key=lambda item: (item.start, item.end))
    keyframes_by_shot = keyframes_by_shot or {}
    aligned: list[TranscriptSegment] = []
    left = 0

    for segment in ordered_segments:
        while left < len(ordered_shots) and ordered_shots[left].end <= segment.start:
            left += 1
        overlaps: list[tuple[ShotInterval, float]] = []
        cursor = left
        while cursor < len(ordered_shots) and ordered_shots[cursor].start < segment.end:
            shot = ordered_shots[cursor]
            overlap = min(segment.end, shot.end) - max(segment.start, shot.start)
            if overlap > 1e-9:
                overlaps.append((shot, overlap))
            cursor += 1

        shot_ids = tuple(item.shot_id for item, _ in overlaps)
        primary = (
            max(overlaps, key=lambda item: (item[1], -item[0].shot_id))[0].shot_id
            if overlaps
            else None
        )
        keyframe_ids = tuple(
            sorted(
                {
                    frame_id
                    for shot_id in shot_ids
                    for frame_id in keyframes_by_shot.get(shot_id, ())
                }
            )
        )
        aligned.append(
            segment.with_updates(
                shot_ids=shot_ids,
                primary_shot=primary,
                keyframe_ids=keyframe_ids,
            )
        )
    return aligned
