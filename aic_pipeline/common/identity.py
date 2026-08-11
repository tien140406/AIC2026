from __future__ import annotations

import hashlib
import math
from typing import Any

import pandas as pd


def frame_uid(video_id: str, frame_id: int) -> str:
    video_id = str(video_id).strip()
    if not video_id:
        raise ValueError("video_id must not be empty")
    frame_id = int(frame_id)
    if frame_id < 0:
        raise ValueError("frame_id must be non-negative")
    return f"{video_id}:{frame_id:09d}"


def roi_uid(
    parent_frame_uid: str,
    model_id: str,
    run_signature: str,
    detection_id: int,
) -> str:
    parent_frame_uid = str(parent_frame_uid).strip()
    model_id = str(model_id).strip()
    run_signature = str(run_signature).strip()
    detection_id = int(detection_id)
    if not parent_frame_uid or not model_id or not run_signature:
        raise ValueError("ROI identity fields must not be empty")
    if detection_id < 0:
        raise ValueError("detection_id must be non-negative")
    version = hashlib.sha256(f"{model_id}\0{run_signature}".encode()).hexdigest()[:16]
    return f"{parent_frame_uid}:roi:{version}:{detection_id:04d}"


def seconds_to_ms(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"Invalid non-negative timestamp: {value}")
    return round(seconds * 1000)


def ensure_frame_identity(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"video_id", "frame_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Frame metadata is missing identity columns: {missing}")

    enriched = frame.copy()
    enriched["frame_uid"] = [
        frame_uid(video_id, frame_id)
        for video_id, frame_id in zip(enriched["video_id"], enriched["frame_id"])
    ]
    if "pts_time" in enriched.columns:
        derived = enriched["pts_time"].map(seconds_to_ms).astype("Int64")
        if "pts_ms" in enriched.columns:
            enriched["pts_ms"] = enriched["pts_ms"].astype("Int64").fillna(derived)
        else:
            enriched["pts_ms"] = derived
    elif "pts_ms" not in enriched.columns and "fps" in enriched.columns:
        enriched["pts_ms"] = [
            seconds_to_ms(int(frame_id) / float(fps))
            for frame_id, fps in zip(enriched["frame_id"], enriched["fps"])
        ]
        enriched["pts_ms"] = enriched["pts_ms"].astype("Int64")
    return enriched
