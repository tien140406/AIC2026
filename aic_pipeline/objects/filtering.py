from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..common.io import atomic_write_json, atomic_write_parquet
from .schema import valid_raw_payload

DEFAULT_SCORE_THRESHOLD = 0.30
DEFAULT_MIN_AREA_RATIO = 0.0005
DEFAULT_DUPLICATE_IOU = 0.90
DEFAULT_MAX_DETECTIONS = 100


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bbox_iou(first: list[float], second: list[float]) -> float:
    x_min = max(float(first[0]), float(second[0]))
    y_min = max(float(first[1]), float(second[1]))
    x_max = min(float(first[2]), float(second[2]))
    y_max = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def validate_filter_options(
    *,
    score_threshold: float,
    min_area_ratio: float,
    duplicate_iou: float,
    max_detections: int,
) -> None:
    if not 0 <= score_threshold <= 1:
        raise ValueError("score_threshold must be between 0 and 1")
    if not 0 <= min_area_ratio <= 1:
        raise ValueError("min_area_ratio must be between 0 and 1")
    if not 0 <= duplicate_iou <= 1:
        raise ValueError("duplicate_iou must be between 0 and 1")
    if max_detections <= 0:
        raise ValueError("max_detections must be positive")


def filter_detections(
    detections: list[dict[str, Any]],
    *,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    min_area_ratio: float = DEFAULT_MIN_AREA_RATIO,
    duplicate_iou: float = DEFAULT_DUPLICATE_IOU,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
) -> list[dict[str, Any]]:
    validate_filter_options(
        score_threshold=score_threshold,
        min_area_ratio=min_area_ratio,
        duplicate_iou=duplicate_iou,
        max_detections=max_detections,
    )
    candidates = sorted(
        (
            detection
            for detection in detections
            if float(detection["score"]) >= score_threshold
            and float(detection["area_ratio"]) >= min_area_ratio
        ),
        key=lambda detection: float(detection["score"]),
        reverse=True,
    )

    kept: list[dict[str, Any]] = []
    for detection in candidates:
        duplicate = any(
            int(previous["class_id"]) == int(detection["class_id"])
            and bbox_iou(previous["bbox_xyxy"], detection["bbox_xyxy"]) >= duplicate_iou
            for previous in kept
        )
        if duplicate:
            continue
        kept.append(detection)
        if len(kept) >= max_detections:
            break
    return kept


def _frame_summary(
    payload: dict[str, Any],
    filtered: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter(str(item["class_name"]) for item in filtered)
    scores: dict[str, list[float]] = defaultdict(list)
    for item in filtered:
        scores[str(item["class_name"])].append(float(item["score"]))
    max_scores = {
        name: max(values) for name, values in sorted(scores.items()) if values
    }
    ordered_counts = dict(sorted(counts.items()))
    return {
        "video_id": payload["video_id"],
        "shot_id": payload.get("shot_id"),
        "frame_id": int(payload["frame_id"]),
        "pts_time": payload.get("pts_time"),
        "image_path": payload["image_path"],
        "image_sha256": payload["image_sha256"],
        "image_width": int(payload["image_width"]),
        "image_height": int(payload["image_height"]),
        "raw_detection_count": len(payload["detections"]),
        "filtered_detection_count": len(filtered),
        "object_labels_json": json.dumps(sorted(ordered_counts), ensure_ascii=False),
        "object_counts_json": json.dumps(ordered_counts, ensure_ascii=False),
        "max_scores_json": json.dumps(max_scores, ensure_ascii=False),
        "object_signature": json.dumps(
            ordered_counts, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        "detections_json": json.dumps(filtered, ensure_ascii=False),
        "model_id": payload["model_id"],
        "run_signature": payload["run_signature"],
        "processed_at": payload["processed_at"],
    }


def _detection_rows(
    payload: dict[str, Any],
    filtered: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for detection_id, detection in enumerate(filtered):
        x_min, y_min, x_max, y_max = detection["bbox_xyxy"]
        y_min_1000, x_min_1000, y_max_1000, x_max_1000 = detection["bbox_yxyx_1000"]
        rows.append(
            {
                "video_id": payload["video_id"],
                "shot_id": payload.get("shot_id"),
                "frame_id": int(payload["frame_id"]),
                "pts_time": payload.get("pts_time"),
                "detection_id": detection_id,
                "class_id": int(detection["class_id"]),
                "class_name": str(detection["class_name"]),
                "score": float(detection["score"]),
                "x_min": float(x_min),
                "y_min": float(y_min),
                "x_max": float(x_max),
                "y_max": float(y_max),
                "x_min_1000": int(x_min_1000),
                "y_min_1000": int(y_min_1000),
                "x_max_1000": int(x_max_1000),
                "y_max_1000": int(y_max_1000),
                "area_ratio": float(detection["area_ratio"]),
                "image_path": payload["image_path"],
                "image_sha256": payload["image_sha256"],
                "model_id": payload["model_id"],
                "run_signature": payload["run_signature"],
            }
        )
    return rows


def write_summary_tables(
    output_dir: Path,
    *,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    min_area_ratio: float = DEFAULT_MIN_AREA_RATIO,
    duplicate_iou: float = DEFAULT_DUPLICATE_IOU,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
    selected_ids: set[int] | None = None,
    run_signature: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    validate_filter_options(
        score_threshold=score_threshold,
        min_area_ratio=min_area_ratio,
        duplicate_iou=duplicate_iou,
        max_detections=max_detections,
    )
    frame_records: list[dict[str, Any]] = []
    detection_records: list[dict[str, Any]] = []
    invalid_raw_files = 0
    for path in sorted((output_dir / "raw").glob("frame_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid_raw_files += 1
            continue
        frame_id = int(payload.get("frame_id", -1))
        if selected_ids is not None and frame_id not in selected_ids:
            continue
        if run_signature is not None and payload.get("run_signature") != run_signature:
            continue
        if not valid_raw_payload(payload):
            invalid_raw_files += 1
            continue
        filtered = filter_detections(
            payload["detections"],
            score_threshold=score_threshold,
            min_area_ratio=min_area_ratio,
            duplicate_iou=duplicate_iou,
            max_detections=max_detections,
        )
        frame_records.append(_frame_summary(payload, filtered))
        detection_records.extend(_detection_rows(payload, filtered))

    frame_columns = [
        "video_id",
        "shot_id",
        "frame_id",
        "pts_time",
        "image_path",
        "image_sha256",
        "image_width",
        "image_height",
        "raw_detection_count",
        "filtered_detection_count",
        "object_labels_json",
        "object_counts_json",
        "max_scores_json",
        "object_signature",
        "detections_json",
        "model_id",
        "run_signature",
        "processed_at",
    ]
    detection_columns = [
        "video_id",
        "shot_id",
        "frame_id",
        "pts_time",
        "detection_id",
        "class_id",
        "class_name",
        "score",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "x_min_1000",
        "y_min_1000",
        "x_max_1000",
        "y_max_1000",
        "area_ratio",
        "image_path",
        "image_sha256",
        "model_id",
        "run_signature",
    ]
    frames = pd.DataFrame(frame_records, columns=frame_columns)
    detections = pd.DataFrame(detection_records, columns=detection_columns)
    if not frames.empty:
        frames = frames.sort_values(["video_id", "frame_id"]).reset_index(drop=True)
    if not detections.empty:
        detections = detections.sort_values(
            ["video_id", "frame_id", "detection_id"]
        ).reset_index(drop=True)
    atomic_write_parquet(frames, output_dir / "frames.parquet")
    atomic_write_parquet(detections, output_dir / "detections.parquet")

    report = {
        "filtered_at": utc_now(),
        "frames": len(frames),
        "frames_with_objects": int((frames["filtered_detection_count"] > 0).sum())
        if not frames.empty
        else 0,
        "raw_detections": int(frames["raw_detection_count"].sum())
        if not frames.empty
        else 0,
        "filtered_detections": len(detections),
        "invalid_raw_files": invalid_raw_files,
        "parameters": {
            "score_threshold": score_threshold,
            "min_area_ratio": min_area_ratio,
            "duplicate_iou": duplicate_iou,
            "max_detections": max_detections,
        },
        "outputs": {
            "frames": str(output_dir / "frames.parquet"),
            "detections": str(output_dir / "detections.parquet"),
        },
    }
    atomic_write_json(report, output_dir / "filter_report.json")
    return frames, detections, report
