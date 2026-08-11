from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..asr.models import TranscriptSegment
from ..common.identity import ensure_frame_identity, frame_uid, seconds_to_ms
from ..common.paths import PipelinePaths

DEFAULT_SELECTION_ENCODER_ID = "timm_beit3_base_patch16_224_pt"


def optional(value: Any) -> Any:
    if value is None:
        return None
    missing = pd.isna(value)
    if not hasattr(missing, "__len__") and bool(missing):
        return None
    return value


def json_list(value: Any) -> list[Any]:
    value = optional(value)
    if value is None:
        return []
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return list(value)


def portable_path(path: str | Path | None, root: Path) -> str | None:
    if optional(path) is None:
        return None
    resolved = Path(str(path)).expanduser().resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def load_keyframe_documents(
    paths: PipelinePaths,
    *,
    selection_encoder_id: str = DEFAULT_SELECTION_ENCODER_ID,
) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    selection_root = paths.keyframes / selection_encoder_id
    for manifest_path in sorted(selection_root.glob("*.parquet")):
        if manifest_path.name.endswith(".audit.parquet"):
            continue
        frame = ensure_frame_identity(pd.read_parquet(manifest_path))
        if "selected" in frame.columns:
            frame = frame.loc[frame["selected"].astype(bool)].copy()
        video_id = manifest_path.stem
        jpg_manifest = (
            paths.workspace / "keyframe_images" / video_id / "manifest.parquet"
        )
        if jpg_manifest.is_file():
            images = ensure_frame_identity(pd.read_parquet(jpg_manifest))
            columns = [
                column
                for column in ("frame_uid", "image_path", "image_sha256")
                if column in images.columns
            ]
            frame = frame.drop(
                columns=[
                    column
                    for column in ("image_path", "image_sha256")
                    if column in frame.columns
                ]
            ).merge(images[columns], on="frame_uid", how="left", validate="one_to_one")

        for record in frame.to_dict(orient="records"):
            uid = str(record["frame_uid"])
            documents[uid] = {
                "frame_uid": uid,
                "video_id": str(record["video_id"]),
                "frame_id": int(record["frame_id"]),
                "shot_id": (
                    int(record["shot_id"])
                    if optional(record.get("shot_id")) is not None
                    else None
                ),
                "pts_ms": (
                    int(record["pts_ms"])
                    if optional(record.get("pts_ms")) is not None
                    else seconds_to_ms(record.get("pts_time"))
                ),
                "image_path": portable_path(record.get("image_path"), paths.root),
                "image_sha256": optional(record.get("image_sha256")),
                "selection_encoder_id": selection_encoder_id,
                "selection_version": str(
                    record.get("selection_version") or selection_encoder_id
                ),
                "source_manifest": portable_path(manifest_path, paths.root),
                "ocr": [],
                "ocr_text": "",
                "objects": [],
                "object_classes": [],
                "object_text": "",
            }
    return documents


def _append_ocr_documents(
    paths: PipelinePaths, documents: dict[str, dict[str, Any]]
) -> None:
    preprocess_root = paths.workspace / "ocr" / "preprocess"
    for manifest_path in sorted(preprocess_root.glob("*/screening.parquet")):
        frame = ensure_frame_identity(pd.read_parquet(manifest_path))
        for record in frame.to_dict(orient="records"):
            uid = str(record["frame_uid"])
            document = documents.get(uid)
            if document is None:
                continue
            if optional(record.get("image_path")) is not None:
                document["image_path"] = portable_path(
                    record.get("image_path"), paths.root
                )
            if optional(record.get("image_sha256")) is not None:
                document["image_sha256"] = str(record["image_sha256"])
            document["ocr"].append(
                {
                    "stage": "screening",
                    "model": str(record.get("model_name") or "ppocr"),
                    "version": str(record.get("preprocess_version") or "screening"),
                    "status": (
                        "candidate"
                        if bool(record.get("screening_has_text"))
                        else "screened_out"
                    ),
                    "has_text": bool(record.get("screening_has_text")),
                    "raw_text": None,
                    "normalized_text": None,
                    "processed_at": optional(record.get("processed_at")),
                    "blocks": [],
                }
            )

    gemini_root = paths.workspace / "ocr" / "gemini"
    for frame_path in sorted(gemini_root.glob("*/*/frames.parquet")):
        frame = ensure_frame_identity(pd.read_parquet(frame_path))
        for record in frame.to_dict(orient="records"):
            uid = str(record["frame_uid"])
            document = documents.get(uid)
            if document is None:
                continue
            blocks = []
            for block_id, block in enumerate(json_list(record.get("blocks_json"))):
                y_min, x_min, y_max, x_max = map(int, block["bbox_2d"])
                blocks.append(
                    {
                        "block_id": block_id,
                        "text": str(block["text"]),
                        "legibility": optional(block.get("legibility")),
                        "y_min": y_min,
                        "x_min": x_min,
                        "y_max": y_max,
                        "x_max": x_max,
                    }
                )
            normalized_text = str(record.get("normalized_text") or "").strip()
            raw_text = str(record.get("raw_text") or normalized_text).strip()
            document["ocr"].append(
                {
                    "stage": "ocr",
                    "model": str(record["model"]),
                    "version": str(record["prompt_version"]),
                    "status": "success",
                    "has_text": bool(record.get("has_text")),
                    "raw_text": raw_text or None,
                    "normalized_text": normalized_text or None,
                    "processed_at": optional(record.get("processed_at")),
                    "blocks": blocks,
                }
            )
            if normalized_text:
                existing = [value for value in document["ocr_text"].split("\n") if value]
                if normalized_text not in existing:
                    existing.append(normalized_text)
                document["ocr_text"] = "\n".join(existing)

    for failure_path in sorted(gemini_root.glob("*/*/failures.parquet")):
        frame = ensure_frame_identity(pd.read_parquet(failure_path))
        for record in frame.to_dict(orient="records"):
            uid = str(record["frame_uid"])
            document = documents.get(uid)
            if document is None:
                continue
            document["ocr"].append(
                {
                    "stage": "ocr",
                    "model": str(record.get("model") or "unknown"),
                    "version": str(record.get("prompt_version") or "unknown"),
                    "status": "failure",
                    "has_text": None,
                    "raw_text": None,
                    "normalized_text": None,
                    "processed_at": optional(record.get("failed_at")),
                    "blocks": [],
                }
            )


def _append_object_documents(
    paths: PipelinePaths, documents: dict[str, dict[str, Any]]
) -> None:
    detections_by_run: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    object_root = paths.workspace / "objects"
    for detection_path in sorted(object_root.glob("*/*/detections.parquet")):
        frame = ensure_frame_identity(pd.read_parquet(detection_path))
        for record in frame.to_dict(orient="records"):
            uid = str(record["frame_uid"])
            if uid not in documents:
                continue
            model_id = str(record["model_id"])
            run_signature = str(record["run_signature"])
            detections_by_run[(uid, model_id, run_signature)].append(
                {
                    "detection_id": int(record["detection_id"]),
                    "class_id": int(record["class_id"]),
                    "class_name": str(record["class_name"]),
                    "score": float(record["score"]),
                    "x_min": float(record["x_min"]),
                    "y_min": float(record["y_min"]),
                    "x_max": float(record["x_max"]),
                    "y_max": float(record["y_max"]),
                    "x_min_1000": int(record["x_min_1000"]),
                    "y_min_1000": int(record["y_min_1000"]),
                    "x_max_1000": int(record["x_max_1000"]),
                    "y_max_1000": int(record["y_max_1000"]),
                    "area_ratio": float(record["area_ratio"]),
                }
            )

    seen_runs: set[tuple[str, str, str]] = set()
    for frame_path in sorted(object_root.glob("*/*/frames.parquet")):
        frame = ensure_frame_identity(pd.read_parquet(frame_path))
        for record in frame.to_dict(orient="records"):
            uid = str(record["frame_uid"])
            document = documents.get(uid)
            if document is None:
                continue
            model_id = str(record["model_id"])
            run_signature = str(record["run_signature"])
            key = (uid, model_id, run_signature)
            detections = sorted(
                detections_by_run.get(key, []), key=lambda item: item["detection_id"]
            )
            document["objects"].append(
                {
                    "model_id": model_id,
                    "run_signature": run_signature,
                    "raw_detection_count": int(record["raw_detection_count"]),
                    "filtered_detection_count": int(
                        record["filtered_detection_count"]
                    ),
                    "object_signature": str(record.get("object_signature") or "{}"),
                    "processed_at": optional(record.get("processed_at")),
                    "detections": detections,
                }
            )
            seen_runs.add(key)

    # Preserve detections even if an older artifact has no frames.parquet summary.
    for key, detections in detections_by_run.items():
        if key in seen_runs:
            continue
        uid, model_id, run_signature = key
        documents[uid]["objects"].append(
            {
                "model_id": model_id,
                "run_signature": run_signature,
                "raw_detection_count": len(detections),
                "filtered_detection_count": len(detections),
                "object_signature": "{}",
                "processed_at": None,
                "detections": sorted(
                    detections, key=lambda item: item["detection_id"]
                ),
            }
        )

    for document in documents.values():
        classes = sorted(
            {
                detection["class_name"]
                for run in document["objects"]
                for detection in run["detections"]
            }
        )
        document["object_classes"] = classes
        document["object_text"] = " ".join(classes)


def iter_frame_documents(
    paths: PipelinePaths,
    *,
    selection_encoder_id: str = DEFAULT_SELECTION_ENCODER_ID,
) -> Iterator[dict[str, Any]]:
    documents = load_keyframe_documents(
        paths, selection_encoder_id=selection_encoder_id
    )
    _append_ocr_documents(paths, documents)
    _append_object_documents(paths, documents)
    for uid in sorted(documents):
        yield documents[uid]


def iter_asr_documents(paths: PipelinePaths) -> Iterator[dict[str, Any]]:
    for transcript_path in sorted(paths.transcripts.glob("*/l[01].parquet")):
        frame = pd.read_parquet(transcript_path)
        for record in frame.to_dict(orient="records"):
            segment = TranscriptSegment.from_record(record)
            yield {
                "segment_id": segment.segment_id,
                "video_id": segment.video_id,
                "level": segment.level,
                "start_ms": seconds_to_ms(segment.start),
                "end_ms": seconds_to_ms(segment.end),
                "language": segment.language,
                "source": segment.source,
                "source_type": segment.source_type,
                "provider": segment.provider,
                "model": segment.model,
                "raw_text": segment.raw_text,
                "clean_text": segment.clean_text or segment.raw_text,
                "shot_ids": list(segment.shot_ids),
                "primary_shot": segment.primary_shot,
                "keyframe_ids": list(segment.keyframe_ids),
                "keyframe_uids": [
                    frame_uid(segment.video_id, value)
                    for value in segment.keyframe_ids
                ],
                "confidence": segment.confidence,
                "chunk_id": segment.chunk_id,
                "quality_score": segment.quality_score,
                "created_at": segment.created_at,
                "source_manifest": portable_path(transcript_path, paths.root),
            }
