from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image

from ..common.identity import ensure_frame_identity, roi_uid
from .openclip import (
    DEFAULT_ENCODER_ID,
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED,
    PROJECT_ROOT,
    encode,
)

DEFAULT_OBJECT_MODEL_ID = "co_dino_5scale_lsj_r50_3x_coco"
DEFAULT_ROI_ENCODER_ID = f"{DEFAULT_ENCODER_ID}_roi"


def expand_bbox_xyxy(
    bbox: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if padding_ratio < 0:
        raise ValueError("padding ratio must be non-negative")
    x_min, y_min, x_max, y_max = map(float, bbox)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Invalid ROI box: {bbox}")
    pad_x = (x_max - x_min) * padding_ratio
    pad_y = (y_max - y_min) * padding_ratio
    left = max(0, math.floor(x_min - pad_x))
    top = max(0, math.floor(y_min - pad_y))
    right = min(image_width, math.ceil(x_max + pad_x))
    bottom = min(image_height, math.ceil(y_max + pad_y))
    if right <= left or bottom <= top:
        raise ValueError(f"ROI box is empty after clipping: {bbox}")
    return left, top, right, bottom


def load_roi_jobs(
    detections_path: Path,
    image_manifest_path: Path,
    *,
    video_id: str,
    min_score: float = 0.30,
    min_area_ratio: float = 0.0005,
    max_rois: int | None = None,
) -> pd.DataFrame:
    if not 0 <= min_score <= 1:
        raise ValueError("min score must be between 0 and 1")
    if min_area_ratio < 0:
        raise ValueError("min area ratio must be non-negative")
    if max_rois is not None and max_rois <= 0:
        raise ValueError("max ROIs must be positive")

    detections = ensure_frame_identity(pd.read_parquet(detections_path))
    required_detection_columns = {
        "model_id",
        "run_signature",
        "detection_id",
        "class_id",
        "class_name",
        "score",
        "area_ratio",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "x_min_1000",
        "y_min_1000",
        "x_max_1000",
        "y_max_1000",
    }
    missing = sorted(required_detection_columns - set(detections.columns))
    if missing:
        raise ValueError(f"Object detections are missing columns: {missing}")
    detections = detections.loc[
        (detections["video_id"].astype(str) == video_id)
        & (detections["score"].astype(float) >= min_score)
        & (detections["area_ratio"].astype(float) >= min_area_ratio)
    ].copy()
    images = ensure_frame_identity(pd.read_parquet(image_manifest_path))
    if "image_path" not in images.columns:
        raise ValueError("Final-keyframe image manifest is missing image_path")
    image_columns = [
        column
        for column in ("frame_uid", "image_path", "image_sha256")
        if column in images.columns
    ]
    jobs = detections.drop(
        columns=[
            column
            for column in ("image_path", "image_sha256")
            if column in detections.columns
        ]
    ).merge(images[image_columns], on="frame_uid", how="left", validate="many_to_one")
    if jobs.empty:
        raise ValueError("No object detections matched the ROI embedding filters")
    if jobs["image_path"].isna().any():
        missing = jobs.loc[jobs["image_path"].isna(), "frame_uid"].iloc[0]
        raise ValueError(f"Missing final-keyframe image for ROI parent: {missing}")
    jobs["roi_uid"] = [
        roi_uid(frame, model, signature, detection)
        for frame, model, signature, detection in zip(
            jobs["frame_uid"],
            jobs["model_id"],
            jobs["run_signature"],
            jobs["detection_id"],
        )
    ]
    if jobs["roi_uid"].duplicated().any():
        raise ValueError("Duplicate ROI identities in object detections")
    jobs = jobs.sort_values(
        ["frame_id", "model_id", "detection_id"]
    ).reset_index(drop=True)
    if max_rois is not None:
        jobs = jobs.iloc[:max_rois].copy()
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract OpenCLIP embeddings from filtered object ROI crops."
    )
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--model-id", default=DEFAULT_OBJECT_MODEL_ID)
    parser.add_argument("--detections", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--encoder-id", default=DEFAULT_ROI_ENCODER_ID)
    parser.add_argument("--embedding-model", default=DEFAULT_ENCODER_ID)
    parser.add_argument("--embedding-version", default=DEFAULT_ENCODER_ID)
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=PROJECT_ROOT / "models" / "openclip",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--padding-ratio", type=float, default=0.05)
    parser.add_argument("--min-score", type=float, default=0.30)
    parser.add_argument("--min-area-ratio", type=float, default=0.0005)
    parser.add_argument("--max-rois", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.padding_ratio < 0:
        raise ValueError("padding ratio must be non-negative")

    detections_path = args.detections or (
        PROJECT_ROOT
        / "workspace"
        / "objects"
        / args.model_id
        / args.video_id
        / "detections.parquet"
    )
    image_manifest_path = args.manifest or (
        PROJECT_ROOT
        / "workspace"
        / "keyframe_images"
        / args.video_id
        / "manifest.parquet"
    )
    jobs = load_roi_jobs(
        detections_path,
        image_manifest_path,
        video_id=args.video_id,
        min_score=args.min_score,
        min_area_ratio=args.min_area_ratio,
        max_rois=args.max_rois,
    )

    output_dir = PROJECT_ROOT / "workspace" / "features" / args.encoder_id
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".smoke{len(jobs)}" if args.max_rois is not None else ""
    matrix_path = output_dir / f"{args.video_id}{suffix}.f16.npy"
    mapping_path = output_dir / f"{args.video_id}{suffix}.parquet"
    if matrix_path.exists() and not args.force:
        raise FileExistsError(f"Output exists: {matrix_path}. Use --force to rebuild.")

    started = time.time()
    device = torch.device("cuda")
    args.model_cache.mkdir(parents=True, exist_ok=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
        device=device,
        precision="fp16",
        cache_dir=str(args.model_cache),
    )
    model.eval()
    feature_dim = int(model.visual.output_dim)
    torch.cuda.reset_peak_memory_stats()

    temporary = matrix_path.with_suffix(matrix_path.suffix + ".tmp")
    matrix = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float16, shape=(len(jobs), feature_dim)
    )
    images: list[torch.Tensor] = []
    rows: list[int] = []
    crop_boxes: list[tuple[int, int, int, int] | None] = [None] * len(jobs)
    extracted = 0

    def flush(target: np.ndarray) -> None:
        nonlocal extracted
        if not images:
            return
        target[np.asarray(rows, dtype=np.int64)] = encode(images, model, device).astype(
            np.float16
        )
        extracted += len(images)
        images.clear()
        rows.clear()

    for row_index, record in enumerate(jobs.to_dict(orient="records")):
        image_path = Path(str(record["image_path"]))
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as source:
            source = source.convert("RGB")
            crop_box = expand_bbox_xyxy(
                (
                    record["x_min"],
                    record["y_min"],
                    record["x_max"],
                    record["y_max"],
                ),
                image_width=source.width,
                image_height=source.height,
                padding_ratio=args.padding_ratio,
            )
            crop_boxes[row_index] = crop_box
            images.append(preprocess(source.crop(crop_box)))
        rows.append(row_index)
        if len(images) >= args.batch_size:
            flush(matrix)
            if extracted % 1024 < args.batch_size:
                print(f"Embedded {extracted:,}/{len(jobs):,} ROIs", flush=True)
    flush(matrix)
    if extracted != len(jobs):
        raise RuntimeError(f"Extracted {extracted}/{len(jobs)} ROI embeddings")
    matrix.flush()
    del matrix
    os.replace(temporary, matrix_path)

    mapping = jobs.copy()
    mapping.insert(0, "embedding_row", np.arange(len(mapping), dtype=np.int64))
    mapping.insert(1, "encoder_id", args.encoder_id)
    mapping.insert(2, "embedding_model", args.embedding_model)
    mapping.insert(3, "embedding_version", args.embedding_version)
    mapping.insert(4, "entity_type", "roi")
    mapping.insert(5, "parent_frame_uid", mapping["frame_uid"])
    mapping["crop_x_min"] = [box[0] for box in crop_boxes if box is not None]
    mapping["crop_y_min"] = [box[1] for box in crop_boxes if box is not None]
    mapping["crop_x_max"] = [box[2] for box in crop_boxes if box is not None]
    mapping["crop_y_max"] = [box[3] for box in crop_boxes if box is not None]
    temporary_mapping = mapping_path.with_suffix(".parquet.tmp")
    mapping.to_parquet(temporary_mapping, index=False)
    os.replace(temporary_mapping, mapping_path)

    persisted = np.load(matrix_path, mmap_mode="r")
    check_idx = np.linspace(
        0, len(persisted) - 1, min(2000, len(persisted)), dtype=np.int64
    )
    norms = np.linalg.norm(np.asarray(persisted[check_idx], dtype=np.float32), axis=1)
    elapsed = time.time() - started
    report = {
        "video_id": args.video_id,
        "entity_type": "roi",
        "encoder_id": args.encoder_id,
        "embedding_model": args.embedding_model,
        "embedding_version": args.embedding_version,
        "object_model_id": args.model_id,
        "source_detections": str(detections_path.resolve()),
        "source_manifest": str(image_manifest_path.resolve()),
        "rows": len(mapping),
        "dimension": feature_dim,
        "batch_size": args.batch_size,
        "padding_ratio": args.padding_ratio,
        "min_score": args.min_score,
        "min_area_ratio": args.min_area_ratio,
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2),
        "elapsed_seconds": round(elapsed, 3),
        "roi_fps": round(len(mapping) / elapsed, 3),
        "outputs": {"embeddings": str(matrix_path), "mapping": str(mapping_path)},
    }
    report_dir = PROJECT_ROOT / "workspace" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"openclip_rois_{args.video_id}{suffix}.json"
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_report, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
