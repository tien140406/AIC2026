from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import open_clip
import pandas as pd
from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "ViT-B-16"
DEFAULT_PRETRAINED = "laion2b_s34b_b88k"
DEFAULT_ENCODER_ID = "openclip_vit_b16_laion2b_s34b_b88k"
DEFAULT_KEYFRAME_ENCODER_ID = "timm_beit3_base_patch16_224_pt"


def find_video(video_id: str) -> Path:
    videos = pd.read_parquet(PROJECT_ROOT / "workspace" / "manifests" / "videos.parquet")
    match = videos.loc[videos["video_id"] == video_id]
    if len(match) != 1:
        raise ValueError(f"Expected one video row for {video_id}, got {len(match)}")
    return Path(match.iloc[0]["video_path"])


def encode(
    images: list[torch.Tensor],
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    batch = torch.stack(images).to(device, non_blocking=True)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        features = model.encode_image(batch, normalize=True)
    return features.float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract OpenCLIP image embeddings from selected keyframes."
    )
    parser.add_argument("--video-id", required=True)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--keyframes",
        type=Path,
        default=None,
        help="Selected keyframe manifest. Defaults to the configured keyframe encoder output.",
    )
    source_group.add_argument(
        "--sampling",
        type=Path,
        default=None,
        help="Use a sampling manifest instead when building a dense frame index.",
    )
    parser.add_argument(
        "--keyframe-encoder-id",
        default=DEFAULT_KEYFRAME_ENCODER_ID,
        help="Encoder ID of the selected keyframe manifest used by default.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--encoder-id", default=DEFAULT_ENCODER_ID)
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=PROJECT_ROOT / "models" / "openclip",
        help="Directory for downloaded OpenCLIP weights.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if not args.encoder_id.strip():
        raise ValueError("encoder ID must not be empty")

    started = time.time()
    video_path = find_video(args.video_id)
    if args.sampling is not None:
        source_kind = "sampling"
        source_path = args.sampling
    else:
        source_kind = "keyframes"
        source_path = args.keyframes or (
            PROJECT_ROOT
            / "workspace"
            / "keyframes"
            / args.keyframe_encoder_id
            / f"{args.video_id}.parquet"
        )
    frames = pd.read_parquet(source_path).sort_values("frame_id").reset_index(drop=True)
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError("max frames must be positive")
        frames = frames.iloc[: args.max_frames].copy()
    if "frame_id" not in frames.columns:
        raise ValueError(f"Missing frame_id column in {source_path}")
    if frames.empty or frames["frame_id"].duplicated().any():
        raise ValueError("Invalid source manifest")

    output_dir = PROJECT_ROOT / "workspace" / "features" / args.encoder_id
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".smoke{len(frames)}" if args.max_frames is not None else ""
    matrix_path = output_dir / f"{args.video_id}{suffix}.f16.npy"
    mapping_path = output_dir / f"{args.video_id}{suffix}.parquet"
    if matrix_path.exists() and not args.force:
        raise FileExistsError(f"Output exists: {matrix_path}. Use --force to rebuild.")

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
        temporary, mode="w+", dtype=np.float16, shape=(len(frames), feature_dim)
    )
    target_to_row = {int(frame): row for row, frame in enumerate(frames["frame_id"])}
    last_target = max(target_to_row)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    images: list[torch.Tensor] = []
    rows: list[int] = []
    extracted = 0
    decoded = 0

    def flush() -> None:
        nonlocal extracted
        if not images:
            return
        matrix[np.asarray(rows, dtype=np.int64)] = encode(images, model, device).astype(np.float16)
        extracted += len(images)
        images.clear()
        rows.clear()

    try:
        frame_id = 0
        while frame_id <= last_target:
            ok, bgr = capture.read()
            if not ok:
                break
            decoded += 1
            row = target_to_row.get(frame_id)
            if row is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                images.append(preprocess(Image.fromarray(rgb)))
                rows.append(row)
                if len(images) >= args.batch_size:
                    flush()
                    if extracted % 1024 < args.batch_size:
                        print(f"Embedded {extracted:,}/{len(frames):,}", flush=True)
            frame_id += 1
        flush()
    finally:
        capture.release()

    if extracted != len(frames):
        raise RuntimeError(f"Extracted {extracted}/{len(frames)} frames")
    matrix.flush()
    del matrix
    os.replace(temporary, matrix_path)

    mapping = frames.copy()
    # Preserve provenance from the keyframe selector before assigning this encoder's rows.
    source_columns = {
        column: f"source_{column}"
        for column in ("embedding_row", "encoder_id")
        if column in mapping.columns
    }
    mapping = mapping.rename(columns=source_columns)
    mapping.insert(0, "embedding_row", np.arange(len(mapping), dtype=np.int64))
    mapping.insert(1, "encoder_id", args.encoder_id)
    temporary_mapping = mapping_path.with_suffix(".parquet.tmp")
    mapping.to_parquet(temporary_mapping, index=False)
    os.replace(temporary_mapping, mapping_path)

    persisted = np.load(matrix_path, mmap_mode="r")
    check_idx = np.linspace(0, len(persisted) - 1, min(2000, len(persisted)), dtype=np.int64)
    norms = np.linalg.norm(np.asarray(persisted[check_idx], dtype=np.float32), axis=1)
    elapsed = time.time() - started
    report = {
        "video_id": args.video_id,
        "encoder_id": args.encoder_id,
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "model_cache": str(args.model_cache.resolve()),
        "model_role": "OpenCLIP image encoder for image-text retrieval",
        "source_kind": source_kind,
        "source_manifest": str(source_path.resolve()),
        "rows": len(frames),
        "dimension": feature_dim,
        "batch_size": args.batch_size,
        "decoded_frames": decoded,
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2),
        "elapsed_seconds": round(elapsed, 3),
        "sample_fps": round(len(frames) / elapsed, 3),
        "outputs": {"embeddings": str(matrix_path), "mapping": str(mapping_path)},
    }
    report_path = PROJECT_ROOT / "workspace" / "reports" / f"openclip_{args.video_id}{suffix}.json"
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_report, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
