from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from ..common.identity import ensure_frame_identity, frame_uid, seconds_to_ms
from ..common.io import atomic_write_json, atomic_write_parquet
from .codetr import (
    DEFAULT_CHECKPOINT_NAME,
    DEFAULT_CONFIG_RELATIVE,
    DEFAULT_MODEL_ID,
    CoDetrDetector,
)
from .filtering import (
    DEFAULT_DUPLICATE_IOU,
    DEFAULT_MAX_DETECTIONS,
    DEFAULT_MIN_AREA_RATIO,
    DEFAULT_SCORE_THRESHOLD,
    write_summary_tables,
)
from .schema import SCHEMA_VERSION, normalize_detection, valid_raw_payload

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CODE_ROOT = PROJECT_ROOT / "third_party" / "Co-DETR"
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "models" / "codetr"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_slug(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id).strip("_")


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def codetr_code_revision(codetr_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(codetr_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_jobs(
    manifest_path: Path,
    video_id: str,
    *,
    offset: int = 0,
    limit: int | None = None,
    hash_workers: int = 4,
) -> pd.DataFrame:
    manifest = pd.read_parquet(manifest_path)
    required = {"video_id", "frame_id", "image_path"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    jobs = manifest.loc[manifest["video_id"] == video_id].copy()
    if "image_exists" in jobs.columns:
        jobs = jobs.loc[jobs["image_exists"].astype(bool)]
    jobs = jobs.sort_values(
        [column for column in ("pts_time", "frame_id") if column in jobs.columns]
    ).reset_index(drop=True)
    if jobs.empty:
        raise ValueError(f"No final keyframes found for {video_id}: {manifest_path}")
    if jobs["frame_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate frame IDs")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    jobs = jobs.iloc[offset:]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        jobs = jobs.iloc[:limit]
    if jobs.empty:
        raise ValueError("No jobs remain after offset/limit")

    image_paths = [Path(path) for path in jobs["image_path"]]
    missing_images = [path for path in image_paths if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"Missing {len(missing_images)} final-keyframe images; first: "
            f"{missing_images[0]}"
        )
    if hash_workers <= 0:
        raise ValueError("hash_workers must be positive")
    if hash_workers == 1:
        hashes = [file_sha256(path) for path in image_paths]
    else:
        with ThreadPoolExecutor(max_workers=hash_workers) as executor:
            hashes = list(executor.map(file_sha256, image_paths))
    jobs["image_sha256"] = hashes
    return ensure_frame_identity(jobs)


def _read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.width, image.height


def run_signature(
    *,
    model_id: str,
    config_sha256: str,
    checkpoint_sha256: str,
    code_revision: str,
    raw_score_floor: float,
    fp16: bool,
) -> str:
    canonical = json.dumps(
        {
            "model_id": model_id,
            "config_sha256": config_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "code_revision": code_revision,
            "raw_score_floor": raw_score_floor,
            "fp16": fp16,
            "schema_version": SCHEMA_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def success_path(output_dir: Path, frame_id: int) -> Path:
    return output_dir / "raw" / f"frame_{frame_id:09d}.json"


def failure_path(output_dir: Path, frame_id: int) -> Path:
    return output_dir / "failures" / f"frame_{frame_id:09d}.json"


def reusable_success(
    path: Path,
    *,
    frame_id: int,
    image_sha256: str,
    run_signature_value: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("frame_id") == frame_id
        and payload.get("image_sha256") == image_sha256
        and payload.get("run_signature") == run_signature_value
        and valid_raw_payload(payload)
    )


def failure_payload(
    row: dict[str, Any],
    *,
    model_id: str,
    run_signature_value: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "video_id": row["video_id"],
        "frame_uid": row.get("frame_uid")
        or frame_uid(row["video_id"], int(row["frame_id"])),
        "shot_id": int(row["shot_id"]) if pd.notna(row.get("shot_id")) else None,
        "frame_id": int(row["frame_id"]),
        "pts_time": float(row["pts_time"]) if pd.notna(row.get("pts_time")) else None,
        "pts_ms": (
            int(row["pts_ms"])
            if pd.notna(row.get("pts_ms"))
            else seconds_to_ms(row.get("pts_time"))
        ),
        "image_path": str(row["image_path"]),
        "image_sha256": row["image_sha256"],
        "model_id": model_id,
        "run_signature": run_signature_value,
        "error_type": type(error).__name__,
        "error": str(error),
        "failed_at": utc_now(),
    }


def write_failure_table(
    output_dir: Path,
    *,
    selected_ids: set[int],
    run_signature_value: str,
    successful_ids: set[int] | None = None,
) -> pd.DataFrame:
    successful_ids = successful_ids or set()
    records: list[dict[str, Any]] = []
    for path in sorted((output_dir / "failures").glob("frame_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            int(payload.get("frame_id", -1)) in selected_ids
            and int(payload.get("frame_id", -1)) not in successful_ids
            and payload.get("run_signature") == run_signature_value
        ):
            records.append(payload)
    columns = [
        "schema_version",
        "video_id",
        "frame_uid",
        "shot_id",
        "frame_id",
        "pts_time",
        "pts_ms",
        "image_path",
        "image_sha256",
        "model_id",
        "run_signature",
        "error_type",
        "error",
        "failed_at",
    ]
    failures = pd.DataFrame(records, columns=columns)
    if not failures.empty:
        failures = failures.sort_values(["video_id", "frame_id"]).reset_index(drop=True)
    atomic_write_parquet(failures, output_dir / "failures.parquet")
    return failures


def checkpoint_report(
    *,
    output_dir: Path,
    selection: dict[str, Any],
    selected_ids: set[int],
    run_signature_value: str,
    counters: dict[str, int],
    started: float,
    status: str,
    peak_vram_mb: float | None,
) -> dict[str, Any]:
    frames, detections, filter_report = write_summary_tables(
        output_dir,
        score_threshold=selection["score_threshold"],
        min_area_ratio=selection["min_area_ratio"],
        duplicate_iou=selection["duplicate_iou"],
        max_detections=selection["max_detections"],
        selected_ids=selected_ids,
        run_signature=run_signature_value,
    )
    successful_ids = set(frames["frame_id"].astype(int)) if not frames.empty else set()
    failures = write_failure_table(
        output_dir,
        selected_ids=selected_ids,
        run_signature_value=run_signature_value,
        successful_ids=successful_ids,
    )
    failed_ids = set(failures["frame_id"].astype(int)) if not failures.empty else set()
    failed_ids -= successful_ids
    report = {
        **selection,
        "status": status,
        "new_successes": counters["success"],
        "new_failures": counters["failure"],
        "processed_in_run": counters["completed"],
        "selected_successes": len(successful_ids),
        "selected_failures": len(failed_ids),
        "selected_remaining": len(selected_ids - successful_ids - failed_ids),
        "filtered_detections": len(detections),
        "filter_report": filter_report,
        "peak_vram_mb": peak_vram_mb,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "checkpointed_at": utc_now(),
        "outputs": {
            "raw_directory": str(output_dir / "raw"),
            "failure_directory": str(output_dir / "failures"),
            "frames": str(output_dir / "frames.parquet"),
            "detections": str(output_dir / "detections.parquet"),
            "failures": str(output_dir / "failures.parquet"),
            "report": str(output_dir / "report.json"),
        },
    }
    atomic_write_json(report, output_dir / "report.json")
    atomic_write_json(
        report,
        PROJECT_ROOT
        / "workspace"
        / "reports"
        / f"codetr_objects_{selection['video_id']}.json",
    )
    return report


def run_pipeline(
    *,
    video_id: str,
    manifest_path: Path,
    model_id: str,
    config_path: Path,
    checkpoint_path: Path,
    codetr_root: Path,
    output_dir: Path,
    device: str,
    fp16: bool,
    raw_score_floor: float,
    score_threshold: float,
    min_area_ratio: float,
    duplicate_iou: float,
    max_detections: int,
    offset: int = 0,
    limit: int | None = None,
    hash_workers: int = 4,
    checkpoint_every: int = 100,
    progress_every: int = 25,
    force: bool = False,
    detector_factory: Callable[..., Any] = CoDetrDetector,
) -> dict[str, Any]:
    if not 0 <= raw_score_floor <= 1:
        raise ValueError("raw_score_floor must be between 0 and 1")
    if checkpoint_every <= 0 or progress_every <= 0:
        raise ValueError("checkpoint_every and progress_every must be positive")
    jobs = load_jobs(
        manifest_path,
        video_id,
        offset=offset,
        limit=limit,
        hash_workers=hash_workers,
    )
    config_sha = file_sha256(config_path)
    checkpoint_sha = file_sha256(checkpoint_path)
    code_revision = codetr_code_revision(codetr_root)
    signature = run_signature(
        model_id=model_id,
        config_sha256=config_sha,
        checkpoint_sha256=checkpoint_sha,
        code_revision=code_revision,
        raw_score_floor=raw_score_floor,
        fp16=fp16,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pending: list[dict[str, Any]] = []
    reusable = 0
    for row in jobs.to_dict(orient="records"):
        frame_id = int(row["frame_id"])
        if not force and reusable_success(
            success_path(output_dir, frame_id),
            frame_id=frame_id,
            image_sha256=row["image_sha256"],
            run_signature_value=signature,
        ):
            reusable += 1
        else:
            pending.append(row)

    selection = {
        "video_id": video_id,
        "manifest": str(manifest_path),
        "model_id": model_id,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "config_sha256": config_sha,
        "checkpoint_sha256": checkpoint_sha,
        "code_revision": code_revision,
        "run_signature": signature,
        "schema_version": SCHEMA_VERSION,
        "selected_jobs": len(jobs),
        "pending_jobs": len(pending),
        "reusable_successes": reusable,
        "device": device,
        "fp16": fp16,
        "raw_score_floor": raw_score_floor,
        "score_threshold": score_threshold,
        "min_area_ratio": min_area_ratio,
        "duplicate_iou": duplicate_iou,
        "max_detections": max_detections,
        "checkpoint_every": checkpoint_every,
        "output_dir": str(output_dir),
    }
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)
    started = time.monotonic()
    selected_ids = set(jobs["frame_id"].astype(int))
    counters = {"completed": 0, "success": 0, "failure": 0}
    detector: Any | None = None
    peak_vram_mb: float | None = None
    next_checkpoint = checkpoint_every
    next_progress = progress_every
    interrupted = False

    try:
        if pending:
            detector = detector_factory(
                codetr_root=codetr_root,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                device=device,
                fp16=fp16,
            )
        for row in pending:
            frame_id = int(row["frame_id"])
            try:
                width, height = _read_image_size(Path(row["image_path"]))
                raw = detector.predict(Path(row["image_path"]))
                detections = []
                for item in raw:
                    if float(item["score"]) < raw_score_floor:
                        continue
                    detection = normalize_detection(
                        class_id=int(item["class_id"]),
                        class_name=str(item["class_name"]),
                        score=float(item["score"]),
                        bbox_xyxy=item["bbox_xyxy"],
                        image_width=width,
                        image_height=height,
                    )
                    if detection is not None:
                        detections.append(detection)
                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "video_id": row["video_id"],
                    "frame_uid": row.get("frame_uid")
                    or frame_uid(row["video_id"], frame_id),
                    "shot_id": int(row["shot_id"])
                    if pd.notna(row.get("shot_id"))
                    else None,
                    "frame_id": frame_id,
                    "pts_time": float(row["pts_time"])
                    if pd.notna(row.get("pts_time"))
                    else None,
                    "pts_ms": (
                        int(row["pts_ms"])
                        if pd.notna(row.get("pts_ms"))
                        else seconds_to_ms(row.get("pts_time"))
                    ),
                    "image_path": str(row["image_path"]),
                    "image_sha256": row["image_sha256"],
                    "image_width": width,
                    "image_height": height,
                    "model_id": model_id,
                    "config_sha256": config_sha,
                    "checkpoint_sha256": checkpoint_sha,
                    "code_revision": code_revision,
                    "run_signature": signature,
                    "detections": detections,
                    "processed_at": utc_now(),
                }
                atomic_write_json(payload, success_path(output_dir, frame_id))
                counters["success"] += 1
            except Exception as error:  # noqa: BLE001 - isolate one frame.
                failure = failure_payload(
                    row,
                    model_id=model_id,
                    run_signature_value=signature,
                    error=error,
                )
                atomic_write_json(failure, failure_path(output_dir, frame_id))
                counters["failure"] += 1
            counters["completed"] += 1

            if detector is not None and hasattr(detector, "peak_vram_mb"):
                peak_vram_mb = detector.peak_vram_mb()
            if counters["completed"] >= next_checkpoint:
                checkpoint_report(
                    output_dir=output_dir,
                    selection=selection,
                    selected_ids=selected_ids,
                    run_signature_value=signature,
                    counters=counters,
                    started=started,
                    status="running",
                    peak_vram_mb=peak_vram_mb,
                )
                while next_checkpoint <= counters["completed"]:
                    next_checkpoint += checkpoint_every
            if counters["completed"] >= next_progress:
                print(
                    f"Processed {counters['completed']}/{len(pending)}: "
                    f"success={counters['success']}, failure={counters['failure']}",
                    flush=True,
                )
                while next_progress <= counters["completed"]:
                    next_progress += progress_every
    except KeyboardInterrupt:
        interrupted = True
        print(
            "Interrupted; checkpointing completed final-keyframe detections", flush=True
        )

    status = "interrupted" if interrupted else "completed"
    report = checkpoint_report(
        output_dir=output_dir,
        selection=selection,
        selected_ids=selected_ids,
        run_signature_value=signature,
        counters=counters,
        started=started,
        status=status,
        peak_vram_mb=peak_vram_mb,
    )
    if status == "completed" and report["selected_failures"]:
        status = "completed_with_failures"
        report["status"] = status
        atomic_write_json(report, output_dir / "report.json")
        atomic_write_json(
            report,
            PROJECT_ROOT / "workspace" / "reports" / f"codetr_objects_{video_id}.json",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if status == "interrupted":
        raise SystemExit(130)
    if status == "completed_with_failures":
        raise SystemExit(2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Co-DETR objects from final keyframes."
    )
    parser.add_argument("--video-id", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Final-keyframe JPG manifest; defaults to workspace/keyframe_images.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--codetr-root", type=Path, default=DEFAULT_CODE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--raw-score-floor", type=float, default=0.05)
    parser.add_argument(
        "--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD
    )
    parser.add_argument("--min-area-ratio", type=float, default=DEFAULT_MIN_AREA_RATIO)
    parser.add_argument("--duplicate-iou", type=float, default=DEFAULT_DUPLICATE_IOU)
    parser.add_argument("--max-detections", type=int, default=DEFAULT_MAX_DETECTIONS)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hash-workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config or args.codetr_root / DEFAULT_CONFIG_RELATIVE
    checkpoint_path = (
        args.checkpoint or DEFAULT_CHECKPOINT_ROOT / DEFAULT_CHECKPOINT_NAME
    )
    manifest_path = args.manifest or (
        PROJECT_ROOT
        / "workspace"
        / "keyframe_images"
        / args.video_id
        / "manifest.parquet"
    )
    output_dir = args.output_dir or (
        PROJECT_ROOT
        / "workspace"
        / "objects"
        / model_slug(args.model_id)
        / args.video_id
    )
    if args.dry_run:
        jobs = load_jobs(
            manifest_path,
            args.video_id,
            offset=args.offset,
            limit=args.limit,
            hash_workers=args.hash_workers,
        )
        print(
            json.dumps(
                {
                    "video_id": args.video_id,
                    "manifest": str(manifest_path),
                    "model_id": args.model_id,
                    "config": str(config_path),
                    "checkpoint": str(checkpoint_path),
                    "final_keyframes": len(jobs),
                    "output_dir": str(output_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Co-DETR config/checkpoint is missing. Run scripts/setup_codetr.py."
        )
    run_pipeline(
        video_id=args.video_id,
        manifest_path=manifest_path,
        model_id=args.model_id,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        codetr_root=args.codetr_root,
        output_dir=output_dir,
        device=args.device,
        fp16=False,
        raw_score_floor=args.raw_score_floor,
        score_threshold=args.score_threshold,
        min_area_ratio=args.min_area_ratio,
        duplicate_iou=args.duplicate_iou,
        max_detections=args.max_detections,
        offset=args.offset,
        limit=args.limit,
        hash_workers=args.hash_workers,
        checkpoint_every=args.checkpoint_every,
        progress_every=args.progress_every,
        force=args.force,
    )


if __name__ == "__main__":
    main()
