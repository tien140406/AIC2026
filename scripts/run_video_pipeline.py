from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION_ENCODER_ID = "timm_beit3_base_patch16_224_pt"
DEFAULT_OPENCLIP_ENCODER_ID = "openclip_vit_b16_laion2b_s34b_b88k"
DEFAULT_OBJECT_MODEL_ID = "co_dino_5scale_lsj_r50_3x_coco"

STAGE_NAMES = (
    "audit",
    "shots",
    "sampling",
    "beit3",
    "keyframes",
    "openclip_keyframes",
    "keyframe_jpgs",
    "asr",
    "ocr",
    "codetr_setup",
    "objects",
    "object_filter",
    "openclip_rois",
    "serving_up",
    "serving_check",
    "elasticsearch",
    "milvus_keyframes",
    "milvus_rois",
)


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    enabled: bool = True
    skip_reason: str | None = None
    cached_outputs: tuple[Path, ...] = ()
    rerun_on_force: bool = True


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def uv_command(project: str, script: str, *arguments: object) -> tuple[str, ...]:
    return (
        "uv",
        "run",
        "--project",
        f"environments/{project}",
        "python",
        f"scripts/{script}",
        *(str(argument) for argument in arguments),
    )


def append_flag(command: tuple[str, ...], enabled: bool, flag: str) -> tuple[str, ...]:
    return (*command, flag) if enabled else command


def build_stages(args: argparse.Namespace, root: Path) -> list[Stage]:
    video_id = args.video_id
    video_path = args.video_path or root / "archive" / "video" / f"{video_id}.mp4"
    if not video_path.is_absolute():
        video_path = root / video_path

    selection_dir = root / "workspace" / "keyframes" / args.selection_encoder_id
    selection_feature_dir = (
        root / "workspace" / "features" / args.selection_encoder_id
    )
    openclip_dir = root / "workspace" / "features" / args.openclip_encoder_id
    roi_encoder_id = f"{args.openclip_encoder_id}_roi"
    roi_dir = root / "workspace" / "features" / roi_encoder_id
    image_dir = root / "workspace" / "keyframe_images" / video_id
    object_dir = root / "workspace" / "objects" / args.object_model_id / video_id
    compose_file = "environments/serving/docker-compose.local.yml"

    shots = uv_command(
        "omnishotcut", "run_omnishotcut.py", "--input", video_path
    )
    sampling = uv_command(
        "omnishotcut",
        "build_sampling_manifest.py",
        "--video-id",
        video_id,
        "--stride",
        args.stride,
    )
    beit3 = uv_command(
        "beit3-timm",
        "extract_timm_beit3_embeddings.py",
        "--video-id",
        video_id,
        "--batch-size",
        args.beit_batch_size,
    )
    keyframes = uv_command(
        "beit3-timm",
        "select_keyframes.py",
        "--video-id",
        video_id,
        "--encoder-id",
        args.selection_encoder_id,
    )
    openclip_keyframes = uv_command(
        "beit3-timm",
        "openclip_embeddings.py",
        "--video-id",
        video_id,
        "--keyframe-encoder-id",
        args.selection_encoder_id,
        "--encoder-id",
        args.openclip_encoder_id,
        "--batch-size",
        args.openclip_batch_size,
    )
    keyframe_jpgs = uv_command(
        "beit3-timm",
        "export_keyframe_jpgs.py",
        "--video-id",
        video_id,
        "--encoder-id",
        args.selection_encoder_id,
        "--quality",
        args.jpg_quality,
    )
    asr = uv_command(
        "asr",
        "extract_transcripts.py",
        "--video-id",
        video_id,
        "--provider",
        args.asr_provider,
        "--model",
        args.asr_model,
        "--keyframe-encoder-id",
        args.selection_encoder_id,
    )
    asr = append_flag(asr, not args.allow_youtube, "--skip-youtube")
    ocr = uv_command(
        "gemini",
        "extract_gemini_ocr.py",
        "--video-id",
        video_id,
        "--batch-size",
        args.ocr_batch_size,
        "--concurrency",
        args.ocr_concurrency,
        "--checkpoint-every",
        args.ocr_checkpoint_every,
    )
    objects = uv_command(
        "codetr",
        "extract_codetr_objects.py",
        "--video-id",
        video_id,
        "--model-id",
        args.object_model_id,
        "--score-threshold",
        args.object_score_threshold,
        "--checkpoint-every",
        args.object_checkpoint_every,
    )
    object_filter = uv_command(
        "codetr",
        "filter_object_detections.py",
        "--output-dir",
        object_dir,
        "--score-threshold",
        args.filtered_object_score_threshold,
    )
    openclip_rois = uv_command(
        "beit3-timm",
        "openclip_roi_embeddings.py",
        "--video-id",
        video_id,
        "--model-id",
        args.object_model_id,
        "--encoder-id",
        roi_encoder_id,
        "--batch-size",
        args.openclip_batch_size,
        "--padding-ratio",
        args.roi_padding_ratio,
    )

    if args.force:
        shots = (*shots, "--force")
        sampling = (*sampling, "--force")
        beit3 = (*beit3, "--force")
        keyframes = (*keyframes, "--force")
        openclip_keyframes = (*openclip_keyframes, "--force")
        keyframe_jpgs = (*keyframe_jpgs, "--force")
        asr = (*asr, "--force")
        ocr = (*ocr, "--force", "--screening-force")
        objects = (*objects, "--force")
        openclip_rois = (*openclip_rois, "--force")

    elasticsearch = uv_command(
        "serving",
        "index_elasticsearch.py",
        "--selection-encoder-id",
        args.selection_encoder_id,
    )
    milvus_keyframes = uv_command(
        "serving",
        "ingest_milvus_artifacts.py",
        "--entity-type",
        "keyframe",
        "--encoder-id",
        args.openclip_encoder_id,
        "--selection-encoder-id",
        args.selection_encoder_id,
    )
    milvus_rois = uv_command(
        "serving",
        "ingest_milvus_artifacts.py",
        "--entity-type",
        "roi",
        "--encoder-id",
        roi_encoder_id,
        "--selection-encoder-id",
        args.selection_encoder_id,
    )
    if args.recreate_serving:
        elasticsearch = (*elasticsearch, "--recreate")
        milvus_keyframes = (*milvus_keyframes, "--drop-existing")
        milvus_rois = (*milvus_rois, "--drop-existing")

    serving_check = uv_command("serving", "check_local_serving.py")
    serving_check = append_flag(
        serving_check, args.serving_smoke_write, "--smoke-write"
    )
    optional = {
        "asr": (not args.skip_asr, "disabled by --skip-asr"),
        "ocr": (not args.skip_ocr, "disabled by --skip-ocr"),
        "objects": (not args.skip_objects, "disabled by --skip-objects"),
        "serving": (not args.skip_serving, "disabled by --skip-serving"),
    }

    def optional_stage(name: str, command: tuple[str, ...], group: str) -> Stage:
        enabled, reason = optional[group]
        return Stage(name, command, enabled=enabled, skip_reason=None if enabled else reason)

    return [
        Stage("audit", uv_command("omnishotcut", "audit_dataset.py")),
        Stage(
            "shots",
            shots,
            cached_outputs=(root / "workspace" / "shots" / f"{video_id}.json",),
        ),
        Stage(
            "sampling",
            sampling,
            cached_outputs=(root / "workspace" / "samples" / f"{video_id}.parquet",),
        ),
        Stage(
            "beit3",
            beit3,
            cached_outputs=(
                selection_feature_dir / f"{video_id}.f16.npy",
                selection_feature_dir / f"{video_id}.parquet",
            ),
        ),
        Stage(
            "keyframes",
            keyframes,
            cached_outputs=(
                selection_dir / f"{video_id}.parquet",
                selection_dir / f"{video_id}.audit.parquet",
                selection_dir / f"{video_id}.f16.npy",
            ),
        ),
        Stage(
            "openclip_keyframes",
            openclip_keyframes,
            cached_outputs=(
                openclip_dir / f"{video_id}.f16.npy",
                openclip_dir / f"{video_id}.parquet",
            ),
        ),
        Stage(
            "keyframe_jpgs",
            keyframe_jpgs,
            cached_outputs=(image_dir / "manifest.parquet",),
        ),
        optional_stage("asr", asr, "asr"),
        optional_stage("ocr", ocr, "ocr"),
        Stage(
            "codetr_setup",
            uv_command("codetr", "setup_codetr.py"),
            enabled=optional["objects"][0],
            skip_reason=None if optional["objects"][0] else optional["objects"][1],
            cached_outputs=(
                root
                / "third_party"
                / "Co-DETR"
                / "projects"
                / "configs"
                / "co_dino"
                / "co_dino_5scale_lsj_r50_3x_coco.py",
                root
                / "models"
                / "codetr"
                / "co_dino_5scale_lsj_r50_3x_coco.pth",
            ),
            rerun_on_force=False,
        ),
        optional_stage("objects", objects, "objects"),
        optional_stage("object_filter", object_filter, "objects"),
        Stage(
            "openclip_rois",
            openclip_rois,
            enabled=optional["objects"][0],
            skip_reason=None if optional["objects"][0] else optional["objects"][1],
            cached_outputs=(
                roi_dir / f"{video_id}.f16.npy",
                roi_dir / f"{video_id}.parquet",
            ),
        ),
        Stage(
            "serving_up",
            ("docker", "compose", "-f", compose_file, "up", "-d", "--wait"),
            enabled=optional["serving"][0],
            skip_reason=None if optional["serving"][0] else optional["serving"][1],
        ),
        optional_stage("serving_check", serving_check, "serving"),
        optional_stage("elasticsearch", elasticsearch, "serving"),
        optional_stage("milvus_keyframes", milvus_keyframes, "serving"),
        Stage(
            "milvus_rois",
            milvus_rois,
            enabled=optional["serving"][0] and optional["objects"][0],
            skip_reason=(
                None
                if optional["serving"][0] and optional["objects"][0]
                else "requires serving and object stages"
            ),
        ),
    ]


def stage_is_cached(stage: Stage, *, force: bool) -> bool:
    if not stage.cached_outputs:
        return False
    if force and stage.rerun_on_force:
        return False
    return all(path.exists() for path in stage.cached_outputs)


def stage_in_range(name: str, from_stage: str, to_stage: str) -> bool:
    index = STAGE_NAMES.index(name)
    return STAGE_NAMES.index(from_stage) <= index <= STAGE_NAMES.index(to_stage)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def run_pipeline(args: argparse.Namespace, root: Path, stages: Sequence[Stage]) -> int:
    report_path = (
        root / "workspace" / "reports" / f"offline_pipeline_{args.video_id}.json"
    )
    report: dict[str, Any] = {
        "video_id": args.video_id,
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "root": str(root),
        "options": vars(args),
        "stages": [],
    }
    write_report(report, report_path)

    selected = [
        stage
        for stage in stages
        if stage_in_range(stage.name, args.from_stage, args.to_stage)
    ]
    for position, stage in enumerate(selected, start=1):
        command_text = shlex.join(stage.command)
        record: dict[str, Any] = {
            "name": stage.name,
            "command": list(stage.command),
            "started_at": utc_now(),
            "finished_at": None,
            "elapsed_seconds": 0.0,
            "return_code": None,
        }
        print(f"\n=== [{position}/{len(selected)}] {stage.name} ===", flush=True)
        print(command_text, flush=True)

        if not stage.enabled:
            record["status"] = "skipped_disabled"
            record["reason"] = stage.skip_reason
            print(f"Skipped: {stage.skip_reason}", flush=True)
        elif stage_is_cached(stage, force=args.force):
            record["status"] = "skipped_cached"
            record["outputs"] = [str(path) for path in stage.cached_outputs]
            print("Skipped: expected outputs already exist", flush=True)
        elif args.dry_run:
            record["status"] = "dry_run"
        else:
            started = time.monotonic()
            try:
                completed = subprocess.run(stage.command, cwd=root, check=False)
                record["return_code"] = completed.returncode
                record["status"] = (
                    "completed" if completed.returncode == 0 else "failed"
                )
            except KeyboardInterrupt:
                record["return_code"] = 130
                record["status"] = "interrupted"
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)

        record["finished_at"] = utc_now()
        report["stages"].append(record)
        write_report(report, report_path)
        if record["status"] in {"failed", "interrupted"}:
            report["status"] = record["status"]
            report["failed_stage"] = stage.name
            report["finished_at"] = utc_now()
            write_report(report, report_path)
            print(f"Pipeline stopped at {stage.name}. Report: {report_path}")
            return int(record["return_code"] or 1)

    report["status"] = "dry_run" if args.dry_run else "completed"
    report["finished_at"] = utc_now()
    write_report(report, report_path)
    print(f"\nPipeline {report['status']}. Report: {report_path}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run offline extraction, embedding, and serving ingestion for one video."
        )
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--video-id")
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument("--list-stages", action="store_true")
    parser.add_argument("--from-stage", choices=STAGE_NAMES, default=STAGE_NAMES[0])
    parser.add_argument("--to-stage", choices=STAGE_NAMES, default=STAGE_NAMES[-1])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--skip-objects", action="store_true")
    parser.add_argument("--skip-serving", action="store_true")
    parser.add_argument("--recreate-serving", action="store_true")
    parser.add_argument(
        "--serving-smoke-write",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--jpg-quality", type=int, default=92)
    parser.add_argument("--beit-batch-size", type=int, default=32)
    parser.add_argument("--openclip-batch-size", type=int, default=32)
    parser.add_argument("--selection-encoder-id", default=DEFAULT_SELECTION_ENCODER_ID)
    parser.add_argument("--openclip-encoder-id", default=DEFAULT_OPENCLIP_ENCODER_ID)
    parser.add_argument(
        "--asr-provider",
        choices=("auto", "local", "youtube", "whisper"),
        default="auto",
    )
    parser.add_argument("--asr-model", default="large-v3")
    parser.add_argument("--allow-youtube", action="store_true")
    parser.add_argument("--ocr-batch-size", type=int, default=6)
    parser.add_argument("--ocr-concurrency", type=int, default=8)
    parser.add_argument("--ocr-checkpoint-every", type=int, default=100)
    parser.add_argument("--object-model-id", default=DEFAULT_OBJECT_MODEL_ID)
    parser.add_argument("--object-score-threshold", type=float, default=0.30)
    parser.add_argument("--filtered-object-score-threshold", type=float, default=0.40)
    parser.add_argument("--object-checkpoint-every", type=int, default=100)
    parser.add_argument("--roi-padding-ratio", type=float, default=0.05)
    args = parser.parse_args(argv)

    if args.list_stages:
        return args
    if not args.video_id:
        parser.error("--video-id is required")
    if STAGE_NAMES.index(args.from_stage) > STAGE_NAMES.index(args.to_stage):
        parser.error("--from-stage must not come after --to-stage")
    if args.skip_serving and args.recreate_serving:
        parser.error("--recreate-serving cannot be used with --skip-serving")
    positive = {
        "--stride": args.stride,
        "--jpg-quality": args.jpg_quality,
        "--beit-batch-size": args.beit_batch_size,
        "--openclip-batch-size": args.openclip_batch_size,
        "--ocr-batch-size": args.ocr_batch_size,
        "--ocr-concurrency": args.ocr_concurrency,
        "--ocr-checkpoint-every": args.ocr_checkpoint_every,
        "--object-checkpoint-every": args.object_checkpoint_every,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        parser.error(f"arguments must be positive: {', '.join(invalid)}")
    if not 1 <= args.jpg_quality <= 100:
        parser.error("--jpg-quality must be between 1 and 100")
    if not 0 <= args.object_score_threshold <= 1:
        parser.error("--object-score-threshold must be between 0 and 1")
    if not 0 <= args.filtered_object_score_threshold <= 1:
        parser.error("--filtered-object-score-threshold must be between 0 and 1")
    if args.roi_padding_ratio < 0:
        parser.error("--roi-padding-ratio must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_stages:
        print("\n".join(STAGE_NAMES))
        return 0

    root = args.root.expanduser().resolve()
    video_path = args.video_path or root / "archive" / "video" / f"{args.video_id}.mp4"
    if not video_path.is_absolute():
        video_path = root / video_path
    args.video_path = video_path.resolve()

    if not args.dry_run and not args.video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {args.video_path}")
    if not args.dry_run and shutil.which("uv") is None:
        raise RuntimeError("uv is required but was not found on PATH")
    serving_selected = not args.skip_serving and stage_in_range(
        "serving_up", args.from_stage, args.to_stage
    )
    if not args.dry_run and serving_selected and shutil.which("docker") is None:
        raise RuntimeError("Docker is required for the serving stages")
    if (
        args.force
        and not args.skip_ocr
        and stage_in_range("ocr", args.from_stage, args.to_stage)
    ):
        print(
            "Warning: --force also reruns Gemini OCR and may incur API usage.",
            file=sys.stderr,
        )

    stages = build_stages(args, root)
    return run_pipeline(args, root, stages)


if __name__ == "__main__":
    raise SystemExit(main())
