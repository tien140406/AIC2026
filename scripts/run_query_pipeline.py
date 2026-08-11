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
QUERY_STAGE_NAMES = ("serving_up", "serving_check", "hybrid_query")


@dataclass(frozen=True)
class QueryStage:
    name: str
    command: tuple[str, ...]
    enabled: bool = True
    skip_reason: str | None = None


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


def append_option(
    command: tuple[str, ...], flag: str, value: object | None
) -> tuple[str, ...]:
    return (*command, flag, str(value)) if value is not None else command


def build_query_stages(args: argparse.Namespace) -> list[QueryStage]:
    compose_file = "environments/serving/docker-compose.local.yml"
    serving_check = uv_command(
        "serving",
        "check_local_serving.py",
        "--elasticsearch-url",
        args.elasticsearch_url,
        "--milvus-uri",
        args.milvus_uri,
    )
    hybrid_query = uv_command(
        "beit3-timm",
        "query_hybrid.py",
        "--query",
        args.query,
        "--limit",
        args.limit,
        "--candidate-limit",
        args.candidate_limit,
        "--rrf-k",
        args.rrf_k,
        "--device",
        args.device,
        "--elasticsearch-url",
        args.elasticsearch_url,
        "--frame-index",
        args.frame_index,
        "--asr-index",
        args.asr_index,
        "--milvus-uri",
        args.milvus_uri,
    )
    optional_values = (
        ("--video-id", args.video_id),
        ("--level", args.level),
        ("--model-name", args.model_name),
        ("--pretrained", args.pretrained),
        ("--embedding-version", args.embedding_version),
        ("--model-cache", args.model_cache),
        ("--keyframe-collection", args.keyframe_collection),
        ("--roi-collection", args.roi_collection),
    )
    for flag, value in optional_values:
        hybrid_query = append_option(hybrid_query, flag, value)
    if args.insecure:
        hybrid_query = (*hybrid_query, "--insecure")

    return [
        QueryStage(
            "serving_up",
            ("docker", "compose", "-f", compose_file, "up", "-d", "--wait"),
            enabled=args.start_serving,
            skip_reason=None if args.start_serving else "disabled by --no-start-serving",
        ),
        QueryStage(
            "serving_check",
            serving_check,
            enabled=args.check_serving,
            skip_reason=None if args.check_serving else "disabled by --no-check-serving",
        ),
        QueryStage("hybrid_query", hybrid_query),
    ]


def stage_in_range(name: str, from_stage: str, to_stage: str) -> bool:
    index = QUERY_STAGE_NAMES.index(name)
    return (
        QUERY_STAGE_NAMES.index(from_stage)
        <= index
        <= QUERY_STAGE_NAMES.index(to_stage)
    )


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


def run_query_pipeline(
    args: argparse.Namespace, root: Path, stages: Sequence[QueryStage]
) -> int:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = (
        root / "workspace" / "reports" / f"query_pipeline_{timestamp}.json"
    )
    report: dict[str, Any] = {
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
        record: dict[str, Any] = {
            "name": stage.name,
            "command": list(stage.command),
            "started_at": utc_now(),
            "finished_at": None,
            "elapsed_seconds": 0.0,
            "return_code": None,
        }
        print(f"\n=== [{position}/{len(selected)}] {stage.name} ===", flush=True)
        print(shlex.join(stage.command), flush=True)

        if not stage.enabled:
            record["status"] = "skipped_disabled"
            record["reason"] = stage.skip_reason
            print(f"Skipped: {stage.skip_reason}", flush=True)
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
            print(f"Query pipeline stopped at {stage.name}. Report: {report_path}")
            return int(record["return_code"] or 1)

    report["status"] = "dry_run" if args.dry_run else "completed"
    report["finished_at"] = utc_now()
    write_report(report, report_path)
    print(f"\nQuery pipeline {report['status']}. Report: {report_path}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start/check retrieval backends and run one hybrid Elasticsearch/Milvus query."
        )
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--query")
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--level", choices=("L0", "L1"), default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--embedding-version", default=None)
    parser.add_argument("--model-cache", type=Path, default=None)
    parser.add_argument(
        "--elasticsearch-url",
        default=os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"),
    )
    parser.add_argument(
        "--frame-index",
        default=os.getenv("ELASTICSEARCH_FRAME_INDEX", "aic-frames-v1"),
    )
    parser.add_argument(
        "--asr-index",
        default=os.getenv("ELASTICSEARCH_ASR_INDEX", "aic-asr-segments-v1"),
    )
    parser.add_argument(
        "--milvus-uri", default=os.getenv("MILVUS_URI", "http://localhost:19530")
    )
    parser.add_argument("--keyframe-collection", default=None)
    parser.add_argument("--roi-collection", default=None)
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument(
        "--start-serving",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start the repository's local Docker serving stack before querying.",
    )
    parser.add_argument(
        "--check-serving",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a read-only backend health check before querying.",
    )
    parser.add_argument("--list-stages", action="store_true")
    parser.add_argument(
        "--from-stage", choices=QUERY_STAGE_NAMES, default=QUERY_STAGE_NAMES[0]
    )
    parser.add_argument(
        "--to-stage", choices=QUERY_STAGE_NAMES, default=QUERY_STAGE_NAMES[-1]
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.list_stages:
        return args
    if not args.query:
        parser.error("--query is required")
    if QUERY_STAGE_NAMES.index(args.from_stage) > QUERY_STAGE_NAMES.index(
        args.to_stage
    ):
        parser.error("--from-stage must not come after --to-stage")
    positive = {
        "--limit": args.limit,
        "--candidate-limit": args.candidate_limit,
        "--rrf-k": args.rrf_k,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        parser.error(f"arguments must be positive: {', '.join(invalid)}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_stages:
        print("\n".join(QUERY_STAGE_NAMES))
        return 0

    root = args.root.expanduser().resolve()
    if not args.dry_run and shutil.which("uv") is None:
        raise RuntimeError("uv is required but was not found on PATH")
    serving_selected = args.start_serving and stage_in_range(
        "serving_up", args.from_stage, args.to_stage
    )
    if not args.dry_run and serving_selected and shutil.which("docker") is None:
        raise RuntimeError("Docker is required when --start-serving is enabled")

    stages = build_query_stages(args)
    return run_query_pipeline(args, root, stages)


if __name__ == "__main__":
    raise SystemExit(main())
