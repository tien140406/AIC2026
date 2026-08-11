from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
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


def safe_file_component(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or "result"


def format_pts(pts_ms: object) -> str:
    try:
        milliseconds = int(pts_ms)
    except (TypeError, ValueError):
        return "unknown"
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def render_query_report(
    query_result: dict[str, Any],
    *,
    root: Path,
    html_path: Path,
    image_directory: Path,
) -> dict[str, Any]:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    image_directory.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    exported_images: list[str] = []

    results = query_result.get("results") or []
    for rank, result in enumerate(results, start=1):
        video_id = result.get("video_id", "unknown")
        frame_id = result.get("frame_id", "unknown")
        image_path = result.get("image_path")
        image_markup = '<div class="missing-image">Image unavailable</div>'
        if image_path:
            source = Path(str(image_path))
            if not source.is_absolute():
                source = root / source
            if source.is_file():
                suffix = source.suffix.lower() or ".jpg"
                image_name = (
                    f"{rank:03d}_{safe_file_component(video_id)}_"
                    f"{safe_file_component(frame_id)}{suffix}"
                )
                exported = image_directory / image_name
                shutil.copy2(source, exported)
                exported_images.append(str(exported))
                result["report_image_path"] = str(exported)
                image_href = os.path.relpath(exported, html_path.parent)
                image_markup = (
                    f'<a class="image-link" href="{escape(image_href)}">'
                    f'<img src="{escape(image_href)}" '
                    f'alt="Result {rank}: {escape(str(video_id))} frame '
                    f'{escape(str(frame_id))}" loading="lazy"></a>'
                )

        evidence_items = []
        for evidence in result.get("evidence") or []:
            label = escape(str(evidence.get("source", "unknown")))
            evidence_rank = escape(str(evidence.get("rank", "-")))
            score = evidence.get("score")
            score_text = f"{float(score):.4f}" if score is not None else "-"
            class_name = evidence.get("class_name")
            class_text = (
                f" / {escape(str(class_name))}" if class_name is not None else ""
            )
            evidence_items.append(
                f"<li><strong>{label}</strong><span>rank {evidence_rank} / "
                f"score {score_text}{class_text}</span></li>"
            )
        evidence_markup = "".join(evidence_items) or "<li>No evidence</li>"

        frame = result.get("frame") or {}
        ocr_text = frame.get("ocr_text") or ""
        rrf_score = result.get("rrf_score")
        rrf_text = f"{float(rrf_score):.6f}" if rrf_score is not None else "-"
        cards.append(
            f"""
            <article class="result-card">
              <div class="rank">#{rank}</div>
              {image_markup}
              <div class="content">
                <div class="identity">
                  <h2>{escape(str(video_id))}</h2>
                  <span>frame {escape(str(frame_id))}</span>
                </div>
                <dl>
                  <div><dt>Timestamp</dt><dd>{format_pts(result.get("pts_ms"))}</dd></div>
                  <div><dt>Shot</dt><dd>{escape(str(result.get("shot_id", "-")))}</dd></div>
                  <div><dt>RRF</dt><dd>{rrf_text}</dd></div>
                </dl>
                <p class="ocr">{escape(str(ocr_text)) if ocr_text else "No OCR text"}</p>
                <ul class="evidence">{evidence_markup}</ul>
              </div>
            </article>
            """
        )

    query = escape(str(query_result.get("query", "")))
    video_filter = escape(str(query_result.get("video_id") or "all videos"))
    result_count = len(results)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AIC Query Report</title>
  <style>
    :root {{
      --paper: #f2eadb;
      --ink: #17231e;
      --muted: #657168;
      --accent: #cb4b2c;
      --forest: #244b3d;
      --line: #d5c9b6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 5%, #fff8e8 0 12%, transparent 36%),
        linear-gradient(135deg, var(--paper), #e2dccd);
      font-family: "Trebuchet MS", sans-serif;
    }}
    header {{
      padding: 48px clamp(20px, 6vw, 88px) 36px;
      color: #fffaf0;
      background: var(--forest);
      border-bottom: 8px solid var(--accent);
    }}
    header p {{ margin: 8px 0 0; color: #cddbd4; }}
    h1 {{ margin: 0; font: 700 clamp(32px, 6vw, 72px)/1 Georgia, serif; }}
    main {{ width: min(1400px, 94vw); margin: 34px auto 80px; }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px 28px;
      margin-bottom: 24px;
      color: var(--muted);
    }}
    .result-grid {{ display: grid; gap: 22px; }}
    .result-card {{
      position: relative;
      display: grid;
      grid-template-columns: minmax(280px, 42%) 1fr;
      overflow: hidden;
      background: rgba(255, 253, 247, 0.94);
      border: 1px solid var(--line);
      box-shadow: 0 16px 34px rgba(36, 45, 38, 0.12);
    }}
    .rank {{
      position: absolute;
      z-index: 2;
      top: 14px;
      left: 14px;
      padding: 7px 11px;
      color: white;
      background: var(--accent);
      font-weight: 800;
    }}
    .image-link, .missing-image {{ min-height: 270px; background: #c9c3b5; }}
    img {{ display: block; width: 100%; height: 100%; min-height: 270px; object-fit: cover; }}
    .missing-image {{ display: grid; place-items: center; color: var(--muted); }}
    .content {{ padding: clamp(20px, 3vw, 38px); }}
    .identity {{ display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }}
    h2 {{ margin: 0; font: 700 30px/1 Georgia, serif; }}
    .identity span {{ color: var(--accent); font-weight: 700; }}
    dl {{ display: flex; flex-wrap: wrap; gap: 10px 24px; margin: 20px 0; }}
    dl div {{ min-width: 110px; }}
    dt {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    dd {{ margin: 3px 0 0; font-weight: 700; }}
    .ocr {{ padding: 14px; background: #f4eddf; border-left: 4px solid var(--accent); }}
    .evidence {{ list-style: none; padding: 0; margin: 18px 0 0; }}
    .evidence li {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 8px 0;
      border-bottom: 1px solid var(--line);
    }}
    .evidence span {{ color: var(--muted); text-align: right; }}
    @media (max-width: 760px) {{
      header {{ padding-top: 34px; }}
      .result-card {{ grid-template-columns: 1fr; }}
      .image-link, .missing-image, img {{ min-height: 210px; max-height: 360px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Query Results</h1>
    <p>{query}</p>
  </header>
  <main>
    <div class="summary">
      <span><strong>{result_count}</strong> results</span>
      <span>Scope: <strong>{video_filter}</strong></span>
      <span>Generated: <strong>{escape(utc_now())}</strong></span>
    </div>
    <section class="result-grid">
      {''.join(cards) if cards else '<p>No results found.</p>'}
    </section>
  </main>
</body>
</html>
"""
    html_path.write_text(document, encoding="utf-8")
    return {
        "html_report": str(html_path),
        "image_directory": str(image_directory),
        "exported_images": exported_images,
    }


def run_query_pipeline(
    args: argparse.Namespace, root: Path, stages: Sequence[QueryStage]
) -> int:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = (
        root / "workspace" / "reports" / f"query_pipeline_{timestamp}.json"
    )
    html_path = root / "workspace" / "reports" / f"query_report_{timestamp}.html"
    image_directory = (
        root / "workspace" / "reports" / f"query_assets_{timestamp}"
    )
    report: dict[str, Any] = {
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "root": str(root),
        "options": vars(args),
        "outputs": {"json_report": str(report_path)},
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
                capture_output = stage.name == "hybrid_query"
                completed = subprocess.run(
                    stage.command,
                    cwd=root,
                    check=False,
                    stdout=subprocess.PIPE if capture_output else None,
                    text=capture_output,
                )
                if capture_output and completed.stdout:
                    print(completed.stdout, end="")
                record["return_code"] = completed.returncode
                record["status"] = (
                    "completed" if completed.returncode == 0 else "failed"
                )
                if capture_output and completed.returncode == 0:
                    try:
                        query_result = json.loads(completed.stdout)
                    except json.JSONDecodeError as error:
                        record["status"] = "failed"
                        record["return_code"] = 1
                        record["error"] = f"Invalid query JSON output: {error}"
                    else:
                        report["query_result"] = query_result
                        report["outputs"].update(
                            render_query_report(
                                query_result,
                                root=root,
                                html_path=html_path,
                                image_directory=image_directory,
                            )
                        )
                        record["result_count"] = len(
                            query_result.get("results") or []
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
    if html_path.is_file():
        print(f"Visual report: {html_path}")
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
