from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageOps

from aic_pipeline.gemini_ocr import (
    PROMPT_VERSION,
    FrameOcr,
    build_prompt,
    classify_error,
    failure_path,
    file_sha256,
    reusable_success,
    success_path,
    write_summary_tables,
)
from aic_pipeline.io import atomic_write_json

THREAD_STATE = threading.local()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model).strip("_")


def client_for_thread(api_key: str, request_timeout_seconds: float) -> genai.Client:
    timeout_ms = max(1, round(request_timeout_seconds * 1000))
    config_key = timeout_ms
    client = getattr(THREAD_STATE, "client", None)
    if client is None or getattr(THREAD_STATE, "config_key", None) != config_key:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=timeout_ms,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        THREAD_STATE.client = client
        THREAD_STATE.config_key = config_key
    return client


def usage_payload(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage_metadata", None)
    return usage.model_dump(mode="json") if usage else None


def failure_payload(
    row: dict[str, Any],
    *,
    model: str,
    attempt: int,
    error: Exception,
) -> dict[str, Any]:
    disposition = classify_error(error)
    return {
        "video_id": row["video_id"],
        "shot_id": int(row["shot_id"]) if pd.notna(row.get("shot_id")) else None,
        "frame_id": int(row["frame_id"]),
        "pts_time": float(row["pts_time"]) if pd.notna(row.get("pts_time")) else None,
        "image_path": str(row["image_path"]),
        "image_sha256": row["image_sha256"],
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "attempts": attempt,
        "error_category": disposition.category,
        "status_code": disposition.status_code,
        "retry_after_seconds": disposition.retry_after_seconds,
        "terminal": disposition.terminal,
        "error_type": type(error).__name__,
        "error": str(error),
        "failed_at": utc_now(),
    }


def process_frame(
    row: dict[str, Any],
    *,
    api_key: str,
    model: str,
    output_dir: Path,
    max_attempts: int,
    retry_base_seconds: float,
    request_timeout_seconds: float,
    stop_event: threading.Event,
) -> dict[str, Any]:
    frame_id = int(row["frame_id"])
    image_path = Path(row["image_path"])
    started = time.monotonic()
    if stop_event.is_set():
        return {"status": "cancelled", "frame_id": frame_id}

    try:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB").copy()
    except (OSError, ValueError) as error:
        failure = failure_payload(row, model=model, attempt=0, error=error)
        atomic_write_json(failure, failure_path(output_dir, frame_id))
        return {"status": "failure", "frame_id": frame_id, "payload": failure}

    for attempt in range(1, max_attempts + 1):
        if stop_event.is_set():
            return {"status": "cancelled", "frame_id": frame_id}
        try:
            response = client_for_thread(
                api_key, request_timeout_seconds
            ).models.generate_content(
                model=model,
                contents=[build_prompt(frame_id), image],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=FrameOcr,
                    temperature=0.0,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.MINIMAL,
                    ),
                ),
            )
            if not response.text:
                raise ValueError("Gemini returned an empty response")
            parsed = FrameOcr.model_validate_json(response.text)
            if parsed.frame_id != frame_id:
                raise ValueError(
                    f"Response frame_id {parsed.frame_id} does not match input {frame_id}"
                )
            payload = {
                "video_id": row["video_id"],
                "shot_id": int(row["shot_id"])
                if pd.notna(row.get("shot_id"))
                else None,
                "frame_id": frame_id,
                "pts_time": float(row["pts_time"])
                if pd.notna(row.get("pts_time"))
                else None,
                "image_path": str(image_path),
                "image_sha256": row["image_sha256"],
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "attempts": attempt,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "processed_at": utc_now(),
                "ocr": parsed.model_dump(mode="json"),
                "response_text": response.text,
                "usage": usage_payload(response),
            }
            atomic_write_json(payload, success_path(output_dir, frame_id))
            return {"status": "success", "frame_id": frame_id, "payload": payload}
        except Exception as error:  # noqa: BLE001 - API/schema errors share one retry policy.
            disposition = classify_error(error)
            if disposition.terminal:
                stop_event.set()
            if (
                disposition.terminal
                or not disposition.retryable
                or attempt >= max_attempts
            ):
                failure = failure_payload(
                    row, model=model, attempt=attempt, error=error
                )
                atomic_write_json(failure, failure_path(output_dir, frame_id))
                status = "quota_exhausted" if disposition.terminal else "failure"
                return {"status": status, "frame_id": frame_id, "payload": failure}

            exponential_delay = min(retry_base_seconds * (2 ** (attempt - 1)), 60.0)
            delay = max(exponential_delay, disposition.retry_after_seconds or 0.0)
            jitter = random.uniform(0.0, min(1.0, delay / 4))
            if stop_event.wait(delay + jitter):
                return {"status": "cancelled", "frame_id": frame_id}

    raise AssertionError("Unreachable retry state")


def sum_usage(output_dir: Path) -> dict[str, int]:
    totals: dict[str, int] = {}
    for path in (output_dir / "raw").glob("frame_*.json"):
        try:
            usage = json.loads(path.read_text(encoding="utf-8")).get("usage") or {}
        except (OSError, json.JSONDecodeError):
            continue
        for key, value in usage.items():
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0) + value
    return totals


def load_jobs(
    manifest_path: Path,
    video_id: str,
    offset: int,
    limit: int | None,
    hash_workers: int,
) -> pd.DataFrame:
    manifest = pd.read_parquet(manifest_path)
    required = {"video_id", "frame_id", "image_path", "image_exists"}
    missing_columns = sorted(required - set(manifest.columns))
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {missing_columns}")

    jobs = manifest.loc[
        (manifest["video_id"] == video_id) & manifest["image_exists"].astype(bool)
    ].copy()
    sort_columns = [
        column for column in ("pts_time", "frame_id") if column in jobs.columns
    ]
    jobs = jobs.sort_values(sort_columns).reset_index(drop=True)
    if jobs.empty:
        raise ValueError(f"No usable images found for {video_id} in {manifest_path}")
    if jobs["frame_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate frame IDs")
    jobs = jobs.iloc[offset:]
    if limit is not None:
        jobs = jobs.iloc[:limit]
    if jobs.empty:
        raise ValueError("No jobs remain after applying --offset/--limit")

    image_paths = [Path(path) for path in jobs["image_path"]]
    missing_images = [path for path in image_paths if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"Missing {len(missing_images)} images; first: {missing_images[0]}"
        )
    if hash_workers == 1:
        hashes = [file_sha256(path) for path in image_paths]
    else:
        with ThreadPoolExecutor(max_workers=hash_workers) as executor:
            hashes = list(executor.map(file_sha256, image_paths))
    jobs["image_sha256"] = hashes
    return jobs


def checkpoint_report(
    *,
    selection: dict[str, Any],
    output_dir: Path,
    selected_ids: set[int],
    counters: dict[str, int],
    status: str,
    started: float,
    terminal_error: dict[str, Any] | None,
) -> dict[str, Any]:
    frames, failure_frame = write_summary_tables(output_dir)
    successful_ids = set(frames["frame_id"].astype(int)) & selected_ids
    failed_ids = set(failure_frame["frame_id"].astype(int)) & selected_ids
    failed_ids -= successful_ids
    selected_frames = frames.loc[frames["frame_id"].isin(successful_ids)]
    report = {
        **selection,
        "status": status,
        "new_successes": counters["success"],
        "new_failures": counters["failure"],
        "cancelled_in_run": counters["cancelled"],
        "processed_in_run": counters["completed"],
        "selected_successes": len(successful_ids),
        "selected_failures": len(failed_ids),
        "selected_remaining": len(selected_ids - successful_ids - failed_ids),
        "frames_with_text": int(selected_frames["has_text"].sum())
        if not selected_frames.empty
        else 0,
        "usage": sum_usage(output_dir),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "checkpointed_at": utc_now(),
        "terminal_error": terminal_error,
        "outputs": {
            "raw_directory": str(output_dir / "raw"),
            "failure_directory": str(output_dir / "failures"),
            "frames": str(output_dir / "frames.parquet"),
            "failures": str(output_dir / "failures.parquet"),
            "report": str(output_dir / "report.json"),
        },
    }
    atomic_write_json(report, output_dir / "report.json")
    atomic_write_json(
        report,
        ROOT / "workspace" / "reports" / f"gemini_ocr_{selection['video_id']}.json",
    )
    return report


def submit_until_full(
    executor: ThreadPoolExecutor,
    jobs: Iterator[dict[str, Any]],
    in_flight: dict[Future[dict[str, Any]], dict[str, Any]],
    *,
    concurrency: int,
    submit_kwargs: dict[str, Any],
    stop_event: threading.Event,
) -> bool:
    exhausted = False
    while len(in_flight) < concurrency and not stop_event.is_set():
        try:
            row = next(jobs)
        except StopIteration:
            exhausted = True
            break
        future = executor.submit(process_frame, row, **submit_kwargs)
        in_flight[future] = row
    return exhausted


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frame-level OCR with Gemini.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--hash-workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--retry-base-seconds", type=float, default=1.5)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    positive_values = {
        "--concurrency": args.concurrency,
        "--hash-workers": args.hash_workers,
        "--checkpoint-every": args.checkpoint_every,
        "--progress-every": args.progress_every,
        "--max-attempts": args.max_attempts,
        "--request-timeout-seconds": args.request_timeout_seconds,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid or args.retry_base_seconds < 0:
        raise ValueError(f"Invalid non-positive arguments: {invalid}")

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key and not args.dry_run:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    model = args.model or os.getenv("GEMINI_MODEL") or "gemini-3.6-flash"
    manifest_path = args.manifest or (
        ROOT / "workspace" / "keyframe_images" / args.video_id / "manifest.parquet"
    )
    jobs = load_jobs(
        manifest_path, args.video_id, args.offset, args.limit, args.hash_workers
    )
    output_dir = (
        ROOT / "workspace" / "ocr" / "gemini" / model_slug(model) / args.video_id
    )

    pending: list[dict[str, Any]] = []
    skipped = 0
    for row in jobs.to_dict(orient="records"):
        frame_id = int(row["frame_id"])
        path = success_path(output_dir, frame_id)
        if not args.force and reusable_success(
            path,
            frame_id=frame_id,
            image_sha256=row["image_sha256"],
            model=model,
        ):
            skipped += 1
        else:
            pending.append(row)

    selection = {
        "video_id": args.video_id,
        "manifest": str(manifest_path),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "selected_jobs": len(jobs),
        "pending_jobs": len(pending),
        "reusable_successes": skipped,
        "concurrency": args.concurrency,
        "hash_workers": args.hash_workers,
        "checkpoint_every": args.checkpoint_every,
        "output_dir": str(output_dir),
    }
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    assert api_key is not None

    started = time.monotonic()
    selected_ids = set(jobs["frame_id"].astype(int))
    counters = {"completed": 0, "success": 0, "failure": 0, "cancelled": 0}
    stop_event = threading.Event()
    terminal_error: dict[str, Any] | None = None
    interrupted = False
    quota_exhausted = False
    next_checkpoint = args.checkpoint_every
    in_flight: dict[Future[dict[str, Any]], dict[str, Any]] = {}
    job_iterator = iter(pending)
    executor = ThreadPoolExecutor(max_workers=args.concurrency)
    submit_kwargs = {
        "api_key": api_key,
        "model": model,
        "output_dir": output_dir,
        "max_attempts": args.max_attempts,
        "retry_base_seconds": args.retry_base_seconds,
        "request_timeout_seconds": args.request_timeout_seconds,
        "stop_event": stop_event,
    }

    try:
        jobs_exhausted = submit_until_full(
            executor,
            job_iterator,
            in_flight,
            concurrency=args.concurrency,
            submit_kwargs=submit_kwargs,
            stop_event=stop_event,
        )
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                row = in_flight.pop(future)
                try:
                    result = future.result()
                except Exception as error:  # noqa: BLE001 - isolate worker failures.
                    failure = failure_payload(row, model=model, attempt=0, error=error)
                    atomic_write_json(
                        failure,
                        failure_path(output_dir, int(row["frame_id"])),
                    )
                    result = {
                        "status": "failure",
                        "frame_id": int(row["frame_id"]),
                        "payload": failure,
                    }
                status = result["status"]
                if status == "cancelled":
                    counters["cancelled"] += 1
                    continue
                counters["completed"] += 1
                if status == "success":
                    counters["success"] += 1
                else:
                    counters["failure"] += 1
                if status == "quota_exhausted":
                    quota_exhausted = True
                    terminal_error = result.get("payload")
                    stop_event.set()

            if stop_event.is_set():
                for future in in_flight:
                    future.cancel()
            elif not jobs_exhausted:
                jobs_exhausted = submit_until_full(
                    executor,
                    job_iterator,
                    in_flight,
                    concurrency=args.concurrency,
                    submit_kwargs=submit_kwargs,
                    stop_event=stop_event,
                )

            if counters["completed"] >= next_checkpoint:
                checkpoint_report(
                    selection=selection,
                    output_dir=output_dir,
                    selected_ids=selected_ids,
                    counters=counters,
                    status="running",
                    started=started,
                    terminal_error=terminal_error,
                )
                while next_checkpoint <= counters["completed"]:
                    next_checkpoint += args.checkpoint_every
                print(
                    f"Checkpointed {counters['completed']:,} processed frames",
                    flush=True,
                )

            if (
                counters["completed"] == 1
                or counters["completed"] % args.progress_every == 0
            ):
                print(
                    f"Processed {counters['completed']}/{len(pending)}: "
                    f"success={counters['success']}, failure={counters['failure']}",
                    flush=True,
                )

    except KeyboardInterrupt:
        interrupted = True
        stop_event.set()
        print(
            "Interrupted; waiting for active requests to stop before checkpointing",
            flush=True,
        )
    finally:
        stop_event.set()
        for future in in_flight:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)

    if interrupted:
        final_status = "interrupted"
    elif quota_exhausted:
        final_status = "quota_exhausted"
    else:
        final_status = "completed"
    report = checkpoint_report(
        selection=selection,
        output_dir=output_dir,
        selected_ids=selected_ids,
        counters=counters,
        status=final_status,
        started=started,
        terminal_error=terminal_error,
    )
    if final_status == "completed" and report["selected_failures"]:
        final_status = "completed_with_failures"
        report["status"] = final_status
        atomic_write_json(report, output_dir / "report.json")
        atomic_write_json(
            report, ROOT / "workspace" / "reports" / f"gemini_ocr_{args.video_id}.json"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if final_status == "interrupted":
        raise SystemExit(130)
    if final_status == "quota_exhausted":
        raise SystemExit(3)
    if final_status == "completed_with_failures":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
