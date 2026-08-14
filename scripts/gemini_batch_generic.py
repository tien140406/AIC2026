from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "workspace/gemini_batches/generic_jobs.sqlite3"
sys.path.insert(0, str(ROOT))

from aic_pipeline.router_batch import router_batch_status, run_router_batch
from aic_pipeline.router_chat import RouterChatClient


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB); db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""CREATE TABLE IF NOT EXISTS jobs(
      task TEXT, video_id TEXT, input_hash TEXT PRIMARY KEY, model TEXT, input_path TEXT,
      uploaded_name TEXT, job_name TEXT, state TEXT, result_name TEXT, output_path TEXT,
      created_at TEXT, updated_at TEXT, error TEXT)""")
    return db


def paths(task: str, video_id: str) -> tuple[Path, Path]:
    if task == "asr-normalize":
        root = ROOT / "workspace/asr_normalization" / video_id
    elif task == "ocr-crops":
        root = ROOT / "workspace/ocr_fullres" / video_id
    else:
        raise ValueError(task)
    return root / ("requests.jsonl" if task == "asr-normalize" else "gemini_requests.jsonl"), root / "responses.jsonl"


def latest(db: sqlite3.Connection, task: str, video_id: str) -> sqlite3.Row:
    row = db.execute("SELECT * FROM jobs WHERE task=? AND video_id=? ORDER BY created_at DESC LIMIT 1",
                     (task, video_id)).fetchone()
    if not row:
        raise ValueError("Run register first")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable 9Router runner for prepared auxiliary tasks.")
    parser.add_argument("command", choices=("register", "submit", "status", "download"))
    parser.add_argument("--task", choices=("asr-normalize", "ocr-crops"), required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=64000)
    args = parser.parse_args()
    args.model = RouterChatClient(ROOT).model
    input_path, output_path = paths(args.task, args.video_id)
    with connect() as db:
        if args.command == "register":
            payload = input_path.read_bytes()
            if not payload.strip():
                print(json.dumps({"task": args.task, "video_id": args.video_id,
                                  "requests": 0, "status": "nothing_to_submit"}, indent=2))
                return
            digest = hashlib.sha256(payload).hexdigest()
            db.execute("""INSERT INTO jobs(task,video_id,input_hash,model,input_path,state,
                output_path,created_at,updated_at) VALUES(?,?,?,?,?,'PREPARED',?,?,?)
                ON CONFLICT(input_hash) DO UPDATE SET model=excluded.model,input_path=excluded.input_path,
                output_path=excluded.output_path,uploaded_name=NULL,job_name=NULL,state='PREPARED',
                result_name=NULL,error=NULL,updated_at=excluded.updated_at""",
                (args.task, args.video_id, digest, args.model, str(input_path), str(output_path), now(), now()))
            print(json.dumps({"task": args.task, "video_id": args.video_id, "requests": len(payload.splitlines()),
                              "sha256": digest, "input": str(input_path)}, indent=2)); return
        row = latest(db, args.task, args.video_id)
        if args.command == "submit":
            report = run_router_batch(
                ROOT, Path(row["input_path"]), output_path,
                workers=args.workers, max_retries=args.max_retries,
                timeout_seconds=args.timeout, default_max_tokens=args.max_tokens,
                progress=lambda result: print(json.dumps({
                    "key": result["key"], "ok": "response" in result,
                    "latency_seconds": (result.get("router") or {}).get("latency_seconds"),
                    "attempts": (result.get("router") or {}).get("attempts")},
                    ensure_ascii=False), flush=True),
            )
            state = "SUCCEEDED" if report["failed"] == 0 else "PARTIAL_FAILURE"
            error = None if report["failed"] == 0 else f"{report['failed']} requests failed"
            job = f"9router:{row['input_hash'][:10]}"
            db.execute("""UPDATE jobs SET uploaded_name=NULL,job_name=?,state=?,result_name=?,
                updated_at=?,error=? WHERE input_hash=?""",
                       (job, state, str(output_path), now(), error, row["input_hash"]))
            print(json.dumps({"job": job, "state": state, **report}, ensure_ascii=False, indent=2))
            if report["failed"]:
                raise RuntimeError(error)
            return
        counts = router_batch_status(Path(row["input_path"]), output_path)
        state = "SUCCEEDED" if counts["requests"] == counts["succeeded"] else row["state"]
        if args.command == "download" and state != "SUCCEEDED":
            raise RuntimeError(f"Batch is not ready: {state}")
        print(json.dumps({"job": row["job_name"], "state": state,
                          "model": row["model"], "result": str(output_path), **counts}, indent=2))


if __name__ == "__main__":
    main()
