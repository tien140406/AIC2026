from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aic_pipeline.scene_evidence import adaptive_frame_ids
from aic_pipeline.router_batch import router_batch_status, run_router_batch
from aic_pipeline.router_chat import RouterChatClient


BATCH_ROOT = ROOT / "workspace" / "gemini_batches"
DB_PATH = BATCH_ROOT / "jobs.sqlite3"
SCHEMA = {
    "type": "object",
    "properties": {
        "summary_vi": {"type": "string"},
        "ocr_lines": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "frame_label": {"type": "string"}},
            "required": ["text", "frame_label"], "additionalProperties": False}},
        "people": {"type": "array", "items": {"type": "string"}},
        "objects": {"type": "array", "items": {"type": "string"}},
        "actions": {"type": "array", "items": {"type": "string"}},
        "locations": {"type": "array", "items": {"type": "string"}},
        "events": {"type": "array", "items": {"type": "string"}},
        "search_keywords_vi": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary_vi", "ocr_lines", "people", "objects", "actions", "locations", "events",
                 "search_keywords_vi", "uncertainties"],
    "additionalProperties": False,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def db() -> sqlite3.Connection:
    BATCH_ROOT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
        input_hash TEXT PRIMARY KEY, video_id TEXT NOT NULL, model TEXT NOT NULL,
        jsonl_path TEXT NOT NULL, uploaded_file_name TEXT, batch_job_name TEXT,
        state TEXT NOT NULL, result_file_name TEXT, created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL, last_error TEXT)""")
    return conn


def load_scenes(video_id: str) -> list[dict[str, Any]]:
    path = ROOT / "workspace" / "semantic_scenes" / video_id / "scenes.json"
    return json.loads(path.read_text(encoding="utf-8"))["scenes"]


def video_path(video_id: str) -> Path:
    rows = pd.read_parquet(ROOT / "workspace" / "manifests" / "videos.parquet")
    row = rows.loc[rows.video_id == video_id]
    if len(row) != 1:
        raise ValueError(f"Expected one manifest row for {video_id}")
    return Path(row.iloc[0].video_path)


def extract_frames(path: Path, frame_ids: set[int]) -> dict[int, Image.Image]:
    import cv2
    capture = cv2.VideoCapture(str(path))
    wanted = sorted(frame_ids)
    output: dict[int, Image.Image] = {}
    for frame_id in wanted:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
        if ok:
            output[frame_id] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    missing = frame_ids - output.keys()
    if missing:
        raise RuntimeError(f"Could not decode {len(missing)} representative frames")
    return output


def contact_sheet(scene: dict[str, Any], frames: dict[int, Image.Image], output: Path,
                  frame_ids: list[int] | None = None) -> None:
    ids = frame_ids or [int(row["frame_id"]) for row in scene["representative_frames"]]
    tile_w, tile_h, label_h, cols = 320, 180, 24, 5
    rows = (len(ids) + cols - 1) // cols
    sheet = Image.new("RGB", (tile_w * cols, (tile_h + label_h) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, frame_id in enumerate(ids):
        image = frames[int(frame_id)].copy()
        image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (tile_w, tile_h), "black")
        canvas.paste(image, ((tile_w - image.width) // 2, (tile_h - image.height) // 2))
        x, y = (index % cols) * tile_w, (index // cols) * (tile_h + label_h)
        sheet.paste(canvas, (x, y))
        fps = max(1e-6, (int(scene["end_frame_exclusive"]) - int(scene["start_frame"]))
                  / max(1e-6, float(scene["end_time"]) - float(scene["start_time"])))
        pts_time = float(scene["start_time"]) + (int(frame_id) - int(scene["start_frame"])) / fps
        label = f"F{index + 1} | frame {frame_id} | {pts_time:.2f}s"
        draw.text((x + 6, y + tile_h + 7), label, fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "JPEG", quality=82, optimize=True)


def prompt(scene: dict[str, Any]) -> str:
    transcript = scene.get("asr_exact") or scene.get("asr_context") or "(không có transcript)"
    return f"""Phân tích scene video AIC từ các frame đại diện F1..Fn theo thời gian.
Chỉ ghi nhận chi tiết nhìn thấy rõ hoặc được transcript hỗ trợ; không đoán danh tính.
OCR phải chép nguyên văn và gắn frame_label. Mô tả/từ khóa bằng tiếng Việt, giữ nguyên tên riêng.
Không đưa vào ocr_lines: đồng hồ/timestamp, ngày giờ phát sóng, watermark/logo kênh,
website/hotline cố định, ticker chạy hoặc footer lặp lại. Vẫn giữ lower-third nếu nó
chứa tên người, chức vụ, địa danh, tiêu đề hoặc thông tin riêng của scene.
Khoảng thời gian: {scene['start_time']:.2f}s–{scene['end_time']:.2f}s.
Transcript ASR (có thể sai): {transcript}
Trả về đúng JSON theo schema."""


def prepare(args: argparse.Namespace) -> None:
    scenes = load_scenes(args.video_id)
    out_dir = BATCH_ROOT / args.video_id
    sheets = out_dir / "contact_sheets"
    jsonl = out_dir / "requests.jsonl"
    evidence = {str(scene["scene_id"]): adaptive_frame_ids(scene) for scene in scenes}
    ids = {frame_id for values in evidence.values() for frame_id in values}
    decoded = extract_frames(video_path(args.video_id), ids)
    lines: list[str] = []
    for scene_index, scene in enumerate(scenes):
        scene_id = str(scene["scene_id"])
        image_path = sheets / f"scene_{scene_index:05d}.jpg"
        scene_ids = evidence[scene_id]
        contact_sheet(scene, decoded, image_path, scene_ids)
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        request = {
            "contents": [{"role": "user", "parts": [
                {"text": prompt(scene)},
                {"inlineData": {"mimeType": "image/jpeg", "data": encoded}},
            ]}],
            "generationConfig": {"temperature": 0.1,
                                   "thinkingConfig": {"thinkingLevel": "MINIMAL"},
                                   "responseMimeType": "application/json",
                                   "responseJsonSchema": SCHEMA},
        }
        lines.append(json.dumps({"key": scene_id, "request": request}, ensure_ascii=False))
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    with db() as conn:
        conn.execute("""INSERT INTO jobs(input_hash,video_id,model,jsonl_path,state,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?) ON CONFLICT(input_hash) DO UPDATE SET jsonl_path=excluded.jsonl_path,
            model=excluded.model,state='PREPARED',uploaded_file_name=NULL,batch_job_name=NULL,
            result_file_name=NULL,last_error=NULL,updated_at=excluded.updated_at""",
            (digest, args.video_id, args.model, str(jsonl), "PREPARED", now(), now()))
    print(json.dumps({"video_id": args.video_id, "scenes": len(scenes), "unique_frames": len(ids),
                      "jsonl": str(jsonl), "bytes": len(payload), "sha256": digest}, indent=2))


def find_job(conn: sqlite3.Connection, video_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM jobs WHERE video_id=? ORDER BY created_at DESC LIMIT 1", (video_id,)).fetchone()
    if not row:
        raise ValueError(f"No prepared batch for {video_id}")
    return row


def submit(args: argparse.Namespace) -> None:
    output = BATCH_ROOT / args.video_id / "responses.jsonl"
    with db() as conn:
        row = find_job(conn, args.video_id)
        input_path = Path(row["jsonl_path"])
        report = run_router_batch(
            ROOT, input_path, output, workers=args.workers,
            max_retries=args.max_retries, timeout_seconds=args.timeout,
            default_max_tokens=args.max_tokens,
            progress=lambda result: print(json.dumps({
                "key": result["key"], "ok": "response" in result,
                "latency_seconds": (result.get("router") or {}).get("latency_seconds"),
                "attempts": (result.get("router") or {}).get("attempts")},
                ensure_ascii=False), flush=True),
        )
        state = "SUCCEEDED" if report["failed"] == 0 else "PARTIAL_FAILURE"
        error = None if report["failed"] == 0 else f"{report['failed']} requests failed"
        conn.execute("""UPDATE jobs SET model=?,uploaded_file_name=NULL,batch_job_name=?,state=?,
            result_file_name=?,updated_at=?,last_error=? WHERE input_hash=?""",
            (report["model"], f"9router:{row['input_hash'][:12]}", state,
             str(output), now(), error, row["input_hash"]))
    print(json.dumps({"job": f"9router:{row['input_hash'][:12]}", "state": state,
                      **report}, ensure_ascii=False, indent=2))
    if report["failed"]:
        raise RuntimeError(error)


def status(args: argparse.Namespace, download_result: bool = False) -> None:
    with db() as conn:
        row = find_job(conn, args.video_id)
        output = BATCH_ROOT / args.video_id / "responses.jsonl"
        counts = router_batch_status(Path(row["jsonl_path"]), output)
        state = "SUCCEEDED" if counts["requests"] == counts["succeeded"] else row["state"]
        if download_result and state != "SUCCEEDED":
            raise RuntimeError(f"Batch is not ready: {state}")
        print(json.dumps({"job": row["batch_job_name"], "state": state,
                          "model": row["model"], "result_file": str(output), **counts}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and run resumable 9Router scene enrichment.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "submit", "status", "download"):
        item = sub.add_parser(name)
        item.add_argument("--video-id", required=True)
        item.add_argument("--workers", type=int, default=4)
        item.add_argument("--max-retries", type=int, default=3)
        item.add_argument("--timeout", type=float, default=120)
        item.add_argument("--max-tokens", type=int, default=64000)
    args = parser.parse_args()
    args.model = RouterChatClient(ROOT).model
    if args.command == "prepare":
        prepare(args)
    elif args.command == "submit":
        submit(args)
    else:
        status(args, download_result=args.command == "download")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
