from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aic_pipeline.asr.index import (
    build_transcript_index,
    find_transcript_segments_by_keyframe,
    search_transcripts,
)
from aic_pipeline.common.paths import PipelinePaths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the SQLite FTS5 transcript index."
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--level", action="append", choices=("L0", "L1"), default=[])
    parser.add_argument(
        "--query", default=None, help="Optional smoke query after build."
    )
    parser.add_argument(
        "--keyframe-video-id",
        default=None,
        help="Optional video ID for a reverse keyframe-to-transcript query.",
    )
    parser.add_argument(
        "--frame-id",
        type=int,
        default=None,
        help="Original frame ID paired with --keyframe-video-id.",
    )
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if (args.keyframe_video_id is None) != (args.frame_id is None):
        parser.error("--keyframe-video-id and --frame-id must be supplied together")

    paths = PipelinePaths.from_root(args.root)
    destination = args.output or paths.indexes / "transcripts.sqlite"
    levels = tuple(args.level) if args.level else ("L0", "L1")
    report = build_transcript_index(
        paths.transcripts, destination, include_levels=levels
    )
    if args.query or args.keyframe_video_id is not None:
        connection = sqlite3.connect(destination)
        try:
            if args.query:
                report["query"] = args.query
                report["results"] = search_transcripts(
                    connection, args.query, limit=args.limit
                )
            if args.keyframe_video_id is not None:
                report["reverse_query"] = {
                    "video_id": args.keyframe_video_id,
                    "frame_id": args.frame_id,
                }
                report["reverse_results"] = find_transcript_segments_by_keyframe(
                    connection,
                    args.keyframe_video_id,
                    args.frame_id,
                    limit=args.limit,
                )
        finally:
            connection.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
