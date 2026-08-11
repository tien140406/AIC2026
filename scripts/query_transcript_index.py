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
    find_transcript_segments_by_frame_uid,
    find_transcript_segments_by_keyframe,
    search_transcripts,
)
from aic_pipeline.common.paths import PipelinePaths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query an existing transcript index without rebuilding it."
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--index", type=Path, default=None)
    lookup = parser.add_mutually_exclusive_group(required=True)
    lookup.add_argument("--query", help="FTS5 transcript query.")
    lookup.add_argument("--frame-uid", help="Stable frame UID returned by Milvus.")
    lookup.add_argument(
        "--keyframe-video-id",
        help="Video ID for a reverse keyframe-to-transcript query.",
    )
    parser.add_argument(
        "--frame-id",
        type=int,
        default=None,
        help="Original frame ID paired with --keyframe-video-id.",
    )
    parser.add_argument("--level", choices=("L0", "L1"), default=None)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if (args.keyframe_video_id is None) != (args.frame_id is None):
        parser.error("--keyframe-video-id and --frame-id must be supplied together")

    paths = PipelinePaths.from_root(args.root)
    destination = (args.index or paths.indexes / "transcripts.sqlite").resolve()
    if not destination.is_file():
        raise FileNotFoundError(
            f"Transcript index is missing: {destination}. "
            "Run scripts/build_transcript_index.py first."
        )

    connection = sqlite3.connect(f"{destination.as_uri()}?mode=ro", uri=True)
    try:
        if args.query is not None:
            mode = "text"
            query: object = args.query
            results = search_transcripts(
                connection,
                args.query,
                limit=args.limit,
                level=args.level,
            )
        elif args.frame_uid is not None:
            mode = "frame_uid"
            query = args.frame_uid
            results = find_transcript_segments_by_frame_uid(
                connection,
                args.frame_uid,
                limit=args.limit,
                level=args.level,
            )
        else:
            mode = "keyframe"
            query = {
                "video_id": args.keyframe_video_id,
                "frame_id": args.frame_id,
            }
            results = find_transcript_segments_by_keyframe(
                connection,
                args.keyframe_video_id,
                args.frame_id,
                limit=args.limit,
                level=args.level,
            )
    finally:
        connection.close()

    print(
        json.dumps(
            {
                "index": str(destination),
                "mode": mode,
                "query": query,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
