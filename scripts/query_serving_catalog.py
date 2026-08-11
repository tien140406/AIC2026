from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aic_pipeline.common.paths import PipelinePaths
from aic_pipeline.serving.catalog import frame_bundle, search_objects, search_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the global serving catalog.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--catalog", type=Path, default=None)
    lookup = parser.add_mutually_exclusive_group(required=True)
    lookup.add_argument("--frame-uid")
    lookup.add_argument("--query")
    lookup.add_argument("--object-class")
    parser.add_argument("--source-type", choices=("ocr", "asr"), default=None)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--level", choices=("L0", "L1"), default=None)
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    paths = PipelinePaths.from_root(args.root)
    destination = (args.catalog or paths.serving / "catalog.sqlite").resolve()
    if not destination.is_file():
        raise FileNotFoundError(destination)
    connection = sqlite3.connect(f"{destination.as_uri()}?mode=ro", uri=True)
    try:
        if args.frame_uid:
            result: Any = frame_bundle(connection, args.frame_uid)
        elif args.query:
            result = search_text(
                connection,
                args.query,
                source_type=args.source_type,
                video_id=args.video_id,
                level=args.level,
                limit=args.limit,
            )
        else:
            result = search_objects(
                connection,
                args.object_class,
                min_score=args.min_score,
                video_id=args.video_id,
                limit=args.limit,
            )
    finally:
        connection.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
