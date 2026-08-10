from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aic_pipeline.paths import PipelinePaths
from aic_pipeline.transcript_index import build_transcript_index, search_transcripts


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
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    paths = PipelinePaths.from_root(args.root)
    destination = args.output or paths.indexes / "transcripts.sqlite"
    levels = tuple(args.level) if args.level else ("L0", "L1")
    report = build_transcript_index(
        paths.transcripts, destination, include_levels=levels
    )
    if args.query:
        connection = sqlite3.connect(destination)
        try:
            report["query"] = args.query
            report["results"] = search_transcripts(
                connection, args.query, limit=args.limit
            )
        finally:
            connection.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
