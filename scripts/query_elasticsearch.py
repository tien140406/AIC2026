from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aic_pipeline.serving.elasticsearch import (
    DEFAULT_ASR_INDEX,
    DEFAULT_ELASTICSEARCH_URL,
    DEFAULT_FRAME_INDEX,
    create_client,
    search_text,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search OCR, object metadata, and ASR in Elasticsearch."
    )
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--url", default=os.getenv("ELASTICSEARCH_URL", DEFAULT_ELASTICSEARCH_URL)
    )
    parser.add_argument(
        "--frame-index",
        default=os.getenv("ELASTICSEARCH_FRAME_INDEX", DEFAULT_FRAME_INDEX),
    )
    parser.add_argument(
        "--asr-index",
        default=os.getenv("ELASTICSEARCH_ASR_INDEX", DEFAULT_ASR_INDEX),
    )
    parser.add_argument("--api-key", default=os.getenv("ELASTICSEARCH_API_KEY"))
    parser.add_argument("--username", default=os.getenv("ELASTICSEARCH_USERNAME"))
    parser.add_argument("--password", default=os.getenv("ELASTICSEARCH_PASSWORD"))
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--level", choices=("L0", "L1"), default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--insecure", action="store_true")
    args = parser.parse_args()

    client = create_client(
        args.url,
        api_key=args.api_key,
        username=args.username,
        password=args.password,
        verify_certs=not args.insecure,
    )
    result = search_text(
        client,
        args.query,
        frame_index=args.frame_index,
        asr_index=args.asr_index,
        video_id=args.video_id,
        level=args.level,
        limit=args.limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
