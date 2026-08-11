from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aic_pipeline.common.paths import PipelinePaths
from aic_pipeline.serving.artifacts import DEFAULT_SELECTION_ENCODER_ID
from aic_pipeline.serving.elasticsearch import (
    DEFAULT_ASR_INDEX,
    DEFAULT_ELASTICSEARCH_URL,
    DEFAULT_FRAME_INDEX,
    create_client,
    index_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index OCR, object metadata, and ASR artifacts in Elasticsearch."
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
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
    parser.add_argument("--selection-encoder-id", default=DEFAULT_SELECTION_ENCODER_ID)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--insecure", action="store_true")
    args = parser.parse_args()

    client = create_client(
        args.url,
        api_key=args.api_key,
        username=args.username,
        password=args.password,
        verify_certs=not args.insecure,
    )
    report = index_artifacts(
        PipelinePaths.from_root(args.root),
        client=client,
        frame_index=args.frame_index,
        asr_index=args.asr_index,
        selection_encoder_id=args.selection_encoder_id,
        batch_size=args.batch_size,
        recreate=args.recreate,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
