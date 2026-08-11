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
from aic_pipeline.serving.milvus import ingest_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upsert serving-catalog keyframe embeddings into Milvus."
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--uri", default=os.getenv("MILVUS_URI", "http://localhost:19530")
    )
    parser.add_argument("--token", default=os.getenv("MILVUS_TOKEN"))
    parser.add_argument("--collection", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--drop-existing", action="store_true")
    args = parser.parse_args()

    paths = PipelinePaths.from_root(args.root)
    catalog = args.catalog or paths.serving / "catalog.sqlite"
    report = ingest_embeddings(
        catalog,
        args.root,
        model=args.model,
        version=args.version,
        uri=args.uri,
        token=args.token,
        collection_name=args.collection,
        batch_size=args.batch_size,
        drop_existing=args.drop_existing,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
