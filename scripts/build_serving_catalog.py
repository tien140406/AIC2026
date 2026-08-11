from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aic_pipeline.common.paths import PipelinePaths
from aic_pipeline.serving.catalog import (
    DEFAULT_SELECTION_ENCODER_ID,
    build_serving_catalog,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the global query-serving catalog from pipeline artifacts."
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--selection-encoder-id", default=DEFAULT_SELECTION_ENCODER_ID)
    args = parser.parse_args()

    paths = PipelinePaths.from_root(args.root)
    destination = args.output or paths.serving / "catalog.sqlite"
    report = build_serving_catalog(
        paths,
        destination,
        selection_encoder_id=args.selection_encoder_id,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
