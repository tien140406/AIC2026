from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aic_pipeline.objects.filtering import (
    DEFAULT_DUPLICATE_IOU,
    DEFAULT_MAX_DETECTIONS,
    DEFAULT_MIN_AREA_RATIO,
    DEFAULT_SCORE_THRESHOLD,
    write_summary_tables,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter and aggregate raw Co-DETR final-keyframe detections."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD
    )
    parser.add_argument("--min-area-ratio", type=float, default=DEFAULT_MIN_AREA_RATIO)
    parser.add_argument("--duplicate-iou", type=float, default=DEFAULT_DUPLICATE_IOU)
    parser.add_argument("--max-detections", type=int, default=DEFAULT_MAX_DETECTIONS)
    parser.add_argument("--run-signature", default=None)
    args = parser.parse_args()

    _, _, report = write_summary_tables(
        args.output_dir,
        score_threshold=args.score_threshold,
        min_area_ratio=args.min_area_ratio,
        duplicate_iou=args.duplicate_iou,
        max_detections=args.max_detections,
        run_signature=args.run_signature,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
