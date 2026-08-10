from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aic_pipeline.asr.pipeline import (
    TranscriptPipeline,
    TranscriptPipelineConfig,
    select_video_ids,
)
from aic_pipeline.common.paths import PipelinePaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract, normalize, align, and persist video transcripts."
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--all", action="store_true", dest="process_all")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument(
        "--provider",
        choices=("auto", "local", "youtube", "whisper"),
        default="auto",
    )
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--language", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--skip-youtube", action="store_true")
    parser.add_argument("--skip-vad", action="store_true")
    parser.add_argument("--no-semantic-chunks", action="store_true")
    parser.add_argument(
        "--keyframe-encoder-id", default="timm_beit3_base_patch16_224_pt"
    )
    parser.add_argument("--keyframe-manifest", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-audio", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from dotenv import load_dotenv

        load_dotenv(args.root / ".env")
    except ImportError:
        pass

    paths = PipelinePaths.from_root(args.root)
    video_ids = select_video_ids(
        paths,
        video_ids=args.video_id,
        process_all=args.process_all,
        max_videos=args.max_videos,
    )
    config = TranscriptPipelineConfig(
        provider=args.provider,
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        youtube_enabled=not args.skip_youtube,
        vad_enabled=not args.skip_vad,
        semantic_chunking_enabled=not args.no_semantic_chunks,
        keyframe_encoder_id=args.keyframe_encoder_id,
        keyframe_manifest=args.keyframe_manifest,
        force_audio=args.force_audio,
    )
    pipeline = TranscriptPipeline(paths, config)
    failures = 0
    for index, video_id in enumerate(video_ids, start=1):
        try:
            report = pipeline.process_video(
                video_id, force=args.force, dry_run=args.dry_run
            )
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        except Exception as error:  # noqa: BLE001 - continue the requested batch.
            failures += 1
            print(
                json.dumps(
                    {
                        "video_id": video_id,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                file=sys.stderr,
            )
        print(f"Processed {index}/{len(video_ids)} videos", file=sys.stderr, flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
