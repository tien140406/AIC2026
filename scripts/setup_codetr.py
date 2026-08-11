from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import gdown

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aic_pipeline.objects.codetr import (
    DEFAULT_CHECKPOINT_GDRIVE_ID,
    DEFAULT_CHECKPOINT_NAME,
    DEFAULT_CODE_REVISION,
)

DEFAULT_REPO_URL = "https://github.com/Sense-X/Co-DETR.git"


def ensure_repo(destination: Path, revision: str) -> None:
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "init", str(destination)],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(destination),
                "remote",
                "add",
                "origin",
                DEFAULT_REPO_URL,
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(destination),
                "fetch",
                "--depth",
                "1",
                "origin",
                revision,
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"],
            check=True,
        )
    current = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    if current != revision:
        raise RuntimeError(
            f"Unexpected Co-DETR revision {current}; expected {revision}. "
            "Remove only third_party/Co-DETR and rerun setup if it is disposable."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the pinned Co-DETR runtime assets."
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=PROJECT_ROOT / "third_party" / "Co-DETR",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "codetr",
    )
    parser.add_argument("--skip-checkpoint", action="store_true")
    args = parser.parse_args()

    ensure_repo(args.repo_dir, DEFAULT_CODE_REVISION)
    print(f"Co-DETR source ready: {args.repo_dir}")
    if args.skip_checkpoint:
        return

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / DEFAULT_CHECKPOINT_NAME
    if checkpoint_path.exists():
        print(f"Checkpoint already exists: {checkpoint_path}")
        return
    downloaded = gdown.download(
        id=DEFAULT_CHECKPOINT_GDRIVE_ID,
        output=str(checkpoint_path),
        quiet=False,
        resume=True,
    )
    if not downloaded or not checkpoint_path.is_file():
        raise RuntimeError("Co-DETR checkpoint download did not produce an output file")
    print(f"Checkpoint ready: {checkpoint_path}")


if __name__ == "__main__":
    main()
