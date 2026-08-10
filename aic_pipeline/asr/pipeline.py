from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..common.io import atomic_write_json, atomic_write_parquet
from ..common.paths import PipelinePaths
from .alignment import (
    align_segments,
    load_keyframe_mapping,
    load_shot_intervals,
)
from .models import ProviderResult, segments_to_frame, utc_now_iso
from .normalization import (
    build_semantic_chunks,
    normalize_segments,
)
from .providers import (
    ExistingTranscriptProvider,
    FasterWhisperTranscriptProvider,
    TranscriptContext,
    TranscriptSourceRouter,
    YouTubeTranscriptProvider,
    extract_youtube_id,
)

ASR_PIPELINE_VERSION = "1.0.0"


@dataclass(frozen=True)
class TranscriptPipelineConfig:
    provider: str = "auto"
    model: str = "large-v3"
    device: str = "auto"
    compute_type: str = "int8_float16"
    language: str | None = None
    batch_size: int = 4
    beam_size: int = 5
    youtube_enabled: bool = True
    vad_enabled: bool = True
    semantic_chunking_enabled: bool = True
    keyframe_encoder_id: str = "timm_beit3_base_patch16_224_pt"
    keyframe_manifest: Path | None = None
    force_audio: bool = False

    def __post_init__(self) -> None:
        if self.provider not in {"auto", "local", "youtube", "whisper"}:
            raise ValueError(f"Unsupported provider: {self.provider}")
        if min(self.batch_size, self.beam_size) <= 0:
            raise ValueError("Batch size and beam size must be positive")

    def resolved_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["keyframe_manifest"] = (
            str(self.keyframe_manifest.expanduser().resolve())
            if self.keyframe_manifest
            else None
        )
        return values

    def fingerprint_dict(self) -> dict[str, Any]:
        values = self.resolved_dict()
        values.pop("force_audio", None)
        return values


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def _file_signature(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return {"path": str(resolved), "exists": False}
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TranscriptPipeline:
    def __init__(
        self,
        paths: PipelinePaths,
        config: TranscriptPipelineConfig | None = None,
        *,
        router: TranscriptSourceRouter | None = None,
    ) -> None:
        self.paths = paths
        self.config = config or TranscriptPipelineConfig()
        if router is None:
            whisper = FasterWhisperTranscriptProvider(
                model_name=self.config.model,
                device=self.config.device,
                compute_type=self.config.compute_type,
                batch_size=self.config.batch_size,
                beam_size=self.config.beam_size,
                vad_enabled=self.config.vad_enabled,
                force_audio=self.config.force_audio,
            )
            router = TranscriptSourceRouter(
                local=ExistingTranscriptProvider(),
                youtube=YouTubeTranscriptProvider(),
                whisper=whisper,
            )
        self.router = router

    def _video_row(self, video_id: str) -> dict[str, Any]:
        manifest_path = self.paths.manifests / "videos.parquet"
        videos = pd.read_parquet(manifest_path)
        match = videos.loc[videos["video_id"] == video_id]
        if len(match) != 1:
            raise ValueError(
                f"Expected one video manifest row for {video_id}, got {len(match)}"
            )
        record = match.iloc[0].to_dict()
        return {
            key: value.item() if getattr(value, "ndim", None) == 0 else value
            for key, value in record.items()
        }

    def _keyframe_path(self, video_id: str) -> Path | None:
        if self.config.keyframe_manifest is not None:
            return self.config.keyframe_manifest.expanduser().resolve()
        selected = (
            self.paths.keyframes
            / self.config.keyframe_encoder_id
            / f"{video_id}.parquet"
        )
        if selected.is_file():
            return selected
        provided = self.paths.manifests / "provided_keyframes.parquet"
        return provided if provided.is_file() else None

    def _inputs(self, video_id: str) -> dict[str, Any]:
        video = self._video_row(video_id)
        video_path = Path(str(video["video_path"])).expanduser().resolve()
        metadata_path = self.paths.archive / "media-info" / f"{video_id}.json"
        shot_path = self.paths.shots / f"{video_id}.json"
        keyframe_path = self._keyframe_path(video_id)
        local_paths = (
            self.paths.archive / "transcripts" / f"{video_id}.json",
            self.paths.archive / "transcripts" / f"{video_id}.srt",
        )
        metadata = _read_json(metadata_path)
        signatures = {
            "video": _file_signature(video_path),
            "metadata": _file_signature(metadata_path),
            "shots": _file_signature(shot_path),
            "keyframes": _file_signature(keyframe_path),
            "local_transcripts": [_file_signature(path) for path in local_paths],
        }
        fingerprint_payload = {
            "pipeline_version": ASR_PIPELINE_VERSION,
            "config": self.config.fingerprint_dict(),
            "video_manifest_row": {
                "video_id": video_id,
                "video_path": str(video_path),
                "fps": float(video["fps"]),
                "frame_count": int(video["frame_count"]),
                "duration_seconds": float(video["duration_seconds"]),
            },
            "signatures": signatures,
        }
        return {
            "video": video,
            "video_path": video_path,
            "metadata_path": metadata_path,
            "metadata": metadata,
            "shot_path": shot_path,
            "keyframe_path": keyframe_path,
            "local_paths": local_paths,
            "signatures": signatures,
            "fingerprint": _fingerprint(fingerprint_payload),
            "fingerprint_payload": fingerprint_payload,
        }

    @staticmethod
    def _completed_manifest(
        manifest_path: Path, fingerprint: str, l0_path: Path, l1_path: Path
    ) -> dict[str, Any] | None:
        if (
            not manifest_path.is_file()
            or not l0_path.is_file()
            or not l1_path.is_file()
        ):
            return None
        try:
            manifest = _read_json(manifest_path)
        except (OSError, json.JSONDecodeError):
            return None
        if (
            manifest.get("status") != "completed"
            or manifest.get("fingerprint") != fingerprint
        ):
            return None
        return manifest

    def process_video(
        self,
        video_id: str,
        *,
        force: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        started = time.time()
        inputs = self._inputs(video_id)
        output_dir = self.paths.transcript_dir(video_id)
        l0_path = output_dir / "l0.parquet"
        l1_path = output_dir / "l1.parquet"
        manifest_path = output_dir / "manifest.json"
        fingerprint = inputs["fingerprint"]

        existing = (
            None
            if force or self.config.force_audio
            else self._completed_manifest(manifest_path, fingerprint, l0_path, l1_path)
        )
        if existing is not None:
            return {
                "video_id": video_id,
                "status": "skipped",
                "reason": "fingerprint_match",
                "fingerprint": fingerprint,
                "manifest": str(manifest_path),
                "outputs": existing.get("outputs", {}),
            }

        if dry_run:
            return {
                "video_id": video_id,
                "status": "dry_run",
                "fingerprint": fingerprint,
                "provider": self.config.provider,
                "youtube_id": extract_youtube_id(inputs["metadata"]),
                "available_local_transcripts": [
                    str(path) for path in inputs["local_paths"] if path.is_file()
                ],
                "inputs": inputs["signatures"],
                "outputs": {
                    "l0": str(l0_path),
                    "l1": str(l1_path),
                    "manifest": str(manifest_path),
                },
            }

        video = inputs["video"]
        if not inputs["video_path"].is_file():
            raise FileNotFoundError(inputs["video_path"])
        if not inputs["shot_path"].is_file():
            raise FileNotFoundError(
                f"Shot manifest is required for transcript alignment: {inputs['shot_path']}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        context = TranscriptContext(
            video_id=video_id,
            video_path=inputs["video_path"],
            audio_path=self.paths.asr_audio_path(video_id),
            duration=float(video["duration_seconds"]),
            language=self.config.language,
            youtube_id=extract_youtube_id(inputs["metadata"]),
            metadata=inputs["metadata"],
            local_transcript_paths=inputs["local_paths"],
        )

        try:
            provider_result: ProviderResult = self.router.acquire(
                context,
                mode=self.config.provider,
                skip_youtube=not self.config.youtube_enabled,
            )
            l0 = normalize_segments(provider_result.segments)

            shots = load_shot_intervals(inputs["shot_path"], float(video["fps"]))
            keyframes = load_keyframe_mapping(
                inputs["keyframe_path"], video_id=video_id, shots=shots
            )
            l0 = align_segments(l0, shots, keyframes)
            l1 = (
                align_segments(build_semantic_chunks(l0), shots, keyframes)
                if self.config.semantic_chunking_enabled
                else []
            )

            atomic_write_parquet(segments_to_frame(l0), l0_path)
            atomic_write_parquet(segments_to_frame(l1), l1_path)
            report = {
                "pipeline_version": ASR_PIPELINE_VERSION,
                "status": "completed",
                "video_id": video_id,
                "fingerprint": fingerprint,
                "config": self.config.resolved_dict(),
                "input_signatures": inputs["signatures"],
                "youtube_metadata": {
                    "youtube_id": context.youtube_id,
                    "title": inputs["metadata"].get("title"),
                    "description": inputs["metadata"].get("description"),
                    "duration": inputs["metadata"].get("length"),
                    "watch_url": inputs["metadata"].get("watch_url"),
                },
                "provider": provider_result.provider,
                "provider_model": provider_result.model,
                "provider_metadata": provider_result.metadata,
                "counts": {"l0": len(l0), "l1": len(l1)},
                "outputs": {
                    "l0": str(l0_path),
                    "l1": str(l1_path),
                    "manifest": str(manifest_path),
                    "audio": (
                        str(context.audio_path)
                        if context.audio_path.is_file()
                        else None
                    ),
                },
                "created_at": utc_now_iso(),
                "elapsed_seconds": round(time.time() - started, 3),
            }
            atomic_write_json(report, manifest_path)
            return report
        except Exception as error:
            failure = {
                "pipeline_version": ASR_PIPELINE_VERSION,
                "status": "failed",
                "video_id": video_id,
                "fingerprint": fingerprint,
                "config": self.config.resolved_dict(),
                "input_signatures": inputs["signatures"],
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at": utc_now_iso(),
                "elapsed_seconds": round(time.time() - started, 3),
            }
            attempts = getattr(error, "attempts", None)
            if attempts is not None:
                failure["provider_attempts"] = attempts
            atomic_write_json(failure, manifest_path)
            raise


def select_video_ids(
    paths: PipelinePaths,
    *,
    video_ids: Iterable[str] = (),
    process_all: bool = False,
    max_videos: int | None = None,
) -> list[str]:
    requested = list(dict.fromkeys(video_ids))
    if process_all:
        videos = pd.read_parquet(
            paths.manifests / "videos.parquet", columns=["video_id"]
        )
        requested = [str(value) for value in videos["video_id"]]
    if not requested:
        raise ValueError("Specify at least one --video-id or use --all")
    if max_videos is not None:
        if max_videos <= 0:
            raise ValueError("--max-videos must be positive")
        requested = requested[:max_videos]
    return requested
