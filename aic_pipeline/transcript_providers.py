from __future__ import annotations

import json
import math
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .io import atomic_write_json
from .transcript_models import ProviderResult, TranscriptSegment
from .transcript_normalization import evaluate_transcript_quality


@dataclass(frozen=True)
class TranscriptContext:
    video_id: str
    video_path: Path
    audio_path: Path
    duration: float | None = None
    language: str | None = None
    youtube_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    local_transcript_paths: tuple[Path, ...] = ()


class TranscriptProvider(Protocol):
    name: str

    def transcribe(self, context: TranscriptContext) -> ProviderResult: ...


class TranscriptAcquisitionError(RuntimeError):
    def __init__(self, attempts: list[dict[str, Any]]) -> None:
        super().__init__("No transcript provider returned a usable transcript")
        self.attempts = attempts


def extract_youtube_id(metadata: dict[str, Any]) -> str | None:
    watch_url = str(metadata.get("watch_url") or metadata.get("url") or "")
    patterns = [
        r"[?&]v=([A-Za-z0-9_-]{6,})",
        r"youtu\.be/([A-Za-z0-9_-]{6,})",
        r"youtube\.com/(?:shorts|embed)/([A-Za-z0-9_-]{6,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, watch_url)
        if match:
            return match.group(1)
    value = metadata.get("youtube_id")
    return str(value) if value else None


def _seconds(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if ":" not in text:
        return float(text)
    parts = [float(part) for part in text.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60.0 + part
    return seconds


def _segment_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("segments", "transcript", "captions", "utterances", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = _segment_items(value)
            if nested:
                return nested
    return [payload] if any(key in payload for key in ("text", "raw_text")) else []


class ExistingTranscriptProvider:
    name = "local"

    def transcribe(self, context: TranscriptContext) -> ProviderResult:
        path = next(
            (item for item in context.local_transcript_paths if item.is_file()), None
        )
        if path is None:
            raise FileNotFoundError(f"No local transcript for {context.video_id}")
        if path.suffix.lower() == ".srt":
            entries = self._read_srt(path)
            language = context.language
        else:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            entries = _segment_items(payload)
            language = (
                str(payload.get("language"))
                if isinstance(payload, dict) and payload.get("language")
                else context.language
            )

        segments: list[TranscriptSegment] = []
        for item in entries:
            text = str(
                item.get("text") or item.get("raw_text") or item.get("content") or ""
            ).strip()
            if not text:
                continue
            timestamp = item.get("timestamp")
            start_value = next(
                (
                    item[key]
                    for key in ("start", "start_time", "startTime", "from")
                    if key in item
                ),
                timestamp[0]
                if isinstance(timestamp, (list, tuple)) and len(timestamp) == 2
                else None,
            )
            end_value = next(
                (
                    item[key]
                    for key in ("end", "end_time", "endTime", "to")
                    if key in item
                ),
                timestamp[1]
                if isinstance(timestamp, (list, tuple)) and len(timestamp) == 2
                else None,
            )
            if start_value is None:
                continue
            try:
                start = _seconds(start_value)
                if end_value is not None:
                    end = _seconds(end_value)
                elif item.get("duration") is not None:
                    end = start + _seconds(item["duration"])
                else:
                    continue
                confidence = (
                    float(item["confidence"])
                    if item.get("confidence") is not None
                    else None
                )
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            item_language = str(item.get("language") or language or "") or None
            segments.append(
                TranscriptSegment(
                    segment_id=f"{context.video_id}:L0:{len(segments):06d}",
                    video_id=context.video_id,
                    start=start,
                    end=end,
                    language=item_language,
                    source=str(path.resolve()),
                    source_type="existing_caption",
                    provider=self.name,
                    raw_text=text,
                    youtube_id=context.youtube_id,
                    confidence=confidence,
                    metadata={"local_format": path.suffix.lower().lstrip(".")},
                )
            )
        return ProviderResult(
            provider=self.name,
            segments=tuple(segments),
            language=language,
            metadata={"path": str(path.resolve())},
        )

    @staticmethod
    def _read_srt(path: Path) -> list[dict[str, Any]]:
        content = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
        entries: list[dict[str, Any]] = []
        timestamp = re.compile(
            r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d+)\s+-->\s+"
            r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d+)"
        )
        for block in re.split(r"\n\s*\n", content.strip()):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            match_index = next(
                (index for index, line in enumerate(lines) if timestamp.search(line)),
                None,
            )
            if match_index is None:
                continue
            match = timestamp.search(lines[match_index])
            if match is None:
                continue
            text = " ".join(lines[match_index + 1 :]).strip()
            if text:
                entries.append(
                    {
                        "start": match.group("start"),
                        "end": match.group("end"),
                        "text": re.sub(r"<[^>]+>", "", text),
                    }
                )
        return entries


class YouTubeTranscriptProvider:
    name = "youtube"

    def __init__(self, client_factory: Callable[[], Any] | None = None) -> None:
        self._client_factory = client_factory

    def _client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory()
        from youtube_transcript_api import YouTubeTranscriptApi

        return YouTubeTranscriptApi()

    def transcribe(self, context: TranscriptContext) -> ProviderResult:
        if not context.youtube_id:
            raise ValueError(f"No YouTube ID for {context.video_id}")
        client = self._client()
        languages = [context.language] if context.language else None
        if hasattr(client, "fetch"):
            fetched = (
                client.fetch(context.youtube_id, languages=languages)
                if languages
                else client.fetch(context.youtube_id)
            )
            entries = list(fetched)
            language = getattr(fetched, "language_code", None) or context.language
            generated = getattr(fetched, "is_generated", None)
        elif hasattr(client, "get_transcript"):
            entries = (
                client.get_transcript(context.youtube_id, languages=languages)
                if languages
                else client.get_transcript(context.youtube_id)
            )
            language = context.language
            generated = None
        else:
            raise TypeError("Unsupported youtube-transcript-api client")

        segments: list[TranscriptSegment] = []
        for entry in entries:
            if isinstance(entry, dict):
                text = str(entry.get("text") or "").strip()
                start = float(entry.get("start", 0.0))
                duration = float(entry.get("duration", 0.0))
            else:
                text = str(getattr(entry, "text", "")).strip()
                start = float(getattr(entry, "start", 0.0))
                duration = float(getattr(entry, "duration", 0.0))
            if not text or duration <= 0:
                continue
            segments.append(
                TranscriptSegment(
                    segment_id=f"{context.video_id}:L0:{len(segments):06d}",
                    video_id=context.video_id,
                    start=start,
                    end=start + duration,
                    language=language,
                    source=f"https://www.youtube.com/watch?v={context.youtube_id}",
                    source_type=(
                        "youtube_auto_caption" if generated else "youtube_caption"
                    ),
                    provider=self.name,
                    raw_text=text,
                    youtube_id=context.youtube_id,
                    metadata={"is_generated": generated},
                )
            )
        return ProviderResult(
            provider=self.name,
            segments=tuple(segments),
            language=language,
            metadata={"youtube_id": context.youtube_id, "is_generated": generated},
        )


class AudioExtractor:
    def __init__(self, ffmpeg: str = "ffmpeg") -> None:
        self.ffmpeg = ffmpeg

    def extract(
        self, video_path: Path, destination: Path, *, force: bool = False
    ) -> Path:
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        video_stat = video_path.stat()
        cache_manifest = destination.with_suffix(destination.suffix + ".json")
        expected_cache = {
            "video_path": str(video_path.resolve()),
            "video_size": video_stat.st_size,
            "video_mtime_ns": video_stat.st_mtime_ns,
            "audio_channels": 1,
            "sample_rate": 16000,
            "codec": "pcm_s16le",
        }
        if destination.is_file() and cache_manifest.is_file() and not force:
            try:
                cached = json.loads(cache_manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached = None
            if cached == expected_cache:
                return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f"{destination.stem}.tmp.wav")
        if temporary.exists():
            temporary.unlink()
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(temporary),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError(f"FFmpeg produced no audio: {temporary}")
            os.replace(temporary, destination)
            atomic_write_json(expected_cache, cache_manifest)
        except subprocess.CalledProcessError as error:
            temporary.unlink(missing_ok=True)
            message = (error.stderr or error.stdout or str(error)).strip()
            raise RuntimeError(f"FFmpeg audio extraction failed: {message}") from error
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return destination


class FasterWhisperTranscriptProvider:
    name = "whisper"

    def __init__(
        self,
        *,
        model_name: str = "large-v3",
        device: str = "auto",
        compute_type: str = "int8_float16",
        batch_size: int = 4,
        beam_size: int = 5,
        vad_enabled: bool = True,
        force_audio: bool = False,
        audio_extractor: AudioExtractor | None = None,
        model_factory: Callable[..., Any] | None = None,
        batched_pipeline_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        if batch_size <= 0 or beam_size <= 0:
            raise ValueError("batch_size and beam_size must be positive")
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.vad_enabled = vad_enabled
        self.force_audio = force_audio
        self.audio_extractor = audio_extractor or AudioExtractor()
        self.model_factory = model_factory
        self.batched_pipeline_factory = batched_pipeline_factory
        self._model: Any = None
        self._batched_pipeline: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            if self.model_factory is not None:
                self._model = self.model_factory(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            else:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                )
        return self._model

    def _get_transcriber(self) -> tuple[Any, bool]:
        model = self._get_model()
        if self.batch_size == 1:
            return model, False
        if self._batched_pipeline is None:
            if self.batched_pipeline_factory is not None:
                self._batched_pipeline = self.batched_pipeline_factory(model)
            elif self.model_factory is not None:
                # Injected test/dummy models generally expose transcribe directly.
                return model, False
            else:
                from faster_whisper import BatchedInferencePipeline

                self._batched_pipeline = BatchedInferencePipeline(model=model)
        return self._batched_pipeline, True

    def transcribe(self, context: TranscriptContext) -> ProviderResult:
        audio_path = self.audio_extractor.extract(
            context.video_path, context.audio_path, force=self.force_audio
        )
        transcriber, is_batched = self._get_transcriber()
        options: dict[str, Any] = {
            "beam_size": self.beam_size,
            "vad_filter": self.vad_enabled,
            "without_timestamps": False,
        }
        if context.language:
            options["language"] = context.language
        if is_batched:
            options["batch_size"] = self.batch_size
        raw_segments, info = transcriber.transcribe(str(audio_path), **options)
        detected_language = getattr(info, "language", None) or context.language
        segments: list[TranscriptSegment] = []
        for raw in raw_segments:
            text = str(getattr(raw, "text", "")).strip()
            start = float(raw.start)
            end = float(raw.end)
            if not text or end <= start:
                continue
            avg_logprob = getattr(raw, "avg_logprob", None)
            confidence = (
                max(0.0, min(1.0, math.exp(float(avg_logprob))))
                if avg_logprob is not None
                else None
            )
            segments.append(
                TranscriptSegment(
                    segment_id=f"{context.video_id}:L0:{len(segments):06d}",
                    video_id=context.video_id,
                    start=start,
                    end=end,
                    language=detected_language,
                    source=str(context.video_path.resolve()),
                    source_type="local_asr",
                    provider=self.name,
                    model=self.model_name,
                    raw_text=text,
                    confidence=confidence,
                    youtube_id=context.youtube_id,
                    metadata={
                        "no_speech_probability": getattr(raw, "no_speech_prob", None),
                        "vad_enabled": self.vad_enabled,
                    },
                )
            )
        return ProviderResult(
            provider=self.name,
            segments=tuple(segments),
            language=detected_language,
            model=self.model_name,
            metadata={
                "audio_path": str(audio_path),
                "batch_size": self.batch_size,
                "beam_size": self.beam_size,
                "vad_enabled": self.vad_enabled,
                "language_probability": getattr(info, "language_probability", None),
            },
        )


class TranscriptSourceRouter:
    def __init__(
        self,
        *,
        local: TranscriptProvider,
        youtube: TranscriptProvider,
        whisper: TranscriptProvider,
        quality_gate: Callable[..., Any] = evaluate_transcript_quality,
    ) -> None:
        self.providers = {
            "local": local,
            "youtube": youtube,
            "whisper": whisper,
        }
        self.quality_gate = quality_gate

    def acquire(
        self,
        context: TranscriptContext,
        *,
        mode: str = "auto",
        skip_youtube: bool = False,
    ) -> ProviderResult:
        if mode == "auto":
            route = ["local", "youtube", "whisper"]
        elif mode == "youtube":
            route = ["youtube", "whisper"]
        elif mode in {"local", "whisper"}:
            route = [mode]
        else:
            raise ValueError(f"Unsupported transcript provider mode: {mode}")
        if skip_youtube or not context.youtube_id:
            route = [name for name in route if name != "youtube"]

        attempts: list[dict[str, Any]] = []
        for name in route:
            provider = self.providers[name]
            try:
                result = provider.transcribe(context)
                quality = self.quality_gate(
                    result.segments,
                    video_duration=context.duration,
                    expected_language=context.language,
                )
                attempts.append(
                    {
                        "provider": name,
                        "status": "accepted" if quality.accepted else "rejected",
                        "segment_count": len(result.segments),
                        "quality_score": quality.score,
                        "quality_reasons": list(quality.reasons),
                    }
                )
                if not quality.accepted:
                    continue
                segments = tuple(
                    item.with_updates(quality_score=quality.score)
                    for item in result.segments
                )
                metadata = dict(result.metadata)
                metadata["attempts"] = attempts
                metadata["quality_score"] = quality.score
                return ProviderResult(
                    provider=result.provider,
                    segments=segments,
                    language=result.language,
                    model=result.model,
                    metadata=metadata,
                )
            except Exception as error:  # noqa: BLE001 - provider fallback is intentional.
                attempts.append(
                    {
                        "provider": name,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
        raise TranscriptAcquisitionError(attempts)
