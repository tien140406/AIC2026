from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from aic_pipeline.asr.alignment import ShotInterval, align_segments
from aic_pipeline.asr.models import (
    ProviderResult,
    TranscriptSegment,
    frame_to_segments,
    segments_to_frame,
)
from aic_pipeline.asr.normalization import (
    build_semantic_chunks,
    evaluate_transcript_quality,
    normalize_segments,
    normalize_transcript_text,
)
from aic_pipeline.asr.pipeline import (
    TranscriptPipeline,
    TranscriptPipelineConfig,
)
from aic_pipeline.asr.providers import (
    ExistingTranscriptProvider,
    FasterWhisperTranscriptProvider,
    TranscriptAcquisitionError,
    TranscriptContext,
    TranscriptSourceRouter,
    YouTubeTranscriptProvider,
)
from aic_pipeline.common.paths import PipelinePaths


def segment(
    text: str = "This is a sufficiently long transcript segment for quality checks.",
    *,
    start: float = 1.0,
    end: float = 3.0,
    segment_id: str = "V:L0:000000",
) -> TranscriptSegment:
    return TranscriptSegment(
        segment_id=segment_id,
        video_id="V",
        start=start,
        end=end,
        raw_text=text,
        source="test",
        source_type="caption",
        provider="test",
        language="en",
    )


def provider_result(name: str, text: str | None = None) -> ProviderResult:
    item = segment(text or f"A valid transcript supplied by the {name} test provider.")
    return ProviderResult(provider=name, segments=(item,), language="en")


class StubProvider:
    def __init__(self, name: str, result: ProviderResult | Exception) -> None:
        self.name = name
        self.result = result
        self.calls = 0

    def transcribe(self, context: TranscriptContext) -> ProviderResult:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def context(tmp_path: Path) -> TranscriptContext:
    return TranscriptContext(
        video_id="V",
        video_path=tmp_path / "V.mp4",
        audio_path=tmp_path / "V.wav",
        duration=10.0,
        language="en",
        youtube_id="youtube123",
    )


def test_provider_routing_accepts_first_usable_source(tmp_path: Path) -> None:
    local = StubProvider("local", provider_result("local"))
    youtube = StubProvider("youtube", provider_result("youtube"))
    whisper = StubProvider("whisper", provider_result("whisper"))
    router = TranscriptSourceRouter(local=local, youtube=youtube, whisper=whisper)

    result = router.acquire(context(tmp_path), mode="auto")

    assert result.provider == "local"
    assert (local.calls, youtube.calls, whisper.calls) == (1, 0, 0)


def test_youtube_failure_falls_back_to_whisper(tmp_path: Path) -> None:
    local = StubProvider("local", FileNotFoundError("missing"))
    youtube = StubProvider("youtube", RuntimeError("captions disabled"))
    whisper = StubProvider("whisper", provider_result("whisper"))
    router = TranscriptSourceRouter(local=local, youtube=youtube, whisper=whisper)

    result = router.acquire(context(tmp_path), mode="auto")

    assert result.provider == "whisper"
    assert [item["status"] for item in result.metadata["attempts"]] == [
        "error",
        "error",
        "accepted",
    ]


def test_quality_rejection_falls_back(tmp_path: Path) -> None:
    local = StubProvider("local", provider_result("local", "bad"))
    youtube = StubProvider("youtube", provider_result("youtube"))
    whisper = StubProvider("whisper", provider_result("whisper"))
    router = TranscriptSourceRouter(local=local, youtube=youtube, whisper=whisper)

    result = router.acquire(context(tmp_path), mode="auto")

    assert result.provider == "youtube"
    assert result.metadata["attempts"][0]["status"] == "rejected"
    assert whisper.calls == 0


def test_router_reports_all_failed_attempts(tmp_path: Path) -> None:
    failed = StubProvider("failed", RuntimeError("no transcript"))
    router = TranscriptSourceRouter(local=failed, youtube=failed, whisper=failed)
    with pytest.raises(TranscriptAcquisitionError) as caught:
        router.acquire(context(tmp_path), mode="auto")
    assert len(caught.value.attempts) == 3


def test_existing_srt_provider_preserves_absolute_timestamps(tmp_path: Path) -> None:
    srt = tmp_path / "V.srt"
    srt.write_text(
        "1\n00:00:12,200 --> 00:00:15,800\nXin chào thế giới.\n",
        encoding="utf-8",
    )
    source_context = context(tmp_path)
    source_context = TranscriptContext(
        **{
            **source_context.__dict__,
            "local_transcript_paths": (srt,),
        }
    )

    result = ExistingTranscriptProvider().transcribe(source_context)

    assert (result.segments[0].start, result.segments[0].end) == (12.2, 15.8)
    assert result.segments[0].source_type == "existing_caption"


def test_youtube_provider_uses_mock_client() -> None:
    class Fetched(list):
        language_code = "vi"
        is_generated = True

    fetched = Fetched(
        [SimpleNamespace(text="Phụ đề thử nghiệm.", start=2.5, duration=1.25)]
    )
    client = SimpleNamespace(fetch=lambda video_id, **kwargs: fetched)
    provider = YouTubeTranscriptProvider(client_factory=lambda: client)
    source_context = TranscriptContext(
        video_id="V",
        video_path=Path("V.mp4"),
        audio_path=Path("V.wav"),
        youtube_id="youtube123",
    )

    result = provider.transcribe(source_context)

    assert result.language == "vi"
    assert (result.segments[0].start, result.segments[0].end) == (2.5, 3.75)
    assert result.segments[0].source_type == "youtube_auto_caption"


def test_normalization_preserves_raw_text_and_timestamps() -> None:
    original = segment("  Xin\u200b  chào ,thế giới!  ", start=12.2, end=15.8)
    normalized = normalize_segments([original])[0]

    assert normalized.raw_text == original.raw_text
    assert normalized.clean_text == "Xin chào, thế giới!"
    assert (normalized.start, normalized.end) == (12.2, 15.8)
    assert normalize_transcript_text("A  B") == "A B"


def test_quality_gate_rejects_insufficient_transcript() -> None:
    quality = evaluate_transcript_quality(
        [segment("tiny")], video_duration=600.0, expected_language="en"
    )
    assert not quality.accepted
    assert "too_few_characters" in quality.reasons


def test_alignment_crosses_two_shots_and_chooses_greatest_overlap() -> None:
    shots = [
        ShotInterval(12, 42.0, 46.5, 1260, 1395),
        ShotInterval(13, 46.5, 51.2, 1395, 1536),
    ]
    original = segment(start=43.2, end=48.1)
    aligned = align_segments([original], shots, {12: (1300, 1380), 13: (1400, 1500)})[0]

    assert aligned.shot_ids == (12, 13)
    assert aligned.primary_shot == 12
    assert aligned.keyframe_ids == (1300, 1380, 1400, 1500)


def test_primary_shot_tie_uses_earliest_shot() -> None:
    shots = [
        ShotInterval(0, 0.0, 5.0, 0, 150),
        ShotInterval(1, 5.0, 10.0, 150, 300),
    ]
    aligned = align_segments([segment(start=4.0, end=6.0)], shots)[0]
    assert aligned.primary_shot == 0


def test_serialization_round_trip() -> None:
    original = segment().with_updates(
        clean_text="Clean text.",
        shot_ids=(2, 3),
        primary_shot=2,
        keyframe_ids=(100, 200),
        metadata={"nested": [1, 2]},
    )
    restored = frame_to_segments(segments_to_frame([original]))[0]

    assert restored == original


def test_semantic_chunks_preserve_l0_and_constituents() -> None:
    l0 = [
        segment(
            "First sentence.", start=0.0, end=5.0, segment_id="V:L0:000000"
        ).with_updates(clean_text="First sentence."),
        segment(
            "Second sentence.", start=5.1, end=10.0, segment_id="V:L0:000001"
        ).with_updates(clean_text="Second sentence."),
    ]
    chunks = build_semantic_chunks(l0, target_duration=20.0)

    assert [item.segment_id for item in l0] == ["V:L0:000000", "V:L0:000001"]
    assert len(chunks) == 1
    assert chunks[0].raw_text == "First sentence. Second sentence."
    assert chunks[0].metadata["constituent_segment_ids"] == [
        "V:L0:000000",
        "V:L0:000001",
    ]


class FakeAudioExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def extract(
        self, video_path: Path, destination: Path, *, force: bool = False
    ) -> Path:
        self.calls += 1
        return destination


class FakeWhisperModel:
    def __init__(self) -> None:
        self.options: list[dict[str, object]] = []

    def transcribe(self, audio_path: str, **options: object):
        self.options.append(options)
        segments = [
            SimpleNamespace(
                start=7.25,
                end=9.75,
                text=" Timestamped fake Whisper output for testing.",
                avg_logprob=-0.2,
                no_speech_prob=0.05,
            )
        ]
        info = SimpleNamespace(language="en", language_probability=0.99)
        return segments, info


def test_whisper_provider_reuses_injected_model_and_preserves_timestamps(
    tmp_path: Path,
) -> None:
    model = FakeWhisperModel()
    factory_calls = 0

    def factory(*args, **kwargs):
        nonlocal factory_calls
        factory_calls += 1
        return model

    audio = FakeAudioExtractor()
    provider = FasterWhisperTranscriptProvider(
        batch_size=1,
        model_factory=factory,
        audio_extractor=audio,
    )
    first = provider.transcribe(context(tmp_path))
    second = provider.transcribe(context(tmp_path))

    assert factory_calls == 1
    assert audio.calls == 2
    assert (first.segments[0].start, first.segments[0].end) == (7.25, 9.75)
    assert second.segments[0].raw_text.startswith("Timestamped")
    assert model.options[0]["vad_filter"] is True
    assert model.options[0]["without_timestamps"] is False


class CountingRouter:
    def __init__(self) -> None:
        self.calls = 0

    def acquire(self, context: TranscriptContext, **kwargs) -> ProviderResult:
        self.calls += 1
        return ProviderResult(
            provider="fake",
            segments=(
                TranscriptSegment(
                    segment_id=f"{context.video_id}:L0:000000",
                    video_id=context.video_id,
                    start=0.25,
                    end=1.75,
                    raw_text="A stable transcript used to verify pipeline resumption.",
                    source="fake",
                    source_type="test",
                    provider="fake",
                ),
            ),
        )


def test_pipeline_resume_skips_provider_when_fingerprint_matches(
    tmp_path: Path,
) -> None:
    paths = PipelinePaths.from_root(tmp_path)
    paths.ensure_output_dirs()
    video = tmp_path / "archive" / "video" / "V.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fake-video")
    metadata = tmp_path / "archive" / "media-info" / "V.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"title": "Test"}), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "video_id": "V",
                "video_path": str(video),
                "fps": 10.0,
                "frame_count": 20,
                "duration_seconds": 2.0,
            }
        ]
    ).to_parquet(paths.manifests / "videos.parquet", index=False)
    (paths.shots / "V.json").write_text(
        json.dumps({"ranges_end_exclusive": True, "ranges": [[0, 10], [10, 20]]}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"video_id": "V", "shot_id": 0, "frame_id": 5},
            {"video_id": "V", "shot_id": 1, "frame_id": 15},
        ]
    ).to_parquet(paths.manifests / "provided_keyframes.parquet", index=False)
    router = CountingRouter()
    pipeline = TranscriptPipeline(
        paths,
        TranscriptPipelineConfig(),
        router=router,  # type: ignore[arg-type]
    )

    first = pipeline.process_video("V")
    second = pipeline.process_video("V")

    assert first["status"] == "completed"
    assert second["status"] == "skipped"
    assert router.calls == 1
    assert pd.read_parquet(paths.transcript_dir("V") / "l0.parquet").iloc[0][
        "keyframe_ids"
    ].tolist() == [5, 15]
