from __future__ import annotations

import sqlite3
from pathlib import Path

from aic_pipeline.asr.alignment import ShotInterval
from aic_pipeline.asr.index import (
    build_transcript_index,
    expand_temporal_neighbors,
    initialize_transcript_index,
    reciprocal_rank_fusion,
    search_transcripts,
    upsert_transcript_segments,
)
from aic_pipeline.asr.models import TranscriptSegment, segments_to_frame


def indexed_segment() -> TranscriptSegment:
    return TranscriptSegment(
        segment_id="V:L0:000000",
        video_id="V",
        start=3.0,
        end=5.0,
        raw_text="The cook adds fresh tomato to the soup.",
        clean_text="The cook adds fresh tomato to the soup.",
        source="test",
        source_type="caption",
        provider="test",
        shot_ids=(4,),
        primary_shot=4,
        keyframe_ids=(120,),
    )


def test_sqlite_fts_query_returns_temporal_metadata() -> None:
    connection = sqlite3.connect(":memory:")
    initialize_transcript_index(connection)
    upsert_transcript_segments(connection, [indexed_segment()])

    results = search_transcripts(connection, "tomato")

    assert len(results) == 1
    assert results[0]["video_id"] == "V"
    assert results[0]["primary_shot"] == 4
    assert results[0]["keyframe_ids"] == [120]


def test_sqlite_fts_supports_diacritic_insensitive_vietnamese_query() -> None:
    connection = sqlite3.connect(":memory:")
    initialize_transcript_index(connection)
    vietnamese = indexed_segment().with_updates(
        segment_id="V:L0:000001",
        raw_text="Cho cà chua vào nồi.",
        clean_text="Cho cà chua vào nồi.",
    )
    upsert_transcript_segments(connection, [vietnamese])

    results = search_transcripts(connection, '"ca chua"')

    assert [item["segment_id"] for item in results] == ["V:L0:000001"]


def test_build_transcript_index_reads_parquet_levels_atomically(
    tmp_path: Path,
) -> None:
    video_dir = tmp_path / "transcripts" / "V"
    video_dir.mkdir(parents=True)
    segments_to_frame([indexed_segment()]).to_parquet(
        video_dir / "l0.parquet", index=False
    )
    segments_to_frame([]).to_parquet(video_dir / "l1.parquet", index=False)
    destination = tmp_path / "indexes" / "transcripts.sqlite"

    report = build_transcript_index(tmp_path / "transcripts", destination)

    assert report["segments"] == 1
    with sqlite3.connect(destination) as connection:
        assert search_transcripts(connection, "tomato")[0]["video_id"] == "V"


def test_rrf_combines_rankings_deterministically() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=10)
    assert fused[0][0] == "b"
    assert {item for item, _ in fused} == {"a", "b", "c"}


def test_temporal_neighbor_expansion_by_count_and_time() -> None:
    shots = [
        ShotInterval(0, 0.0, 5.0, 0, 50),
        ShotInterval(1, 5.0, 10.0, 50, 100),
        ShotInterval(2, 10.0, 15.0, 100, 150),
        ShotInterval(3, 15.0, 20.0, 150, 200),
    ]
    assert expand_temporal_neighbors(2, shots, neighbor_count=1) == [1, 2, 3]
    assert expand_temporal_neighbors(1, shots, time_radius=1.0) == [0, 1, 2]
