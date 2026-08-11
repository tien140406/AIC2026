from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from aic_pipeline.asr.alignment import ShotInterval
from aic_pipeline.asr.index import (
    build_transcript_index,
    expand_temporal_neighbors,
    find_transcript_segments_by_frame_uid,
    find_transcript_segments_by_keyframe,
    initialize_transcript_index,
    keyframe_uid,
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


def test_reverse_query_uses_materialized_keyframe_links() -> None:
    connection = sqlite3.connect(":memory:")
    initialize_transcript_index(connection)
    upsert_transcript_segments(
        connection,
        [
            indexed_segment(),
            indexed_segment().with_updates(
                segment_id="OTHER:L0:000000",
                video_id="OTHER",
                keyframe_ids=(120,),
            ),
        ],
    )

    results = find_transcript_segments_by_keyframe(connection, "V", 120)
    query_plan = connection.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT segments.segment_id
        FROM segment_keyframes AS links
        JOIN segments ON segments.segment_id = links.segment_id
        WHERE links.frame_uid = ?
        """,
        (keyframe_uid("V", 120),),
    ).fetchall()

    assert [item["segment_id"] for item in results] == ["V:L0:000000"]
    assert results[0]["frame_uid"] == keyframe_uid("V", 120)
    assert results[0]["frame_id"] == 120
    assert (
        find_transcript_segments_by_frame_uid(connection, keyframe_uid("V", 120))
        == results
    )
    assert any("idx_segment_keyframes_uid" in row[-1] for row in query_plan)


def test_upsert_replaces_stale_reverse_links() -> None:
    connection = sqlite3.connect(":memory:")
    initialize_transcript_index(connection)
    upsert_transcript_segments(connection, [indexed_segment()])
    updated = indexed_segment().with_updates(keyframe_ids=(130,))

    upsert_transcript_segments(connection, [updated])

    assert find_transcript_segments_by_keyframe(connection, "V", 120) == []
    assert [
        item["segment_id"]
        for item in find_transcript_segments_by_keyframe(connection, "V", 130)
    ] == ["V:L0:000000"]


def test_initialize_backfills_reverse_links_in_existing_index() -> None:
    connection = sqlite3.connect(":memory:")
    initialize_transcript_index(connection)
    upsert_transcript_segments(connection, [indexed_segment()])
    connection.execute("DELETE FROM segment_keyframes")
    connection.commit()

    initialize_transcript_index(connection)

    assert find_transcript_segments_by_keyframe(connection, "V", 120)


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
    assert report["keyframe_links"] == 1
    with sqlite3.connect(destination) as connection:
        assert search_transcripts(connection, "tomato")[0]["video_id"] == "V"
        assert find_transcript_segments_by_keyframe(connection, "V", 120)


def test_reverse_query_cli_reads_existing_index_without_rebuild(tmp_path: Path) -> None:
    video_dir = tmp_path / "transcripts" / "V"
    video_dir.mkdir(parents=True)
    segments_to_frame([indexed_segment()]).to_parquet(
        video_dir / "l0.parquet", index=False
    )
    destination = tmp_path / "transcripts.sqlite"
    build_transcript_index(tmp_path / "transcripts", destination)
    modified_before = destination.stat().st_mtime_ns

    completed = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "scripts/query_transcript_index.py"
            ),
            "--index",
            str(destination),
            "--frame-uid",
            keyframe_uid("V", 120),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["mode"] == "frame_uid"
    assert [item["segment_id"] for item in payload["results"]] == ["V:L0:000000"]
    assert destination.stat().st_mtime_ns == modified_before


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
