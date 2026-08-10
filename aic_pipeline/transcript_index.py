from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from .transcript_alignment import ShotInterval
from .transcript_models import TranscriptSegment, frame_to_segments

SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    segment_id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    level TEXT NOT NULL,
    start REAL NOT NULL,
    end REAL NOT NULL,
    clean_text TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    language TEXT,
    source TEXT,
    source_type TEXT,
    provider TEXT,
    model TEXT,
    shot_ids_json TEXT NOT NULL,
    primary_shot INTEGER,
    keyframe_ids_json TEXT NOT NULL,
    youtube_id TEXT,
    quality_score REAL,
    metadata_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_segments_video_time
    ON segments(video_id, start, end);
CREATE INDEX IF NOT EXISTS idx_segments_primary_shot
    ON segments(video_id, primary_shot);
CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    segment_id UNINDEXED,
    video_id UNINDEXED,
    level UNINDEXED,
    clean_text,
    raw_text,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""


def _insert_record(connection: sqlite3.Connection, segment: TranscriptSegment) -> None:
    record = segment.to_record()
    connection.execute(
        """
        INSERT OR REPLACE INTO segments (
            segment_id, video_id, level, start, end, clean_text, raw_text,
            language, source, source_type, provider, model, shot_ids_json,
            primary_shot, keyframe_ids_json, youtube_id, quality_score, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record["segment_id"],
            record["video_id"],
            record["level"],
            record["start"],
            record["end"],
            record["clean_text"] or record["raw_text"],
            record["raw_text"],
            record["language"],
            record["source"],
            record["source_type"],
            record["provider"],
            record["model"],
            json.dumps(record["shot_ids"]),
            record["primary_shot"],
            json.dumps(record["keyframe_ids"]),
            record["youtube_id"],
            record["quality_score"],
            record["metadata_json"],
        ),
    )
    connection.execute(
        "DELETE FROM segments_fts WHERE segment_id = ?",
        (record["segment_id"],),
    )
    connection.execute(
        "INSERT INTO segments_fts(segment_id, video_id, level, clean_text, raw_text)"
        " VALUES (?, ?, ?, ?, ?)",
        (
            record["segment_id"],
            record["video_id"],
            record["level"],
            record["clean_text"] or record["raw_text"],
            record["raw_text"],
        ),
    )


def initialize_transcript_index(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA)
    connection.commit()


def upsert_transcript_segments(
    connection: sqlite3.Connection,
    segments: Iterable[TranscriptSegment],
    *,
    replace_video: str | None = None,
) -> int:
    materialized = list(segments)
    with connection:
        if replace_video is not None:
            old_ids = [
                row[0]
                for row in connection.execute(
                    "SELECT segment_id FROM segments WHERE video_id = ?",
                    (replace_video,),
                )
            ]
            connection.execute(
                "DELETE FROM segments WHERE video_id = ?", (replace_video,)
            )
            for segment_id in old_ids:
                connection.execute(
                    "DELETE FROM segments_fts WHERE segment_id = ?", (segment_id,)
                )
        for segment in materialized:
            _insert_record(connection, segment)
    return len(materialized)


def search_transcripts(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int = 20,
    level: str | None = None,
    video_id: str | None = None,
) -> list[dict[str, object]]:
    if not query.strip():
        return []
    clauses = ["segments_fts MATCH ?"]
    parameters: list[object] = [query]
    if level:
        clauses.append("segments.level = ?")
        parameters.append(level)
    if video_id:
        clauses.append("segments.video_id = ?")
        parameters.append(video_id)
    parameters.append(limit)
    rows = connection.execute(
        f"""
        SELECT segments.segment_id, segments.video_id, segments.level,
               segments.start, segments.end, segments.clean_text, segments.raw_text,
               segments.language, segments.primary_shot, segments.shot_ids_json,
               segments.keyframe_ids_json, segments.provider, segments.model,
               segments_fts.rank
        FROM segments_fts
        JOIN segments ON segments.segment_id = segments_fts.segment_id
        WHERE {" AND ".join(clauses)}
        ORDER BY segments_fts.rank
        LIMIT ?
        """,
        parameters,
    ).fetchall()
    return [
        {
            "segment_id": row[0],
            "video_id": row[1],
            "level": row[2],
            "start": row[3],
            "end": row[4],
            "clean_text": row[5],
            "raw_text": row[6],
            "language": row[7],
            "primary_shot": row[8],
            "shot_ids": json.loads(row[9]),
            "keyframe_ids": json.loads(row[10]),
            "provider": row[11],
            "model": row[12],
            "score": -float(row[13]),
        }
        for row in rows
    ]


def build_transcript_index(
    transcript_root: Path,
    destination: Path,
    *,
    include_levels: Sequence[str] = ("L0", "L1"),
) -> dict[str, object]:
    transcript_root = transcript_root.expanduser().resolve()
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not transcript_root.is_dir():
        raise FileNotFoundError(transcript_root)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        initialize_transcript_index(connection)
        video_count = 0
        segment_count = 0
        for video_dir in sorted(
            path for path in transcript_root.iterdir() if path.is_dir()
        ):
            segments: list[TranscriptSegment] = []
            for level in include_levels:
                path = video_dir / f"{level.lower()}.parquet"
                if path.is_file():
                    segments.extend(frame_to_segments(pd.read_parquet(path)))
            if segments:
                upsert_transcript_segments(
                    connection, segments, replace_video=video_dir.name
                )
                video_count += 1
                segment_count += len(segments)
        connection.execute("INSERT INTO segments_fts(segments_fts) VALUES('optimize')")
        connection.commit()
    finally:
        connection.close()
    temporary.replace(destination)
    return {
        "destination": str(destination),
        "transcript_root": str(transcript_root),
        "videos": video_count,
        "segments": segment_count,
        "levels": list(include_levels),
    }


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[str]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    if k <= 0:
        raise ValueError("k must be positive")
    materialized = list(rankings)
    if weights is not None and len(weights) != len(materialized):
        raise ValueError("weights must match rankings")
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking_index, ranking in enumerate(materialized):
        weight = float(weights[ranking_index]) if weights is not None else 1.0
        seen: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            item = str(item)
            if item in seen:
                continue
            seen.add(item)
            scores[item] += weight / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def expand_temporal_neighbors(
    primary_shot: int,
    shots: Sequence[ShotInterval],
    *,
    neighbor_count: int = 1,
    time_radius: float | None = None,
) -> list[int]:
    if neighbor_count < 0 or (time_radius is not None and time_radius < 0):
        raise ValueError("Temporal expansion limits must be non-negative")
    ordered = sorted(shots, key=lambda item: item.shot_id)
    positions = {shot.shot_id: index for index, shot in enumerate(ordered)}
    if primary_shot not in positions:
        return []
    if time_radius is not None:
        center = ordered[positions[primary_shot]]
        start = center.start - time_radius
        end = center.end + time_radius
        return [
            shot.shot_id for shot in ordered if shot.end > start and shot.start < end
        ]
    position = positions[primary_shot]
    left = max(0, position - neighbor_count)
    right = min(len(ordered), position + neighbor_count + 1)
    return [shot.shot_id for shot in ordered[left:right]]
