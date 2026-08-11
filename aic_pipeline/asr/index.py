from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from ..common.identity import frame_uid as keyframe_uid
from .alignment import ShotInterval
from .models import TranscriptSegment, frame_to_segments

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
CREATE TABLE IF NOT EXISTS segment_keyframes (
    segment_id TEXT NOT NULL,
    frame_uid TEXT NOT NULL,
    video_id TEXT NOT NULL,
    frame_id INTEGER NOT NULL,
    PRIMARY KEY (segment_id, frame_uid),
    FOREIGN KEY (segment_id) REFERENCES segments(segment_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_segment_keyframes_lookup
    ON segment_keyframes(video_id, frame_id, segment_id);
CREATE INDEX IF NOT EXISTS idx_segment_keyframes_uid
    ON segment_keyframes(frame_uid, segment_id);
CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    segment_id UNINDEXED,
    video_id UNINDEXED,
    level UNINDEXED,
    clean_text,
    raw_text,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""


def _replace_keyframe_links(
    connection: sqlite3.Connection,
    *,
    segment_id: str,
    video_id: str,
    frame_ids: Iterable[int],
) -> int:
    video_id = str(video_id).strip()
    if not video_id:
        raise ValueError("video_id must not be empty")
    connection.execute(
        "DELETE FROM segment_keyframes WHERE segment_id = ?", (segment_id,)
    )
    unique_frame_ids = sorted({int(frame_id) for frame_id in frame_ids})
    rows = [
        (segment_id, keyframe_uid(video_id, frame_id), video_id, frame_id)
        for frame_id in unique_frame_ids
    ]
    connection.executemany(
        """
        INSERT INTO segment_keyframes(segment_id, frame_uid, video_id, frame_id)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def _backfill_keyframe_links(connection: sqlite3.Connection) -> int:
    rows = connection.execute(
        """
        SELECT segments.segment_id, segments.video_id, segments.keyframe_ids_json
        FROM segments
        WHERE NOT EXISTS (
            SELECT 1
            FROM segment_keyframes
            WHERE segment_keyframes.segment_id = segments.segment_id
        )
        """
    ).fetchall()
    inserted = 0
    for segment_id, video_id, encoded_ids in rows:
        inserted += _replace_keyframe_links(
            connection,
            segment_id=str(segment_id),
            video_id=str(video_id),
            frame_ids=json.loads(encoded_ids),
        )
    return inserted


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
    _replace_keyframe_links(
        connection,
        segment_id=segment.segment_id,
        video_id=segment.video_id,
        frame_ids=segment.keyframe_ids,
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
    _backfill_keyframe_links(connection)
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
                "DELETE FROM segment_keyframes WHERE video_id = ?", (replace_video,)
            )
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


def find_transcript_segments_by_frame_uid(
    connection: sqlite3.Connection,
    frame_uid: str,
    *,
    limit: int = 20,
    level: str | None = None,
) -> list[dict[str, object]]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    frame_uid = str(frame_uid).strip()
    if not frame_uid:
        raise ValueError("frame_uid must not be empty")
    clauses = ["links.frame_uid = ?"]
    parameters: list[object] = [frame_uid]
    if level is not None:
        if level not in {"L0", "L1"}:
            raise ValueError("level must be L0 or L1")
        clauses.append("segments.level = ?")
        parameters.append(level)
    parameters.append(limit)
    rows = connection.execute(
        f"""
        SELECT segments.segment_id, segments.video_id, segments.level,
               segments.start, segments.end, segments.clean_text, segments.raw_text,
               segments.language, segments.primary_shot, segments.shot_ids_json,
               segments.keyframe_ids_json, segments.provider, segments.model,
               links.frame_uid, links.frame_id
        FROM segment_keyframes AS links
        JOIN segments ON segments.segment_id = links.segment_id
        WHERE {" AND ".join(clauses)}
        ORDER BY segments.start, segments.end, segments.level, segments.segment_id
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
            "frame_uid": row[13],
            "frame_id": row[14],
        }
        for row in rows
    ]


def find_transcript_segments_by_keyframe(
    connection: sqlite3.Connection,
    video_id: str,
    frame_id: int,
    *,
    limit: int = 20,
    level: str | None = None,
) -> list[dict[str, object]]:
    return find_transcript_segments_by_frame_uid(
        connection,
        keyframe_uid(video_id, frame_id),
        limit=limit,
        level=level,
    )


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
        keyframe_link_count = int(
            connection.execute("SELECT COUNT(*) FROM segment_keyframes").fetchone()[0]
        )
        connection.commit()
    finally:
        connection.close()
    temporary.replace(destination)
    return {
        "destination": str(destination),
        "transcript_root": str(transcript_root),
        "videos": video_count,
        "segments": segment_count,
        "keyframe_links": keyframe_link_count,
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
