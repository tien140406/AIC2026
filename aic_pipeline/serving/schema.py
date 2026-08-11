from __future__ import annotations

import sqlite3

SCHEMA_VERSION = "aic_serving_v1"

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS catalog_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    group_name TEXT,
    video_path TEXT NOT NULL,
    fps REAL,
    frame_count INTEGER,
    duration_ms INTEGER,
    width INTEGER,
    height INTEGER,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS shots (
    video_id TEXT NOT NULL,
    shot_id INTEGER NOT NULL,
    start_frame INTEGER NOT NULL,
    end_frame_exclusive INTEGER NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    detector TEXT,
    source_path TEXT NOT NULL,
    PRIMARY KEY (video_id, shot_id),
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS keyframes (
    frame_uid TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    frame_id INTEGER NOT NULL,
    shot_id INTEGER,
    pts_ms INTEGER,
    image_path TEXT,
    image_sha256 TEXT,
    selection_encoder_id TEXT NOT NULL,
    selection_version TEXT NOT NULL,
    source_manifest TEXT NOT NULL,
    UNIQUE (video_id, frame_id),
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS frame_embeddings (
    frame_uid TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_version TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    dtype TEXT NOT NULL,
    vector_path TEXT NOT NULL,
    embedding_row INTEGER NOT NULL,
    milvus_collection TEXT,
    milvus_id TEXT NOT NULL,
    indexed_at TEXT,
    PRIMARY KEY (frame_uid, embedding_model, embedding_version),
    FOREIGN KEY (frame_uid) REFERENCES keyframes(frame_uid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ocr_frames (
    frame_uid TEXT NOT NULL,
    stage TEXT NOT NULL,
    model TEXT NOT NULL,
    version TEXT NOT NULL,
    status TEXT NOT NULL,
    has_text INTEGER,
    raw_text TEXT,
    normalized_text TEXT,
    processed_at TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (frame_uid, stage, model, version),
    FOREIGN KEY (frame_uid) REFERENCES keyframes(frame_uid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ocr_blocks (
    frame_uid TEXT NOT NULL,
    model TEXT NOT NULL,
    version TEXT NOT NULL,
    block_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    legibility TEXT,
    y_min INTEGER NOT NULL,
    x_min INTEGER NOT NULL,
    y_max INTEGER NOT NULL,
    x_max INTEGER NOT NULL,
    PRIMARY KEY (frame_uid, model, version, block_id),
    FOREIGN KEY (frame_uid) REFERENCES keyframes(frame_uid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS object_frames (
    frame_uid TEXT NOT NULL,
    model_id TEXT NOT NULL,
    run_signature TEXT NOT NULL,
    raw_detection_count INTEGER NOT NULL,
    filtered_detection_count INTEGER NOT NULL,
    object_signature TEXT NOT NULL,
    processed_at TEXT,
    PRIMARY KEY (frame_uid, model_id, run_signature),
    FOREIGN KEY (frame_uid) REFERENCES keyframes(frame_uid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS object_detections (
    frame_uid TEXT NOT NULL,
    model_id TEXT NOT NULL,
    run_signature TEXT NOT NULL,
    detection_id INTEGER NOT NULL,
    class_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    score REAL NOT NULL,
    x_min REAL NOT NULL,
    y_min REAL NOT NULL,
    x_max REAL NOT NULL,
    y_max REAL NOT NULL,
    x_min_1000 INTEGER NOT NULL,
    y_min_1000 INTEGER NOT NULL,
    x_max_1000 INTEGER NOT NULL,
    y_max_1000 INTEGER NOT NULL,
    area_ratio REAL NOT NULL,
    PRIMARY KEY (frame_uid, model_id, run_signature, detection_id),
    FOREIGN KEY (frame_uid) REFERENCES keyframes(frame_uid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS asr_segments (
    segment_id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    level TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    clean_text TEXT NOT NULL,
    language TEXT,
    provider TEXT,
    model TEXT,
    primary_shot INTEGER,
    quality_score REAL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS segment_keyframes (
    segment_id TEXT NOT NULL,
    frame_uid TEXT NOT NULL,
    video_id TEXT NOT NULL,
    frame_id INTEGER NOT NULL,
    PRIMARY KEY (segment_id, frame_uid),
    FOREIGN KEY (segment_id) REFERENCES asr_segments(segment_id) ON DELETE CASCADE,
    FOREIGN KEY (frame_uid) REFERENCES keyframes(frame_uid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS text_documents (
    doc_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    frame_uid TEXT,
    segment_id TEXT,
    video_id TEXT NOT NULL,
    level TEXT,
    start_ms INTEGER,
    end_ms INTEGER,
    normalized_text TEXT NOT NULL,
    raw_text TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS text_documents_fts USING fts5(
    doc_id UNINDEXED,
    source_type UNINDEXED,
    video_id UNINDEXED,
    level UNINDEXED,
    normalized_text,
    raw_text,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE INDEX IF NOT EXISTS idx_shots_time
    ON shots(video_id, start_ms, end_ms);
CREATE INDEX IF NOT EXISTS idx_keyframes_video_time
    ON keyframes(video_id, pts_ms, frame_id);
CREATE INDEX IF NOT EXISTS idx_keyframes_shot
    ON keyframes(video_id, shot_id, pts_ms);
CREATE INDEX IF NOT EXISTS idx_embeddings_model
    ON frame_embeddings(embedding_model, embedding_version, frame_uid);
CREATE INDEX IF NOT EXISTS idx_ocr_frames_status
    ON ocr_frames(status, has_text, frame_uid);
CREATE INDEX IF NOT EXISTS idx_ocr_blocks_frame
    ON ocr_blocks(frame_uid, block_id);
CREATE INDEX IF NOT EXISTS idx_object_class_score
    ON object_detections(class_id, score, frame_uid);
CREATE INDEX IF NOT EXISTS idx_object_frame_class
    ON object_detections(frame_uid, class_id, score);
CREATE INDEX IF NOT EXISTS idx_asr_video_time
    ON asr_segments(video_id, start_ms, end_ms);
CREATE INDEX IF NOT EXISTS idx_segment_keyframes_frame
    ON segment_keyframes(frame_uid, segment_id);
CREATE INDEX IF NOT EXISTS idx_text_documents_frame
    ON text_documents(frame_uid, source_type);
CREATE INDEX IF NOT EXISTS idx_text_documents_video_time
    ON text_documents(video_id, start_ms, end_ms);
"""


def initialize_catalog(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA)
    connection.execute(
        "INSERT OR REPLACE INTO catalog_meta(key, value) VALUES('schema_version', ?)",
        (SCHEMA_VERSION,),
    )
    connection.commit()
