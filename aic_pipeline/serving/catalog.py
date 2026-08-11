from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common.identity import ensure_frame_identity, frame_uid, seconds_to_ms
from ..common.paths import PipelinePaths
from .schema import SCHEMA_VERSION, initialize_catalog

DEFAULT_SELECTION_ENCODER_ID = "timm_beit3_base_patch16_224_pt"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _optional(value: Any) -> Any:
    if value is None:
        return None
    missing = pd.isna(value)
    if not hasattr(missing, "__len__") and bool(missing):
        return None
    return value


def _portable_path(path: str | Path | None, root: Path) -> str | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def _json_list(value: Any) -> list[Any]:
    value = _optional(value)
    if value is None:
        return []
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return list(value)


def _insert_many(
    connection: sqlite3.Connection,
    sql: str,
    rows: Iterable[tuple[Any, ...]],
) -> int:
    materialized = list(rows)
    connection.executemany(sql, materialized)
    return len(materialized)


def _load_videos(
    connection: sqlite3.Connection, paths: PipelinePaths
) -> tuple[dict[str, dict[str, Any]], int]:
    manifest_path = paths.manifests / "videos.parquet"
    if not manifest_path.is_file():
        return {}, 0
    frame = pd.read_parquet(manifest_path)
    records: dict[str, dict[str, Any]] = {}
    rows = []
    for record in frame.to_dict(orient="records"):
        video_id = str(record["video_id"])
        records[video_id] = record
        rows.append(
            (
                video_id,
                _optional(record.get("group")),
                _portable_path(record["video_path"], paths.root),
                float(record["fps"])
                if _optional(record.get("fps")) is not None
                else None,
                int(record["frame_count"])
                if _optional(record.get("frame_count")) is not None
                else None,
                seconds_to_ms(record.get("duration_seconds")),
                int(record["width"])
                if _optional(record.get("width")) is not None
                else None,
                int(record["height"])
                if _optional(record.get("height")) is not None
                else None,
                json.dumps(record, ensure_ascii=False, default=str),
            )
        )
    count = _insert_many(
        connection,
        """
        INSERT INTO videos(
            video_id, group_name, video_path, fps, frame_count, duration_ms,
            width, height, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return records, count


def _load_shots(
    connection: sqlite3.Connection,
    paths: PipelinePaths,
    videos: dict[str, dict[str, Any]],
) -> int:
    rows = []
    for shot_path in sorted(paths.shots.glob("*.json")):
        video_id = shot_path.stem
        video = videos.get(video_id)
        if video is None or not _optional(video.get("fps")):
            continue
        fps = float(video["fps"])
        payload = json.loads(shot_path.read_text(encoding="utf-8"))
        for shot_id, pair in enumerate(payload.get("ranges") or []):
            start_frame, end_frame = map(int, pair)
            rows.append(
                (
                    video_id,
                    shot_id,
                    start_frame,
                    end_frame,
                    seconds_to_ms(start_frame / fps),
                    seconds_to_ms(end_frame / fps),
                    payload.get("detector"),
                    _portable_path(shot_path, paths.root),
                )
            )
    return _insert_many(
        connection,
        """
        INSERT INTO shots(
            video_id, shot_id, start_frame, end_frame_exclusive,
            start_ms, end_ms, detector, source_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _load_keyframes(
    connection: sqlite3.Connection,
    paths: PipelinePaths,
    selection_encoder_id: str,
) -> set[str]:
    selection_root = paths.keyframes / selection_encoder_id
    rows = []
    for manifest_path in sorted(selection_root.glob("*.parquet")):
        if manifest_path.name.endswith(".audit.parquet"):
            continue
        frame = ensure_frame_identity(pd.read_parquet(manifest_path))
        if "selected" in frame.columns:
            frame = frame.loc[frame["selected"].astype(bool)].copy()
        video_id = manifest_path.stem
        jpg_manifest = (
            paths.workspace / "keyframe_images" / video_id / "manifest.parquet"
        )
        if jpg_manifest.is_file():
            jpg = ensure_frame_identity(pd.read_parquet(jpg_manifest))
            jpg = jpg[
                [
                    column
                    for column in ("frame_uid", "image_path", "image_sha256")
                    if column in jpg.columns
                ]
            ]
            frame = frame.drop(
                columns=[
                    column
                    for column in ("image_path", "image_sha256")
                    if column in frame.columns
                ]
            ).merge(jpg, on="frame_uid", how="left", validate="one_to_one")
        for record in frame.to_dict(orient="records"):
            rows.append(
                (
                    record["frame_uid"],
                    str(record["video_id"]),
                    int(record["frame_id"]),
                    int(record["shot_id"])
                    if _optional(record.get("shot_id")) is not None
                    else None,
                    int(record["pts_ms"])
                    if _optional(record.get("pts_ms")) is not None
                    else None,
                    _portable_path(_optional(record.get("image_path")), paths.root),
                    _optional(record.get("image_sha256")),
                    selection_encoder_id,
                    str(record.get("selection_version") or selection_encoder_id),
                    _portable_path(manifest_path, paths.root),
                )
            )
    _insert_many(
        connection,
        """
        INSERT INTO keyframes(
            frame_uid, video_id, frame_id, shot_id, pts_ms, image_path,
            image_sha256, selection_encoder_id, selection_version, source_manifest
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return {str(row[0]) for row in rows}


def _load_embeddings(
    connection: sqlite3.Connection,
    paths: PipelinePaths,
    keyframe_uids: set[str],
) -> int:
    rows = []
    for mapping_path in sorted(paths.features.glob("*/*.parquet")):
        if ".smoke" in mapping_path.stem:
            continue
        vector_path = mapping_path.with_suffix(".f16.npy")
        if not vector_path.is_file():
            continue
        mapping = ensure_frame_identity(pd.read_parquet(mapping_path))
        if "entity_type" in mapping.columns:
            mapping = mapping.loc[mapping["entity_type"] != "roi"].copy()
        mapping = mapping.loc[mapping["frame_uid"].isin(keyframe_uids)]
        if mapping.empty:
            continue
        matrix = np.load(vector_path, mmap_mode="r")
        if matrix.ndim != 2:
            continue
        model = mapping_path.parent.name
        for record in mapping.to_dict(orient="records"):
            embedding_row = int(record["embedding_row"])
            if not 0 <= embedding_row < len(matrix):
                raise ValueError(
                    f"Invalid embedding row in {mapping_path}: {embedding_row}"
                )
            rows.append(
                (
                    record["frame_uid"],
                    str(record.get("encoder_id") or model),
                    str(record.get("embedding_version") or model),
                    int(matrix.shape[1]),
                    str(matrix.dtype),
                    _portable_path(vector_path, paths.root),
                    embedding_row,
                    None,
                    record["frame_uid"],
                    None,
                )
            )
    return _insert_many(
        connection,
        """
        INSERT OR REPLACE INTO frame_embeddings(
            frame_uid, embedding_model, embedding_version, dimension, dtype,
            vector_path, embedding_row, milvus_collection, milvus_id, indexed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _insert_text_document(
    connection: sqlite3.Connection,
    record: tuple[Any, ...],
) -> None:
    connection.execute(
        """
        INSERT OR REPLACE INTO text_documents(
            doc_id, source_type, source_id, frame_uid, segment_id, video_id,
            level, start_ms, end_ms, normalized_text, raw_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        record,
    )
    connection.execute("DELETE FROM text_documents_fts WHERE doc_id = ?", (record[0],))
    connection.execute(
        """
        INSERT INTO text_documents_fts(
            doc_id, source_type, video_id, level, normalized_text, raw_text
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (record[0], record[1], record[5], record[6], record[9], record[10]),
    )


def _load_ocr(
    connection: sqlite3.Connection,
    paths: PipelinePaths,
    keyframe_uids: set[str],
) -> tuple[int, int]:
    frame_count = 0
    block_count = 0
    preprocess_root = paths.workspace / "ocr" / "preprocess"
    for manifest_path in sorted(preprocess_root.glob("*/screening.parquet")):
        frame = ensure_frame_identity(pd.read_parquet(manifest_path))
        for record in frame.to_dict(orient="records"):
            if record["frame_uid"] not in keyframe_uids:
                continue
            version = str(record.get("preprocess_version") or "screening")
            connection.execute(
                """
                INSERT OR REPLACE INTO ocr_frames(
                    frame_uid, stage, model, version, status, has_text,
                    raw_text, normalized_text, processed_at, metadata_json
                ) VALUES (?, 'screening', ?, ?, ?, ?, NULL, NULL, NULL, ?)
                """,
                (
                    record["frame_uid"],
                    str(record.get("model_name") or "ppocr"),
                    version,
                    "candidate"
                    if bool(record["screening_has_text"])
                    else "screened_out",
                    int(bool(record["screening_has_text"])),
                    json.dumps(record, ensure_ascii=False, default=str),
                ),
            )
            if record.get("image_sha256") or record.get("image_path"):
                connection.execute(
                    """
                    UPDATE keyframes
                    SET image_sha256 = COALESCE(image_sha256, ?),
                        image_path = COALESCE(image_path, ?)
                    WHERE frame_uid = ?
                    """,
                    (
                        _optional(record.get("image_sha256")),
                        _portable_path(_optional(record.get("image_path")), paths.root),
                        record["frame_uid"],
                    ),
                )
            frame_count += 1

    gemini_root = paths.workspace / "ocr" / "gemini"
    for frame_path in sorted(gemini_root.glob("*/*/frames.parquet")):
        frame = ensure_frame_identity(pd.read_parquet(frame_path))
        for record in frame.to_dict(orient="records"):
            uid = record["frame_uid"]
            if uid not in keyframe_uids:
                continue
            model = str(record["model"])
            version = str(record["prompt_version"])
            connection.execute(
                """
                INSERT OR REPLACE INTO ocr_frames(
                    frame_uid, stage, model, version, status, has_text,
                    raw_text, normalized_text, processed_at, metadata_json
                ) VALUES (?, 'ocr', ?, ?, 'success', ?, ?, ?, ?, ?)
                """,
                (
                    uid,
                    model,
                    version,
                    int(bool(record["has_text"])),
                    record.get("raw_text"),
                    record.get("normalized_text"),
                    _optional(record.get("processed_at")),
                    json.dumps(record, ensure_ascii=False, default=str),
                ),
            )
            if record.get("image_sha256") or record.get("image_path"):
                connection.execute(
                    """
                    UPDATE keyframes
                    SET image_sha256 = COALESCE(image_sha256, ?),
                        image_path = COALESCE(image_path, ?)
                    WHERE frame_uid = ?
                    """,
                    (
                        _optional(record.get("image_sha256")),
                        _portable_path(_optional(record.get("image_path")), paths.root),
                        record["frame_uid"],
                    ),
                )
            frame_count += 1
            blocks = _json_list(record.get("blocks_json"))
            for block_id, block in enumerate(blocks):
                y_min, x_min, y_max, x_max = map(int, block["bbox_2d"])
                connection.execute(
                    """
                    INSERT OR REPLACE INTO ocr_blocks(
                        frame_uid, model, version, block_id, text, legibility,
                        y_min, x_min, y_max, x_max
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uid,
                        model,
                        version,
                        block_id,
                        str(block["text"]),
                        block.get("legibility"),
                        y_min,
                        x_min,
                        y_max,
                        x_max,
                    ),
                )
                block_count += 1
            if record.get("normalized_text"):
                _insert_text_document(
                    connection,
                    (
                        f"ocr:{model}:{version}:{uid}",
                        "ocr",
                        uid,
                        uid,
                        None,
                        str(record["video_id"]),
                        None,
                        int(record["pts_ms"])
                        if _optional(record.get("pts_ms")) is not None
                        else None,
                        int(record["pts_ms"])
                        if _optional(record.get("pts_ms")) is not None
                        else None,
                        str(record["normalized_text"]),
                        str(record.get("raw_text") or record["normalized_text"]),
                    ),
                )

    for failure_path in sorted(gemini_root.glob("*/*/failures.parquet")):
        frame = ensure_frame_identity(pd.read_parquet(failure_path))
        for record in frame.to_dict(orient="records"):
            uid = record["frame_uid"]
            if uid not in keyframe_uids:
                continue
            connection.execute(
                """
                INSERT OR IGNORE INTO ocr_frames(
                    frame_uid, stage, model, version, status, has_text,
                    raw_text, normalized_text, processed_at, metadata_json
                ) VALUES (?, 'ocr', ?, ?, 'failure', NULL, NULL, NULL, ?, ?)
                """,
                (
                    uid,
                    str(record["model"]),
                    str(record["prompt_version"]),
                    _optional(record.get("failed_at")),
                    json.dumps(record, ensure_ascii=False, default=str),
                ),
            )
            if record.get("image_sha256") or record.get("image_path"):
                connection.execute(
                    """
                    UPDATE keyframes
                    SET image_sha256 = COALESCE(image_sha256, ?),
                        image_path = COALESCE(image_path, ?)
                    WHERE frame_uid = ?
                    """,
                    (
                        _optional(record.get("image_sha256")),
                        _portable_path(_optional(record.get("image_path")), paths.root),
                        record["frame_uid"],
                    ),
                )
            frame_count += 1
    return frame_count, block_count


def _load_objects(
    connection: sqlite3.Connection,
    paths: PipelinePaths,
    keyframe_uids: set[str],
) -> tuple[int, int]:
    frame_count = 0
    detection_count = 0
    object_root = paths.workspace / "objects"
    for frame_path in sorted(object_root.glob("*/*/frames.parquet")):
        frame = ensure_frame_identity(pd.read_parquet(frame_path))
        for record in frame.to_dict(orient="records"):
            if record["frame_uid"] not in keyframe_uids:
                continue
            connection.execute(
                """
                INSERT OR REPLACE INTO object_frames(
                    frame_uid, model_id, run_signature, raw_detection_count,
                    filtered_detection_count, object_signature, processed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["frame_uid"],
                    str(record["model_id"]),
                    str(record["run_signature"]),
                    int(record["raw_detection_count"]),
                    int(record["filtered_detection_count"]),
                    str(record["object_signature"]),
                    _optional(record.get("processed_at")),
                ),
            )
            frame_count += 1
    for detection_path in sorted(object_root.glob("*/*/detections.parquet")):
        frame = ensure_frame_identity(pd.read_parquet(detection_path))
        for record in frame.to_dict(orient="records"):
            if record["frame_uid"] not in keyframe_uids:
                continue
            connection.execute(
                """
                INSERT OR REPLACE INTO object_detections(
                    frame_uid, model_id, run_signature, detection_id,
                    class_id, class_name, score, x_min, y_min, x_max, y_max,
                    x_min_1000, y_min_1000, x_max_1000, y_max_1000, area_ratio
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["frame_uid"],
                    str(record["model_id"]),
                    str(record["run_signature"]),
                    int(record["detection_id"]),
                    int(record["class_id"]),
                    str(record["class_name"]),
                    float(record["score"]),
                    float(record["x_min"]),
                    float(record["y_min"]),
                    float(record["x_max"]),
                    float(record["y_max"]),
                    int(record["x_min_1000"]),
                    int(record["y_min_1000"]),
                    int(record["x_max_1000"]),
                    int(record["y_max_1000"]),
                    float(record["area_ratio"]),
                ),
            )
            detection_count += 1
    return frame_count, detection_count


def _load_asr(
    connection: sqlite3.Connection,
    paths: PipelinePaths,
    keyframe_uids: set[str],
) -> tuple[int, int]:
    segment_count = 0
    link_count = 0
    for transcript_path in sorted(paths.transcripts.glob("*/l[01].parquet")):
        frame = pd.read_parquet(transcript_path)
        for record in frame.to_dict(orient="records"):
            segment_id = str(record["segment_id"])
            video_id = str(record["video_id"])
            start_ms = (
                int(record["start_ms"])
                if _optional(record.get("start_ms")) is not None
                else seconds_to_ms(record["start"])
            )
            end_ms = (
                int(record["end_ms"])
                if _optional(record.get("end_ms")) is not None
                else seconds_to_ms(record["end"])
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO asr_segments(
                    segment_id, video_id, level, start_ms, end_ms, raw_text,
                    clean_text, language, provider, model, primary_shot,
                    quality_score, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    segment_id,
                    video_id,
                    str(record["level"]),
                    start_ms,
                    end_ms,
                    str(record["raw_text"]),
                    str(record.get("clean_text") or record["raw_text"]),
                    _optional(record.get("language")),
                    _optional(record.get("provider")),
                    _optional(record.get("model")),
                    int(record["primary_shot"])
                    if _optional(record.get("primary_shot")) is not None
                    else None,
                    float(record["quality_score"])
                    if _optional(record.get("quality_score")) is not None
                    else None,
                    str(record.get("metadata_json") or "{}"),
                ),
            )
            segment_count += 1
            for frame_id_value in _json_list(record.get("keyframe_ids")):
                uid = frame_uid(video_id, int(frame_id_value))
                if uid not in keyframe_uids:
                    continue
                connection.execute(
                    """
                    INSERT OR REPLACE INTO segment_keyframes(
                        segment_id, frame_uid, video_id, frame_id
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (segment_id, uid, video_id, int(frame_id_value)),
                )
                link_count += 1
            text = str(record.get("clean_text") or record["raw_text"])
            _insert_text_document(
                connection,
                (
                    f"asr:{segment_id}",
                    "asr",
                    segment_id,
                    None,
                    segment_id,
                    video_id,
                    str(record["level"]),
                    start_ms,
                    end_ms,
                    text,
                    str(record["raw_text"]),
                ),
            )
    return segment_count, link_count


def build_serving_catalog(
    paths: PipelinePaths,
    destination: Path,
    *,
    selection_encoder_id: str = DEFAULT_SELECTION_ENCODER_ID,
) -> dict[str, Any]:
    paths.ensure_output_dirs()
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        initialize_catalog(connection)
        videos, video_count = _load_videos(connection, paths)
        shot_count = _load_shots(connection, paths, videos)
        keyframe_uids = _load_keyframes(connection, paths, selection_encoder_id)
        embedding_count = _load_embeddings(connection, paths, keyframe_uids)
        ocr_frame_count, ocr_block_count = _load_ocr(connection, paths, keyframe_uids)
        object_frame_count, object_detection_count = _load_objects(
            connection, paths, keyframe_uids
        )
        asr_segment_count, segment_keyframe_count = _load_asr(
            connection, paths, keyframe_uids
        )
        connection.execute(
            "INSERT INTO text_documents_fts(text_documents_fts) VALUES('optimize')"
        )
        connection.execute(
            "INSERT OR REPLACE INTO catalog_meta(key, value) VALUES('built_at', ?)",
            (utc_now(),),
        )
        connection.commit()
    finally:
        connection.close()
    os.replace(temporary, destination)
    return {
        "schema_version": SCHEMA_VERSION,
        "destination": str(destination),
        "selection_encoder_id": selection_encoder_id,
        "videos": video_count,
        "shots": shot_count,
        "keyframes": len(keyframe_uids),
        "embeddings": embedding_count,
        "ocr_frames": ocr_frame_count,
        "ocr_blocks": ocr_block_count,
        "object_frames": object_frame_count,
        "object_detections": object_detection_count,
        "asr_segments": asr_segment_count,
        "segment_keyframes": segment_keyframe_count,
    }


def search_text(
    connection: sqlite3.Connection,
    query: str,
    *,
    source_type: str | None = None,
    video_id: str | None = None,
    level: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    if not query.strip() or limit <= 0:
        return []
    clauses = ["text_documents_fts MATCH ?"]
    parameters: list[Any] = [query]
    for column, value in (
        ("documents.source_type", source_type),
        ("documents.video_id", video_id),
        ("documents.level", level),
    ):
        if value is not None:
            clauses.append(f"{column} = ?")
            parameters.append(value)
    parameters.append(limit)
    rows = connection.execute(
        f"""
        SELECT documents.doc_id, documents.source_type, documents.source_id,
               documents.frame_uid, documents.segment_id, documents.video_id,
               documents.level, documents.start_ms, documents.end_ms,
               documents.normalized_text, documents.raw_text,
               text_documents_fts.rank
        FROM text_documents_fts
        JOIN text_documents AS documents
          ON documents.doc_id = text_documents_fts.doc_id
        WHERE {" AND ".join(clauses)}
        ORDER BY text_documents_fts.rank
        LIMIT ?
        """,
        parameters,
    ).fetchall()
    return [
        {
            "doc_id": row[0],
            "source_type": row[1],
            "source_id": row[2],
            "frame_uid": row[3],
            "segment_id": row[4],
            "video_id": row[5],
            "level": row[6],
            "start_ms": row[7],
            "end_ms": row[8],
            "normalized_text": row[9],
            "raw_text": row[10],
            "score": -float(row[11]),
        }
        for row in rows
    ]


def search_objects(
    connection: sqlite3.Connection,
    class_name: str,
    *,
    min_score: float = 0.3,
    video_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    class_name = class_name.strip()
    if not class_name or not 0 <= min_score <= 1 or limit <= 0:
        return []
    clauses = ["detections.class_name = ?", "detections.score >= ?"]
    parameters: list[Any] = [class_name, min_score]
    if video_id is not None:
        clauses.append("keyframes.video_id = ?")
        parameters.append(video_id)
    parameters.append(limit)
    rows = connection.execute(
        f"""
        SELECT keyframes.frame_uid, keyframes.video_id, keyframes.frame_id,
               keyframes.shot_id, keyframes.pts_ms, keyframes.image_path,
               detections.class_id, detections.class_name,
               COUNT(*) AS object_count, MAX(detections.score) AS max_score
        FROM object_detections AS detections
        JOIN keyframes ON keyframes.frame_uid = detections.frame_uid
        WHERE {" AND ".join(clauses)}
        GROUP BY keyframes.frame_uid, keyframes.video_id, keyframes.frame_id,
                 keyframes.shot_id, keyframes.pts_ms, keyframes.image_path,
                 detections.class_id, detections.class_name
        ORDER BY max_score DESC, keyframes.video_id, keyframes.frame_id
        LIMIT ?
        """,
        parameters,
    ).fetchall()
    return [
        {
            "frame_uid": row[0],
            "video_id": row[1],
            "frame_id": row[2],
            "shot_id": row[3],
            "pts_ms": row[4],
            "image_path": row[5],
            "class_id": row[6],
            "class_name": row[7],
            "object_count": row[8],
            "max_score": row[9],
        }
        for row in rows
    ]


def frame_bundle(connection: sqlite3.Connection, uid: str) -> dict[str, Any] | None:
    connection.row_factory = sqlite3.Row
    keyframe = connection.execute(
        "SELECT * FROM keyframes WHERE frame_uid = ?", (uid,)
    ).fetchone()
    if keyframe is None:
        return None

    def rows(query: str) -> list[dict[str, Any]]:
        return [dict(row) for row in connection.execute(query, (uid,)).fetchall()]

    return {
        "keyframe": dict(keyframe),
        "embeddings": rows(
            "SELECT * FROM frame_embeddings WHERE frame_uid = ? ORDER BY embedding_model"
        ),
        "ocr_frames": rows(
            "SELECT * FROM ocr_frames WHERE frame_uid = ? ORDER BY stage, model"
        ),
        "ocr_blocks": rows(
            "SELECT * FROM ocr_blocks WHERE frame_uid = ? ORDER BY model, block_id"
        ),
        "object_frames": rows(
            "SELECT * FROM object_frames WHERE frame_uid = ? ORDER BY model_id"
        ),
        "object_detections": rows(
            """SELECT * FROM object_detections
               WHERE frame_uid = ? ORDER BY score DESC, detection_id"""
        ),
        "asr_segments": rows(
            """SELECT segments.* FROM segment_keyframes AS links
               JOIN asr_segments AS segments ON segments.segment_id = links.segment_id
               WHERE links.frame_uid = ?
               ORDER BY segments.start_ms, segments.level"""
        ),
    }
