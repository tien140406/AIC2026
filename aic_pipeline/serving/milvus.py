from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_collection_name(model: str, version: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", f"keyframes_{model}_{version}")
    return value.strip("_").lower()[:255]


def available_embedding_versions(
    connection: sqlite3.Connection,
) -> list[tuple[str, str, int, int]]:
    return [
        (str(row[0]), str(row[1]), int(row[2]), int(row[3]))
        for row in connection.execute(
            """
            SELECT embedding_model, embedding_version, dimension, COUNT(*)
            FROM frame_embeddings
            GROUP BY embedding_model, embedding_version, dimension
            ORDER BY embedding_model, embedding_version
            """
        ).fetchall()
    ]


def _artifact_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def iter_embedding_rows(
    connection: sqlite3.Connection,
    root: Path,
    *,
    model: str,
    version: str,
) -> Iterator[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT embeddings.frame_uid, embeddings.vector_path,
               embeddings.embedding_row, embeddings.dimension,
               keyframes.video_id, keyframes.frame_id, keyframes.shot_id,
               keyframes.pts_ms, keyframes.image_path,
               COALESCE(MAX(CASE WHEN ocr.status = 'success' THEN ocr.has_text END), 0),
               GROUP_CONCAT(DISTINCT detections.class_id)
        FROM frame_embeddings AS embeddings
        JOIN keyframes ON keyframes.frame_uid = embeddings.frame_uid
        LEFT JOIN ocr_frames AS ocr ON ocr.frame_uid = embeddings.frame_uid
        LEFT JOIN object_detections AS detections
          ON detections.frame_uid = embeddings.frame_uid
        WHERE embeddings.embedding_model = ? AND embeddings.embedding_version = ?
        GROUP BY embeddings.frame_uid, embeddings.vector_path,
                 embeddings.embedding_row, embeddings.dimension,
                 keyframes.video_id, keyframes.frame_id, keyframes.shot_id,
                 keyframes.pts_ms, keyframes.image_path
        ORDER BY embeddings.vector_path, embeddings.embedding_row
        """,
        (model, version),
    ).fetchall()
    active_path: Path | None = None
    matrix: np.ndarray | None = None
    for row in rows:
        vector_path = _artifact_path(root, str(row[1])).resolve()
        if vector_path != active_path:
            if not vector_path.is_file():
                raise FileNotFoundError(vector_path)
            matrix = np.load(vector_path, mmap_mode="r")
            active_path = vector_path
        assert matrix is not None
        embedding_row = int(row[2])
        vector = np.asarray(matrix[embedding_row], dtype=np.float32)
        if len(vector) != int(row[3]):
            raise ValueError(f"Embedding dimension mismatch for {row[0]}")
        class_ids = (
            sorted({int(value) for value in str(row[10]).split(",")}) if row[10] else []
        )
        yield {
            "frame_uid": str(row[0]),
            "video_id": str(row[4]),
            "frame_id": int(row[5]),
            "shot_id": int(row[6]) if row[6] is not None else -1,
            "pts_ms": int(row[7]) if row[7] is not None else -1,
            "image_path": str(row[8] or ""),
            "has_text": bool(row[9]),
            "object_class_ids": class_ids,
            "embedding": vector.tolist(),
        }


def _batches(
    rows: Iterator[dict[str, Any]], size: int
) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def ingest_embeddings(
    catalog_path: Path,
    root: Path,
    *,
    model: str,
    version: str,
    uri: str,
    token: str | None = None,
    collection_name: str | None = None,
    batch_size: int = 512,
    drop_existing: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    catalog_path = catalog_path.expanduser().resolve()
    root = root.expanduser().resolve()
    connection = sqlite3.connect(catalog_path)
    try:
        matches = [
            item
            for item in available_embedding_versions(connection)
            if item[0] == model and item[1] == version
        ]
        if len(matches) != 1:
            raise ValueError(f"Embedding version not found: {model}/{version}")
        dimension, expected_rows = matches[0][2], matches[0][3]
        collection = collection_name or default_collection_name(model, version)

        if client is None:
            try:
                from pymilvus import DataType, MilvusClient
            except ImportError as error:  # pragma: no cover - optional runtime.
                raise RuntimeError(
                    "Milvus ingestion requires environments/serving."
                ) from error
            client = (
                MilvusClient(uri=uri, token=token) if token else MilvusClient(uri=uri)
            )
        else:
            DataType = client.DataType

        if drop_existing and client.has_collection(collection_name=collection):
            client.drop_collection(collection_name=collection)
        if not client.has_collection(collection_name=collection):
            schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field(
                field_name="frame_uid",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=128,
            )
            schema.add_field(
                field_name="video_id", datatype=DataType.VARCHAR, max_length=64
            )
            schema.add_field(field_name="frame_id", datatype=DataType.INT64)
            schema.add_field(field_name="shot_id", datatype=DataType.INT64)
            schema.add_field(field_name="pts_ms", datatype=DataType.INT64)
            schema.add_field(
                field_name="image_path", datatype=DataType.VARCHAR, max_length=1024
            )
            schema.add_field(field_name="has_text", datatype=DataType.BOOL)
            schema.add_field(
                field_name="object_class_ids",
                datatype=DataType.ARRAY,
                element_type=DataType.INT64,
                max_capacity=80,
            )
            schema.add_field(
                field_name="embedding",
                datatype=DataType.FLOAT_VECTOR,
                dim=dimension,
            )
            indexes = client.prepare_index_params()
            indexes.add_index(
                field_name="embedding",
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )
            client.create_collection(
                collection_name=collection,
                schema=schema,
                index_params=indexes,
            )

        inserted = 0
        rows = iter_embedding_rows(
            connection,
            root,
            model=model,
            version=version,
        )
        for batch in _batches(rows, batch_size):
            client.upsert(collection_name=collection, data=batch)
            inserted += len(batch)
        client.flush(collection_name=collection)
        indexed_at = utc_now()
        connection.execute(
            """
            UPDATE frame_embeddings
            SET milvus_collection = ?, indexed_at = ?
            WHERE embedding_model = ? AND embedding_version = ?
            """,
            (collection, indexed_at, model, version),
        )
        connection.commit()
    finally:
        connection.close()
    if inserted != expected_rows:
        raise RuntimeError(f"Inserted {inserted}/{expected_rows} embeddings")
    return {
        "catalog": str(catalog_path),
        "collection": collection,
        "model": model,
        "version": version,
        "dimension": dimension,
        "rows": inserted,
        "indexed_at": indexed_at,
    }
