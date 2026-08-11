from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common.identity import ensure_frame_identity
from ..common.paths import PipelinePaths
from .artifacts import DEFAULT_SELECTION_ENCODER_ID, load_keyframe_documents


@dataclass(frozen=True)
class EmbeddingSource:
    mapping_path: Path
    vector_path: Path
    mapping: pd.DataFrame
    dimension: int
    dtype: str


def default_collection_name(entity_type: str, model: str, version: str) -> str:
    value = re.sub(
        r"[^a-zA-Z0-9_]+", "_", f"{entity_type}_{model}_{version}"
    )
    return value.strip("_").lower()[:255]


def discover_embedding_sources(
    paths: PipelinePaths,
    *,
    encoder_id: str,
    entity_type: str,
    selection_encoder_id: str = DEFAULT_SELECTION_ENCODER_ID,
) -> tuple[list[EmbeddingSource], dict[str, dict[str, Any]]]:
    if entity_type not in {"keyframe", "roi"}:
        raise ValueError("entity type must be keyframe or roi")
    keyframes = (
        load_keyframe_documents(paths, selection_encoder_id=selection_encoder_id)
        if entity_type == "keyframe"
        else {}
    )
    sources: list[EmbeddingSource] = []
    dimensions: set[int] = set()
    feature_dir = paths.features / encoder_id
    for mapping_path in sorted(feature_dir.glob("*.parquet")):
        if ".smoke" in mapping_path.stem:
            continue
        vector_path = mapping_path.with_suffix(".f16.npy")
        if not vector_path.is_file():
            continue
        mapping = pd.read_parquet(mapping_path)
        if entity_type == "keyframe":
            if "entity_type" in mapping.columns:
                mapping = mapping.loc[mapping["entity_type"] == "keyframe"].copy()
            mapping = ensure_frame_identity(mapping)
            mapping = mapping.loc[mapping["frame_uid"].isin(keyframes)].copy()
        else:
            if "entity_type" not in mapping.columns:
                continue
            mapping = mapping.loc[mapping["entity_type"] == "roi"].copy()
            required = {"roi_uid", "parent_frame_uid", "embedding_row"}
            missing = sorted(required - set(mapping.columns))
            if missing:
                raise ValueError(f"ROI mapping is missing columns: {missing}")
        if mapping.empty:
            continue
        matrix = np.load(vector_path, mmap_mode="r")
        if matrix.ndim != 2:
            raise ValueError(f"Expected a 2D embedding matrix: {vector_path}")
        rows = mapping["embedding_row"].astype(int)
        if (rows < 0).any() or (rows >= len(matrix)).any():
            raise ValueError(f"Invalid embedding rows in {mapping_path}")
        dimension = int(matrix.shape[1])
        dimensions.add(dimension)
        sources.append(
            EmbeddingSource(
                mapping_path=mapping_path,
                vector_path=vector_path,
                mapping=mapping.sort_values("embedding_row").reset_index(drop=True),
                dimension=dimension,
                dtype=str(matrix.dtype),
            )
        )
    if len(dimensions) > 1:
        raise ValueError(f"Embedding dimensions differ for encoder {encoder_id}")
    return sources, keyframes


def _single_value(
    sources: list[EmbeddingSource], column: str, fallback: str
) -> str:
    values = {
        str(value)
        for source in sources
        if column in source.mapping.columns
        for value in source.mapping[column].dropna().unique()
    }
    if not values:
        return fallback
    if len(values) != 1:
        raise ValueError(f"Multiple {column} values found: {sorted(values)}")
    return values.pop()


def iter_keyframe_rows(
    sources: list[EmbeddingSource],
    keyframes: dict[str, dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    for source in sources:
        matrix = np.load(source.vector_path, mmap_mode="r")
        for record in source.mapping.to_dict(orient="records"):
            uid = str(record["frame_uid"])
            keyframe = keyframes[uid]
            vector = np.asarray(matrix[int(record["embedding_row"])], dtype=np.float32)
            if len(vector) != source.dimension:
                raise ValueError(f"Embedding dimension mismatch for {uid}")
            yield {
                "frame_uid": uid,
                "video_id": str(keyframe["video_id"]),
                "frame_id": int(keyframe["frame_id"]),
                "shot_id": (
                    int(keyframe["shot_id"])
                    if keyframe.get("shot_id") is not None
                    else -1
                ),
                "pts_ms": (
                    int(keyframe["pts_ms"])
                    if keyframe.get("pts_ms") is not None
                    else -1
                ),
                "image_path": str(keyframe.get("image_path") or ""),
                "embedding": vector.tolist(),
            }


def iter_roi_rows(sources: list[EmbeddingSource]) -> Iterator[dict[str, Any]]:
    for source in sources:
        matrix = np.load(source.vector_path, mmap_mode="r")
        for record in source.mapping.to_dict(orient="records"):
            uid = str(record["roi_uid"])
            vector = np.asarray(matrix[int(record["embedding_row"])], dtype=np.float32)
            if len(vector) != source.dimension:
                raise ValueError(f"Embedding dimension mismatch for {uid}")
            yield {
                "roi_uid": uid,
                "parent_frame_uid": str(record["parent_frame_uid"]),
                "video_id": str(record["video_id"]),
                "frame_id": int(record["frame_id"]),
                "shot_id": int(record["shot_id"])
                if pd.notna(record.get("shot_id"))
                else -1,
                "pts_ms": int(record["pts_ms"])
                if pd.notna(record.get("pts_ms"))
                else -1,
                "class_id": int(record["class_id"]),
                "class_name": str(record["class_name"]),
                "confidence": float(record["score"]),
                "x_min_1000": int(record["x_min_1000"]),
                "y_min_1000": int(record["y_min_1000"]),
                "x_max_1000": int(record["x_max_1000"]),
                "y_max_1000": int(record["y_max_1000"]),
                "embedding": vector.tolist(),
            }


def _batches(rows: Iterator[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _client(uri: str, token: str | None, client: Any | None) -> tuple[Any, Any]:
    if client is not None:
        return client, client.DataType
    try:
        from pymilvus import DataType, MilvusClient
    except ImportError as error:  # pragma: no cover - optional runtime.
        raise RuntimeError("Milvus ingestion requires environments/serving.") from error
    instance = MilvusClient(uri=uri, token=token) if token else MilvusClient(uri=uri)
    return instance, DataType


def _create_collection(
    client: Any,
    data_type: Any,
    *,
    collection: str,
    entity_type: str,
    dimension: int,
) -> None:
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    if entity_type == "keyframe":
        schema.add_field(
            field_name="frame_uid",
            datatype=data_type.VARCHAR,
            is_primary=True,
            max_length=128,
        )
        schema.add_field(field_name="video_id", datatype=data_type.VARCHAR, max_length=64)
        schema.add_field(field_name="frame_id", datatype=data_type.INT64)
        schema.add_field(field_name="shot_id", datatype=data_type.INT64)
        schema.add_field(field_name="pts_ms", datatype=data_type.INT64)
        schema.add_field(
            field_name="image_path", datatype=data_type.VARCHAR, max_length=1024
        )
    else:
        schema.add_field(
            field_name="roi_uid",
            datatype=data_type.VARCHAR,
            is_primary=True,
            max_length=192,
        )
        schema.add_field(
            field_name="parent_frame_uid",
            datatype=data_type.VARCHAR,
            max_length=128,
        )
        schema.add_field(field_name="video_id", datatype=data_type.VARCHAR, max_length=64)
        schema.add_field(field_name="frame_id", datatype=data_type.INT64)
        schema.add_field(field_name="shot_id", datatype=data_type.INT64)
        schema.add_field(field_name="pts_ms", datatype=data_type.INT64)
        schema.add_field(field_name="class_id", datatype=data_type.INT64)
        schema.add_field(
            field_name="class_name", datatype=data_type.VARCHAR, max_length=128
        )
        schema.add_field(field_name="confidence", datatype=data_type.FLOAT)
        for field in ("x_min_1000", "y_min_1000", "x_max_1000", "y_max_1000"):
            schema.add_field(field_name=field, datatype=data_type.INT64)
    schema.add_field(
        field_name="embedding", datatype=data_type.FLOAT_VECTOR, dim=dimension
    )
    indexes = client.prepare_index_params()
    indexes.add_index(
        field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE"
    )
    client.create_collection(
        collection_name=collection, schema=schema, index_params=indexes
    )


def ingest_artifact_embeddings(
    paths: PipelinePaths,
    *,
    encoder_id: str,
    entity_type: str,
    uri: str,
    token: str | None = None,
    collection_name: str | None = None,
    model: str | None = None,
    version: str | None = None,
    selection_encoder_id: str = DEFAULT_SELECTION_ENCODER_ID,
    batch_size: int = 512,
    drop_existing: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    sources, keyframes = discover_embedding_sources(
        paths,
        encoder_id=encoder_id,
        entity_type=entity_type,
        selection_encoder_id=selection_encoder_id,
    )
    if not sources:
        raise ValueError(f"No {entity_type} embeddings found for encoder {encoder_id}")
    dimension = sources[0].dimension
    model = model or _single_value(sources, "embedding_model", encoder_id)
    version = version or _single_value(sources, "embedding_version", model)
    collection = collection_name or default_collection_name(entity_type, model, version)
    milvus, data_type = _client(uri, token, client)
    if drop_existing and milvus.has_collection(collection_name=collection):
        milvus.drop_collection(collection_name=collection)
    if not milvus.has_collection(collection_name=collection):
        _create_collection(
            milvus,
            data_type,
            collection=collection,
            entity_type=entity_type,
            dimension=dimension,
        )
    rows = (
        iter_keyframe_rows(sources, keyframes)
        if entity_type == "keyframe"
        else iter_roi_rows(sources)
    )
    inserted = 0
    for batch in _batches(rows, batch_size):
        milvus.upsert(collection_name=collection, data=batch)
        inserted += len(batch)
    milvus.flush(collection_name=collection)
    expected = sum(len(source.mapping) for source in sources)
    if inserted != expected:
        raise RuntimeError(f"Inserted {inserted}/{expected} embeddings")
    return {
        "entity_type": entity_type,
        "encoder_id": encoder_id,
        "model": model,
        "version": version,
        "collection": collection,
        "dimension": dimension,
        "rows": inserted,
        "sources": [str(source.mapping_path) for source in sources],
    }
