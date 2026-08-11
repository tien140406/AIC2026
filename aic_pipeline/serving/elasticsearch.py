from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from ..common.paths import PipelinePaths
from .artifacts import (
    DEFAULT_SELECTION_ENCODER_ID,
    iter_asr_documents,
    iter_frame_documents,
)

DEFAULT_ELASTICSEARCH_URL = "http://localhost:9200"
DEFAULT_FRAME_INDEX = "aic-frames-v1"
DEFAULT_ASR_INDEX = "aic-asr-segments-v1"

TEXT_ANALYSIS = {
    "filter": {
        "aic_asciifolding": {
            "type": "asciifolding",
            "preserve_original": True,
        }
    },
    "analyzer": {
        "aic_text": {
            "type": "custom",
            "tokenizer": "standard",
            "filter": ["lowercase", "aic_asciifolding"],
        }
    },
}

FRAME_MAPPINGS = {
    "dynamic": False,
    "properties": {
        "frame_uid": {"type": "keyword"},
        "video_id": {"type": "keyword"},
        "frame_id": {"type": "long"},
        "shot_id": {"type": "integer"},
        "pts_ms": {"type": "long"},
        "image_path": {"type": "keyword", "index": False},
        "image_sha256": {"type": "keyword"},
        "selection_encoder_id": {"type": "keyword"},
        "selection_version": {"type": "keyword"},
        "source_manifest": {"type": "keyword", "index": False},
        "ocr_text": {"type": "text", "analyzer": "aic_text"},
        "object_text": {"type": "text", "analyzer": "aic_text"},
        "object_classes": {"type": "keyword"},
        "ocr": {
            "type": "nested",
            "properties": {
                "stage": {"type": "keyword"},
                "model": {"type": "keyword"},
                "version": {"type": "keyword"},
                "status": {"type": "keyword"},
                "has_text": {"type": "boolean"},
                "raw_text": {"type": "text", "analyzer": "aic_text"},
                "normalized_text": {"type": "text", "analyzer": "aic_text"},
                "processed_at": {"type": "date"},
                "blocks": {
                    "type": "nested",
                    "properties": {
                        "block_id": {"type": "integer"},
                        "text": {"type": "text", "analyzer": "aic_text"},
                        "legibility": {"type": "keyword"},
                        "y_min": {"type": "integer"},
                        "x_min": {"type": "integer"},
                        "y_max": {"type": "integer"},
                        "x_max": {"type": "integer"},
                    },
                },
            },
        },
        "objects": {
            "type": "nested",
            "properties": {
                "model_id": {"type": "keyword"},
                "run_signature": {"type": "keyword"},
                "raw_detection_count": {"type": "integer"},
                "filtered_detection_count": {"type": "integer"},
                "object_signature": {"type": "keyword", "index": False},
                "processed_at": {"type": "date"},
                "detections": {
                    "type": "nested",
                    "properties": {
                        "detection_id": {"type": "integer"},
                        "class_id": {"type": "integer"},
                        "class_name": {"type": "keyword"},
                        "score": {"type": "float"},
                        "x_min": {"type": "float"},
                        "y_min": {"type": "float"},
                        "x_max": {"type": "float"},
                        "y_max": {"type": "float"},
                        "x_min_1000": {"type": "integer"},
                        "y_min_1000": {"type": "integer"},
                        "x_max_1000": {"type": "integer"},
                        "y_max_1000": {"type": "integer"},
                        "area_ratio": {"type": "float"},
                    },
                },
            },
        },
    },
}

ASR_MAPPINGS = {
    "dynamic": False,
    "properties": {
        "segment_id": {"type": "keyword"},
        "video_id": {"type": "keyword"},
        "level": {"type": "keyword"},
        "start_ms": {"type": "long"},
        "end_ms": {"type": "long"},
        "language": {"type": "keyword"},
        "source": {"type": "keyword"},
        "source_type": {"type": "keyword"},
        "provider": {"type": "keyword"},
        "model": {"type": "keyword"},
        "raw_text": {"type": "text", "analyzer": "aic_text"},
        "clean_text": {"type": "text", "analyzer": "aic_text"},
        "shot_ids": {"type": "integer"},
        "primary_shot": {"type": "integer"},
        "keyframe_ids": {"type": "long"},
        "keyframe_uids": {"type": "keyword"},
        "confidence": {"type": "float"},
        "chunk_id": {"type": "keyword"},
        "quality_score": {"type": "float"},
        "created_at": {"type": "date"},
        "source_manifest": {"type": "keyword", "index": False},
    },
}


def create_client(
    url: str = DEFAULT_ELASTICSEARCH_URL,
    *,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    verify_certs: bool = True,
) -> Any:
    try:
        from elasticsearch import Elasticsearch
    except ImportError as error:  # pragma: no cover - optional runtime.
        raise RuntimeError(
            "Elasticsearch support requires environments/serving."
        ) from error
    kwargs: dict[str, Any] = {"verify_certs": verify_certs}
    if api_key:
        kwargs["api_key"] = api_key
    elif username:
        kwargs["basic_auth"] = (username, password or "")
    return Elasticsearch(url, **kwargs)


def _exists(client: Any, index: str) -> bool:
    response = client.indices.exists(index=index)
    if hasattr(response, "body"):
        response = response.body
    return bool(response)


def ensure_index(
    client: Any,
    index: str,
    mappings: dict[str, Any],
    *,
    recreate: bool = False,
) -> None:
    exists = _exists(client, index)
    if recreate and exists:
        client.indices.delete(index=index)
        exists = False
    if not exists:
        client.indices.create(
            index=index,
            settings={"analysis": TEXT_ANALYSIS},
            mappings=mappings,
        )


def _batches(rows: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
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


def _bulk_index(
    client: Any,
    index: str,
    documents: Iterable[dict[str, Any]],
    *,
    id_field: str,
    batch_size: int,
) -> int:
    indexed = 0
    for batch in _batches(documents, batch_size):
        operations: list[dict[str, Any]] = []
        for document in batch:
            operations.append(
                {"index": {"_index": index, "_id": str(document[id_field])}}
            )
            operations.append(document)
        response = client.bulk(operations=operations, refresh=False)
        if hasattr(response, "body"):
            response = response.body
        errors = []
        for item in response.get("items", []):
            result = item.get("index", {})
            if result.get("error"):
                errors.append(result["error"])
        if response.get("errors") or errors:
            raise RuntimeError(f"Elasticsearch bulk indexing failed: {errors[:3]}")
        indexed += len(batch)
    return indexed


def index_artifacts(
    paths: PipelinePaths,
    *,
    client: Any,
    frame_index: str = DEFAULT_FRAME_INDEX,
    asr_index: str = DEFAULT_ASR_INDEX,
    selection_encoder_id: str = DEFAULT_SELECTION_ENCODER_ID,
    batch_size: int = 500,
    recreate: bool = False,
    refresh: bool = True,
) -> dict[str, Any]:
    if frame_index == asr_index:
        raise ValueError("frame and ASR indices must be different")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    ensure_index(client, frame_index, FRAME_MAPPINGS, recreate=recreate)
    ensure_index(client, asr_index, ASR_MAPPINGS, recreate=recreate)
    frame_count = _bulk_index(
        client,
        frame_index,
        iter_frame_documents(paths, selection_encoder_id=selection_encoder_id),
        id_field="frame_uid",
        batch_size=batch_size,
    )
    asr_count = _bulk_index(
        client,
        asr_index,
        iter_asr_documents(paths),
        id_field="segment_id",
        batch_size=batch_size,
    )
    if refresh:
        client.indices.refresh(index=f"{frame_index},{asr_index}")
    return {
        "frame_index": frame_index,
        "asr_index": asr_index,
        "selection_encoder_id": selection_encoder_id,
        "frame_documents": frame_count,
        "asr_documents": asr_count,
    }


def _hit_records(response: Any, source_type: str) -> list[dict[str, Any]]:
    if hasattr(response, "body"):
        response = response.body
    return [
        {
            "source_type": source_type,
            "id": str(hit["_id"]),
            "score": float(hit.get("_score") or 0.0),
            **dict(hit.get("_source") or {}),
        }
        for hit in response.get("hits", {}).get("hits", [])
    ]


def search_frames(
    client: Any,
    query: str,
    *,
    index: str = DEFAULT_FRAME_INDEX,
    video_id: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    if not query.strip() or limit <= 0:
        return []
    filters = [{"term": {"video_id": video_id}}] if video_id else []
    response = client.search(
        index=index,
        size=limit,
        query={
            "bool": {
                "filter": filters,
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["ocr_text^3", "object_text^2"],
                        }
                    },
                    {"term": {"object_classes": {"value": query, "boost": 4.0}}},
                ],
                "minimum_should_match": 1,
            }
        },
    )
    return _hit_records(response, "frame")


def search_asr(
    client: Any,
    query: str,
    *,
    index: str = DEFAULT_ASR_INDEX,
    video_id: str | None = None,
    level: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    if not query.strip() or limit <= 0:
        return []
    filters = []
    if video_id:
        filters.append({"term": {"video_id": video_id}})
    if level:
        if level not in {"L0", "L1"}:
            raise ValueError("level must be L0 or L1")
        filters.append({"term": {"level": level}})
    response = client.search(
        index=index,
        size=limit,
        query={
            "bool": {
                "filter": filters,
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["clean_text^2", "raw_text"],
                        }
                    }
                ],
            }
        },
    )
    return _hit_records(response, "asr")


def search_text(
    client: Any,
    query: str,
    *,
    frame_index: str = DEFAULT_FRAME_INDEX,
    asr_index: str = DEFAULT_ASR_INDEX,
    video_id: str | None = None,
    level: str | None = None,
    limit: int = 20,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "frames": search_frames(
            client,
            query,
            index=frame_index,
            video_id=video_id,
            limit=limit,
        ),
        "asr": search_asr(
            client,
            query,
            index=asr_index,
            video_id=video_id,
            level=level,
            limit=limit,
        ),
    }
