from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import numpy as np

from .elasticsearch import (
    DEFAULT_ASR_INDEX,
    DEFAULT_FRAME_INDEX,
    search_text,
)


def _milvus_hits(response: Any, source_type: str) -> list[dict[str, Any]]:
    if not response:
        return []
    rows = response[0] if isinstance(response, list) else response
    hits = []
    for row in rows:
        entity = dict(row.get("entity") or {})
        hits.append(
            {
                "source_type": source_type,
                "id": str(row.get("id") or entity.get("frame_uid") or entity.get("roi_uid")),
                "score": float(row.get("distance") or row.get("score") or 0.0),
                **entity,
            }
        )
    return hits


def search_milvus(
    client: Any,
    query_vector: np.ndarray,
    *,
    keyframe_collection: str,
    roi_collection: str,
    video_id: str | None = None,
    limit: int = 100,
) -> dict[str, list[dict[str, Any]]]:
    if limit <= 0:
        return {"keyframes": [], "rois": []}
    vector = np.asarray(query_vector, dtype=np.float32).reshape(-1).tolist()
    filter_expression = f"video_id == {json.dumps(video_id)}" if video_id else ""

    def run(collection: str, output_fields: list[str], source_type: str):
        if not client.has_collection(collection_name=collection):
            return []
        response = client.search(
            collection_name=collection,
            data=[vector],
            anns_field="embedding",
            filter=filter_expression,
            limit=limit,
            output_fields=output_fields,
        )
        return _milvus_hits(response, source_type)

    return {
        "keyframes": run(
            keyframe_collection,
            ["frame_uid", "video_id", "frame_id", "shot_id", "pts_ms", "image_path"],
            "milvus_keyframe",
        ),
        "rois": run(
            roi_collection,
            [
                "roi_uid",
                "parent_frame_uid",
                "video_id",
                "frame_id",
                "shot_id",
                "pts_ms",
                "class_id",
                "class_name",
                "confidence",
                "x_min_1000",
                "y_min_1000",
                "x_max_1000",
                "y_max_1000",
            ],
            "milvus_roi",
        ),
    }


def reciprocal_rank_fusion(
    result_sets: dict[str, list[dict[str, Any]]],
    *,
    rrf_k: int = 60,
    limit: int = 20,
) -> list[dict[str, Any]]:
    if rrf_k <= 0 or limit <= 0:
        raise ValueError("RRF k and limit must be positive")
    scores: dict[str, float] = defaultdict(float)
    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metadata: dict[str, dict[str, Any]] = {}

    for source_name, hits in result_sets.items():
        seen_in_source: set[str] = set()
        for rank, hit in enumerate(hits, start=1):
            if source_name == "milvus_roi":
                frame_uids = [hit.get("parent_frame_uid")]
            elif source_name == "elasticsearch_asr":
                frame_uids = list(hit.get("keyframe_uids") or [])
            else:
                frame_uids = [hit.get("frame_uid")]
            for uid_value in frame_uids:
                if not uid_value:
                    continue
                uid = str(uid_value)
                source_key = f"{source_name}:{uid}"
                if source_key in seen_in_source:
                    continue
                seen_in_source.add(source_key)
                scores[uid] += 1.0 / (rrf_k + rank)
                evidence[uid].append(
                    {
                        "source": source_name,
                        "rank": rank,
                        "score": float(hit.get("score") or 0.0),
                        "id": hit.get("id"),
                        "segment_id": hit.get("segment_id"),
                        "roi_uid": hit.get("roi_uid"),
                        "class_name": hit.get("class_name"),
                    }
                )
                current = metadata.setdefault(uid, {"frame_uid": uid})
                for field in ("video_id", "frame_id", "shot_id", "pts_ms", "image_path"):
                    if current.get(field) is None and hit.get(field) is not None:
                        current[field] = hit[field]

    ranked = sorted(scores, key=lambda uid: (-scores[uid], uid))[:limit]
    return [
        {
            **metadata[uid],
            "rrf_score": scores[uid],
            "evidence": evidence[uid],
        }
        for uid in ranked
    ]


def hydrate_frame_results(
    client: Any,
    results: list[dict[str, Any]],
    *,
    index: str = DEFAULT_FRAME_INDEX,
) -> list[dict[str, Any]]:
    if not results or not hasattr(client, "mget"):
        return results
    response = client.mget(index=index, ids=[item["frame_uid"] for item in results])
    if hasattr(response, "body"):
        response = response.body
    documents = {
        str(item["_id"]): dict(item.get("_source") or {})
        for item in response.get("docs", [])
        if item.get("found")
    }
    hydrated = []
    for result in results:
        document = documents.get(str(result["frame_uid"]), {})
        merged = dict(result)
        for field in ("video_id", "frame_id", "shot_id", "pts_ms", "image_path"):
            if merged.get(field) is None and document.get(field) is not None:
                merged[field] = document[field]
        merged["frame"] = document or None
        hydrated.append(merged)
    return hydrated


def hybrid_search(
    *,
    elasticsearch_client: Any,
    milvus_client: Any,
    query: str,
    query_vector: np.ndarray,
    keyframe_collection: str,
    roi_collection: str,
    frame_index: str = DEFAULT_FRAME_INDEX,
    asr_index: str = DEFAULT_ASR_INDEX,
    video_id: str | None = None,
    level: str | None = None,
    candidate_limit: int = 100,
    limit: int = 20,
    rrf_k: int = 60,
) -> dict[str, Any]:
    text_results = search_text(
        elasticsearch_client,
        query,
        frame_index=frame_index,
        asr_index=asr_index,
        video_id=video_id,
        level=level,
        limit=candidate_limit,
    )
    vector_results = search_milvus(
        milvus_client,
        query_vector,
        keyframe_collection=keyframe_collection,
        roi_collection=roi_collection,
        video_id=video_id,
        limit=candidate_limit,
    )
    result_sets = {
        "milvus_keyframe": vector_results["keyframes"],
        "milvus_roi": vector_results["rois"],
        "elasticsearch_frame": text_results["frames"],
        "elasticsearch_asr": text_results["asr"],
    }
    fused = reciprocal_rank_fusion(result_sets, rrf_k=rrf_k, limit=limit)
    return {
        "query": query,
        "video_id": video_id,
        "collections": {
            "keyframes": keyframe_collection,
            "rois": roi_collection,
        },
        "indices": {"frames": frame_index, "asr": asr_index},
        "candidates": {name: len(rows) for name, rows in result_sets.items()},
        "results": hydrate_frame_results(
            elasticsearch_client, fused, index=frame_index
        ),
    }
