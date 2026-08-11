from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aic_pipeline.serving.elasticsearch import (
    DEFAULT_ELASTICSEARCH_URL,
    create_client,
)


def check_elasticsearch(client: Any, *, smoke_write: bool) -> dict[str, Any]:
    info = client.info()
    if hasattr(info, "body"):
        info = info.body
    health = client.cluster.health(wait_for_status="yellow", timeout="30s")
    if hasattr(health, "body"):
        health = health.body
    report = {
        "version": info.get("version", {}).get("number"),
        "cluster_name": info.get("cluster_name"),
        "status": health.get("status"),
    }
    if not smoke_write:
        return report

    index = f"aic-smoke-{uuid.uuid4().hex[:12]}"
    try:
        client.indices.create(
            index=index,
            mappings={
                "properties": {
                    "frame_uid": {"type": "keyword"},
                    "text": {"type": "text"},
                }
            },
        )
        client.index(
            index=index,
            id="L00_V000:000000001",
            document={"frame_uid": "L00_V000:000000001", "text": "aic smoke"},
            refresh="wait_for",
        )
        response = client.search(index=index, query={"match": {"text": "smoke"}})
        if hasattr(response, "body"):
            response = response.body
        hits = int(response.get("hits", {}).get("total", {}).get("value", 0))
        if hits != 1:
            raise RuntimeError(f"Elasticsearch smoke query returned {hits} hits")
        report["smoke_write"] = "ok"
    finally:
        if client.indices.exists(index=index):
            client.indices.delete(index=index)
    return report


def check_milvus(client: Any, *, smoke_write: bool) -> dict[str, Any]:
    collections = client.list_collections()
    report: dict[str, Any] = {"collections": collections}
    if not smoke_write:
        return report

    from pymilvus import DataType

    collection = f"aic_smoke_{uuid.uuid4().hex[:12]}"
    try:
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="frame_uid",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=128,
        )
        schema.add_field(
            field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=3
        )
        indexes = client.prepare_index_params()
        indexes.add_index(
            field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE"
        )
        client.create_collection(
            collection_name=collection,
            schema=schema,
            index_params=indexes,
        )
        client.insert(
            collection_name=collection,
            data=[
                {
                    "frame_uid": "L00_V000:000000001",
                    "embedding": [1.0, 0.0, 0.0],
                }
            ],
        )
        client.flush(collection_name=collection)
        response = client.search(
            collection_name=collection,
            data=[[1.0, 0.0, 0.0]],
            anns_field="embedding",
            limit=1,
            output_fields=["frame_uid"],
        )
        hit = response[0][0] if response and response[0] else None
        entity = hit.get("entity", {}) if hit is not None else {}
        hit_id = (
            hit.get("id") or hit.get("frame_uid") or entity.get("frame_uid")
            if hit is not None
            else None
        )
        if str(hit_id) != "L00_V000:000000001":
            raise RuntimeError(f"Unexpected Milvus smoke result: {response}")
        report["smoke_write"] = "ok"
    finally:
        if client.has_collection(collection_name=collection):
            client.drop_collection(collection_name=collection)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check local Elasticsearch and Milvus serving backends."
    )
    parser.add_argument(
        "--elasticsearch-url",
        default=os.getenv("ELASTICSEARCH_URL", DEFAULT_ELASTICSEARCH_URL),
    )
    parser.add_argument(
        "--milvus-uri", default=os.getenv("MILVUS_URI", "http://localhost:19530")
    )
    parser.add_argument("--smoke-write", action="store_true")
    args = parser.parse_args()

    elasticsearch_client = create_client(args.elasticsearch_url)
    try:
        from pymilvus import MilvusClient
    except ImportError as error:
        raise RuntimeError("Local serving checks require pymilvus") from error
    milvus_client = MilvusClient(uri=args.milvus_uri)
    report = {
        "elasticsearch": check_elasticsearch(
            elasticsearch_client, smoke_write=args.smoke_write
        ),
        "milvus": check_milvus(milvus_client, smoke_write=args.smoke_write),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
