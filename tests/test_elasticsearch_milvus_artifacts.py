from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aic_pipeline.asr.models import TranscriptSegment, segments_to_frame
from aic_pipeline.common.identity import ensure_frame_identity, frame_uid, roi_uid
from aic_pipeline.common.paths import PipelinePaths
from aic_pipeline.serving.artifacts import iter_asr_documents, iter_frame_documents
from aic_pipeline.serving.catalog import build_serving_catalog
from aic_pipeline.serving.elasticsearch import index_artifacts, search_text
from aic_pipeline.serving.hybrid import reciprocal_rank_fusion
from aic_pipeline.serving.milvus_artifacts import ingest_artifact_embeddings


def prepare_artifacts(tmp_path: Path) -> PipelinePaths:
    paths = PipelinePaths.from_root(tmp_path)
    paths.ensure_output_dirs()
    pd.DataFrame(
        [
            {
                "video_id": "V",
                "group": "L21",
                "video_path": str(tmp_path / "archive/video/V.mp4"),
                "fps": 25.0,
                "frame_count": 100,
                "duration_seconds": 4.0,
                "width": 1920,
                "height": 1080,
            }
        ]
    ).to_parquet(paths.manifests / "videos.parquet", index=False)
    selected_dir = paths.keyframes / "timm_beit3_base_patch16_224_pt"
    selected_dir.mkdir(parents=True)
    selected = pd.DataFrame(
        [
            {
                "video_id": "V",
                "frame_id": 10,
                "shot_id": 1,
                "pts_time": 0.4,
            }
        ]
    )
    selected.to_parquet(selected_dir / "V.parquet", index=False)
    image_dir = paths.workspace / "keyframe_images" / "V"
    image_dir.mkdir(parents=True)
    images = ensure_frame_identity(selected)
    images["image_path"] = str(image_dir / "frame_000000010.jpg")
    images["image_sha256"] = "image-hash"
    images.to_parquet(image_dir / "manifest.parquet", index=False)

    ocr_dir = paths.workspace / "ocr/gemini/gemini-test/V"
    ocr_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "video_id": "V",
                "frame_id": 10,
                "shot_id": 1,
                "pts_time": 0.4,
                "model": "gemini-test",
                "prompt_version": "prompt-v1",
                "has_text": True,
                "raw_text": "Ca chua tuoi",
                "normalized_text": "ca chua tuoi",
                "blocks_json": json.dumps(
                    [
                        {
                            "text": "Ca chua tuoi",
                            "bbox_2d": [10, 20, 30, 40],
                            "legibility": "clear",
                        }
                    ]
                ),
                "processed_at": "2026-08-11T00:00:00+00:00",
            }
        ]
    ).to_parquet(ocr_dir / "frames.parquet", index=False)

    object_dir = paths.workspace / "objects/codetr-test/V"
    object_dir.mkdir(parents=True)
    object_frame = ensure_frame_identity(selected)
    object_frame["model_id"] = "codetr-test"
    object_frame["run_signature"] = "run-v1"
    object_frame["raw_detection_count"] = 1
    object_frame["filtered_detection_count"] = 1
    object_frame["object_signature"] = '{"person":1}'
    object_frame["processed_at"] = "2026-08-11T00:00:00+00:00"
    object_frame.to_parquet(object_dir / "frames.parquet", index=False)
    detection = object_frame[
        ["video_id", "frame_uid", "frame_id", "shot_id", "pts_time", "pts_ms"]
    ].copy()
    detection["model_id"] = "codetr-test"
    detection["run_signature"] = "run-v1"
    detection["detection_id"] = 0
    detection["class_id"] = 0
    detection["class_name"] = "person"
    detection["score"] = 0.9
    detection["x_min"] = 10.0
    detection["y_min"] = 20.0
    detection["x_max"] = 100.0
    detection["y_max"] = 200.0
    detection["x_min_1000"] = 5
    detection["y_min_1000"] = 19
    detection["x_max_1000"] = 52
    detection["y_max_1000"] = 185
    detection["area_ratio"] = 0.01
    detection.to_parquet(object_dir / "detections.parquet", index=False)

    transcript_dir = paths.transcripts / "V"
    transcript_dir.mkdir(parents=True)
    segments_to_frame(
        [
            TranscriptSegment(
                segment_id="V:L0:000000",
                video_id="V",
                start=0.2,
                end=1.0,
                raw_text="ca chua tuoi",
                clean_text="ca chua tuoi",
                keyframe_ids=(10,),
                shot_ids=(1,),
                primary_shot=1,
            )
        ]
    ).to_parquet(transcript_dir / "l0.parquet", index=False)

    keyframe_feature_dir = paths.features / "openclip-test"
    keyframe_feature_dir.mkdir(parents=True)
    np.save(
        keyframe_feature_dir / "V.f16.npy",
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float16),
    )
    keyframe_mapping = ensure_frame_identity(selected)
    keyframe_mapping.insert(0, "embedding_row", [0])
    keyframe_mapping.insert(1, "encoder_id", "openclip-test")
    keyframe_mapping.to_parquet(keyframe_feature_dir / "V.parquet", index=False)

    roi_feature_dir = paths.features / "openclip-test-roi"
    roi_feature_dir.mkdir(parents=True)
    np.save(
        roi_feature_dir / "V.f16.npy",
        np.asarray([[0.0, 1.0, 0.0]], dtype=np.float16),
    )
    roi_mapping = detection.copy()
    roi_mapping.insert(0, "embedding_row", [0])
    roi_mapping.insert(1, "encoder_id", "openclip-test-roi")
    roi_mapping.insert(2, "embedding_model", "openclip-test")
    roi_mapping.insert(3, "embedding_version", "openclip-test")
    roi_mapping.insert(4, "entity_type", "roi")
    roi_mapping.insert(5, "parent_frame_uid", roi_mapping["frame_uid"])
    roi_mapping["roi_uid"] = roi_uid(
        frame_uid("V", 10), "codetr-test", "run-v1", 0
    )
    roi_mapping.to_parquet(roi_feature_dir / "V.parquet", index=False)
    return paths


class FakeIndices:
    def __init__(self) -> None:
        self.names: set[str] = set()
        self.created: dict[str, dict[str, object]] = {}

    def exists(self, *, index: str) -> bool:
        return index in self.names

    def create(self, *, index: str, **kwargs) -> None:
        self.names.add(index)
        self.created[index] = kwargs

    def delete(self, *, index: str) -> None:
        self.names.remove(index)

    def refresh(self, **kwargs) -> None:
        return None


class FakeElasticsearch:
    def __init__(self) -> None:
        self.indices = FakeIndices()
        self.documents: dict[tuple[str, str], dict[str, object]] = {}
        self.searches: list[dict[str, object]] = []

    def bulk(self, *, operations, refresh=False):
        items = []
        for offset in range(0, len(operations), 2):
            action = operations[offset]["index"]
            document = operations[offset + 1]
            self.documents[(action["_index"], action["_id"])] = document
            items.append({"index": {"status": 201}})
        return {"errors": False, "items": items}

    def search(self, **kwargs):
        self.searches.append(kwargs)
        return {"hits": {"hits": []}}


def test_elasticsearch_documents_preserve_frame_and_segment_granularity(
    tmp_path: Path,
) -> None:
    paths = prepare_artifacts(tmp_path)

    frame_document = next(iter_frame_documents(paths))
    asr_document = next(iter_asr_documents(paths))

    assert frame_document["frame_uid"] == "V:000000010"
    assert frame_document["ocr_text"] == "ca chua tuoi"
    assert frame_document["object_classes"] == ["person"]
    assert frame_document["objects"][0]["detections"][0]["score"] == 0.9
    assert asr_document["segment_id"] == "V:L0:000000"
    assert asr_document["keyframe_uids"] == ["V:000000010"]


def test_elasticsearch_indexing_is_idempotent_by_document_id(tmp_path: Path) -> None:
    paths = prepare_artifacts(tmp_path)
    client = FakeElasticsearch()

    first = index_artifacts(paths, client=client)
    second = index_artifacts(paths, client=client)

    assert first["frame_documents"] == second["frame_documents"] == 1
    assert first["asr_documents"] == second["asr_documents"] == 1
    assert len(client.documents) == 2
    search_text(client, "ca chua", video_id="V", level="L0")
    assert len(client.searches) == 2


class FakeDataType:
    VARCHAR = "varchar"
    INT64 = "int64"
    FLOAT = "float"
    FLOAT_VECTOR = "float_vector"


class FakeSchema:
    def __init__(self) -> None:
        self.fields = []

    def add_field(self, **kwargs) -> None:
        self.fields.append(kwargs)


class FakeIndexes:
    def add_index(self, **kwargs) -> None:
        self.index = kwargs


class FakeMilvus:
    DataType = FakeDataType

    def __init__(self) -> None:
        self.created = False
        self.rows = []

    def has_collection(self, **kwargs) -> bool:
        return self.created

    def drop_collection(self, **kwargs) -> None:
        self.created = False

    def create_schema(self, **kwargs):
        self.schema = FakeSchema()
        return self.schema

    def prepare_index_params(self):
        return FakeIndexes()

    def create_collection(self, **kwargs) -> None:
        self.created = True

    def upsert(self, *, collection_name, data) -> None:
        self.rows.extend(data)

    def flush(self, **kwargs) -> None:
        return None


def test_direct_milvus_ingestion_supports_keyframes_and_rois(tmp_path: Path) -> None:
    paths = prepare_artifacts(tmp_path)
    keyframe_client = FakeMilvus()
    roi_client = FakeMilvus()

    keyframes = ingest_artifact_embeddings(
        paths,
        encoder_id="openclip-test",
        entity_type="keyframe",
        uri="unused",
        client=keyframe_client,
    )
    rois = ingest_artifact_embeddings(
        paths,
        encoder_id="openclip-test-roi",
        entity_type="roi",
        uri="unused",
        client=roi_client,
    )

    assert keyframes["rows"] == 1
    assert keyframe_client.rows[0]["frame_uid"] == "V:000000010"
    assert rois["rows"] == 1
    assert roi_client.rows[0]["parent_frame_uid"] == "V:000000010"
    assert roi_client.rows[0]["class_name"] == "person"

    compatibility = build_serving_catalog(
        paths, paths.serving / "catalog.sqlite"
    )
    assert compatibility["embeddings"] == 1


def test_rrf_collapses_roi_and_asr_evidence_to_parent_frames() -> None:
    results = reciprocal_rank_fusion(
        {
            "milvus_keyframe": [
                {"frame_uid": "V:000000010", "score": 0.9, "id": "frame"}
            ],
            "milvus_roi": [
                {
                    "roi_uid": "roi-1",
                    "parent_frame_uid": "V:000000010",
                    "score": 0.8,
                    "id": "roi-1",
                }
            ],
            "elasticsearch_frame": [
                {"frame_uid": "V:000000020", "score": 2.0, "id": "frame-20"}
            ],
            "elasticsearch_asr": [
                {
                    "segment_id": "V:L0:000000",
                    "keyframe_uids": ["V:000000010", "V:000000020"],
                    "score": 3.0,
                    "id": "segment",
                }
            ],
        },
        limit=10,
    )

    assert [item["frame_uid"] for item in results] == [
        "V:000000010",
        "V:000000020",
    ]
    assert {item["source"] for item in results[0]["evidence"]} == {
        "milvus_keyframe",
        "milvus_roi",
        "elasticsearch_asr",
    }
