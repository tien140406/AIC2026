from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from aic_pipeline.asr.models import TranscriptSegment, segments_to_frame
from aic_pipeline.common.identity import ensure_frame_identity, frame_uid
from aic_pipeline.common.paths import PipelinePaths
from aic_pipeline.serving.catalog import (
    build_serving_catalog,
    frame_bundle,
    search_objects,
    search_text,
)
from aic_pipeline.serving.milvus import ingest_embeddings


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
    (paths.shots / "V.json").write_text(
        json.dumps(
            {
                "detector": "omnishotcut",
                "ranges_end_exclusive": True,
                "ranges": [[0, 50], [50, 100]],
            }
        ),
        encoding="utf-8",
    )

    selected_dir = paths.keyframes / "timm_beit3_base_patch16_224_pt"
    selected_dir.mkdir(parents=True)
    selected = pd.DataFrame(
        [
            {"video_id": "V", "shot_id": 0, "frame_id": 10, "pts_time": 0.4},
            {"video_id": "V", "shot_id": 1, "frame_id": 60, "pts_time": 2.4},
        ]
    )
    selected.to_parquet(selected_dir / "V.parquet", index=False)
    image_dir = paths.workspace / "keyframe_images" / "V"
    image_dir.mkdir(parents=True)
    images = ensure_frame_identity(selected)
    images["image_path"] = images["frame_id"].map(
        lambda value: str(image_dir / f"frame_{value:09d}.jpg")
    )
    images["image_exists"] = True
    images.to_parquet(image_dir / "manifest.parquet", index=False)

    feature_dir = paths.features / "openclip-test"
    feature_dir.mkdir(parents=True)
    matrix = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float16)
    np.save(feature_dir / "V.f16.npy", matrix)
    mapping = ensure_frame_identity(selected)
    mapping.insert(0, "embedding_row", [0, 1])
    mapping.insert(1, "encoder_id", "openclip-test")
    mapping.to_parquet(feature_dir / "V.parquet", index=False)

    screening_dir = paths.workspace / "ocr/preprocess/V"
    screening_dir.mkdir(parents=True)
    screening = ensure_frame_identity(selected)
    screening["image_path"] = images["image_path"]
    screening["image_sha256"] = ["hash-10", "hash-60"]
    screening["preprocess_version"] = "screening-v1"
    screening["screening_has_text"] = [True, False]
    screening.to_parquet(screening_dir / "screening.parquet", index=False)

    ocr_dir = paths.workspace / "ocr/gemini/gemini-test/V"
    ocr_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "video_id": "V",
                "frame_id": 10,
                "shot_id": 0,
                "pts_time": 0.4,
                "model": "gemini-test",
                "prompt_version": "prompt-v1",
                "has_text": True,
                "raw_text": "AIC 2026",
                "normalized_text": "AIC 2026",
                "blocks_json": json.dumps(
                    [
                        {
                            "text": "AIC 2026",
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
    pd.DataFrame(
        [
            {
                "video_id": "V",
                "frame_id": 10,
                "shot_id": 0,
                "pts_time": 0.4,
                "model_id": "codetr-test",
                "run_signature": "object-run",
                "raw_detection_count": 1,
                "filtered_detection_count": 1,
                "object_signature": '{"person":1}',
                "processed_at": "2026-08-11T00:00:00+00:00",
            },
            {
                "video_id": "V",
                "frame_id": 60,
                "shot_id": 1,
                "pts_time": 2.4,
                "model_id": "codetr-test",
                "run_signature": "object-run",
                "raw_detection_count": 0,
                "filtered_detection_count": 0,
                "object_signature": "{}",
                "processed_at": "2026-08-11T00:00:00+00:00",
            },
        ]
    ).to_parquet(object_dir / "frames.parquet", index=False)
    pd.DataFrame(
        [
            {
                "video_id": "V",
                "frame_id": 10,
                "shot_id": 0,
                "pts_time": 0.4,
                "model_id": "codetr-test",
                "run_signature": "object-run",
                "detection_id": 0,
                "class_id": 0,
                "class_name": "person",
                "score": 0.9,
                "x_min": 1.0,
                "y_min": 2.0,
                "x_max": 100.0,
                "y_max": 200.0,
                "x_min_1000": 1,
                "y_min_1000": 2,
                "x_max_1000": 100,
                "y_max_1000": 200,
                "area_ratio": 0.1,
            }
        ]
    ).to_parquet(object_dir / "detections.parquet", index=False)

    transcript_dir = paths.transcripts / "V"
    transcript_dir.mkdir(parents=True)
    segments_to_frame(
        [
            TranscriptSegment(
                segment_id="V:L0:000000",
                video_id="V",
                start=0.2,
                end=1.0,
                raw_text="fresh tomato",
                clean_text="fresh tomato",
                keyframe_ids=(10, 60),
                primary_shot=0,
            )
        ]
    ).to_parquet(transcript_dir / "l0.parquet", index=False)
    return paths


def test_frame_identity_is_stable_and_adds_integer_timestamp() -> None:
    frame = ensure_frame_identity(
        pd.DataFrame([{"video_id": "V", "frame_id": 10, "pts_time": 0.4}])
    )

    assert frame.iloc[0]["frame_uid"] == "V:000000010"
    assert frame.iloc[0]["pts_ms"] == 400


def test_build_serving_catalog_normalizes_all_modalities(tmp_path: Path) -> None:
    paths = prepare_artifacts(tmp_path)
    destination = paths.serving / "catalog.sqlite"

    report = build_serving_catalog(paths, destination)

    assert report["keyframes"] == 2
    assert report["embeddings"] == 2
    assert report["ocr_blocks"] == 1
    assert report["object_detections"] == 1
    assert report["asr_segments"] == 1
    assert report["segment_keyframes"] == 2
    with sqlite3.connect(destination) as connection:
        bundle = frame_bundle(connection, frame_uid("V", 10))
        assert bundle is not None
        assert bundle["ocr_blocks"][0]["text"] == "AIC 2026"
        assert bundle["object_detections"][0]["class_name"] == "person"
        assert bundle["asr_segments"][0]["segment_id"] == "V:L0:000000"
        assert search_text(connection, "AIC")[0]["source_type"] == "ocr"
        assert search_text(connection, "tomato")[0]["source_type"] == "asr"
        assert search_objects(connection, "person")[0]["frame_uid"] == frame_uid(
            "V", 10
        )


class FakeDataType:
    VARCHAR = "varchar"
    INT64 = "int64"
    BOOL = "bool"
    ARRAY = "array"
    FLOAT_VECTOR = "float_vector"


class FakeSchema:
    def __init__(self) -> None:
        self.fields: list[dict[str, object]] = []

    def add_field(self, **kwargs) -> None:
        self.fields.append(kwargs)


class FakeIndexes:
    def add_index(self, **kwargs) -> None:
        self.index = kwargs


class FakeMilvusClient:
    DataType = FakeDataType

    def __init__(self) -> None:
        self.created = False
        self.rows: list[dict[str, object]] = []

    def has_collection(self, **kwargs) -> bool:
        return self.created

    def drop_collection(self, **kwargs) -> None:
        self.created = False

    def create_schema(self, **kwargs) -> FakeSchema:
        self.schema = FakeSchema()
        return self.schema

    def prepare_index_params(self) -> FakeIndexes:
        return FakeIndexes()

    def create_collection(self, **kwargs) -> None:
        self.created = True

    def upsert(self, *, collection_name: str, data: list[dict[str, object]]) -> None:
        self.rows.extend(data)

    def flush(self, **kwargs) -> None:
        return None


def test_milvus_ingestion_uses_frame_uid_and_catalog_vectors(tmp_path: Path) -> None:
    paths = prepare_artifacts(tmp_path)
    destination = paths.serving / "catalog.sqlite"
    build_serving_catalog(paths, destination)
    client = FakeMilvusClient()

    report = ingest_embeddings(
        destination,
        tmp_path,
        model="openclip-test",
        version="openclip-test",
        uri="unused",
        client=client,
    )

    assert report["rows"] == 2
    assert client.rows[0]["frame_uid"] == "V:000000010"
    assert client.rows[0]["embedding"] == [1.0, 0.0, 0.0]
    assert client.rows[0]["object_class_ids"] == [0]
