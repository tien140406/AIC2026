import json

import numpy as np
import pandas as pd
from PIL import Image

from aic_pipeline.objects import pipeline as runner
from aic_pipeline.objects.codetr import flatten_codetr_result
from aic_pipeline.objects.filtering import filter_detections, write_summary_tables
from aic_pipeline.objects.pipeline import run_pipeline
from aic_pipeline.objects.schema import (
    SCHEMA_VERSION,
    normalize_detection,
    valid_raw_payload,
)


def test_normalize_detection_clips_box_and_builds_ocr_style_coordinates() -> None:
    detection = normalize_detection(
        class_id=0,
        class_name="person",
        score=0.9,
        bbox_xyxy=[-10, 20, 210, 120],
        image_width=200,
        image_height=100,
    )

    assert detection is not None
    assert detection["bbox_xyxy"] == [0.0, 20.0, 200.0, 100.0]
    assert detection["bbox_yxyx_1000"] == [200, 0, 1000, 1000]
    assert detection["area_ratio"] == 0.8


def test_flatten_codetr_result_maps_class_arrays() -> None:
    result = [
        np.asarray([[1, 2, 11, 22, 0.9]], dtype=np.float32),
        np.empty((0, 5), dtype=np.float32),
    ]

    detections = flatten_codetr_result(result, ("person", "car"))

    assert len(detections) == 1
    assert detections[0]["class_name"] == "person"
    assert detections[0]["bbox_xyxy"] == [1.0, 2.0, 11.0, 22.0]


def test_filter_detections_removes_low_score_small_and_duplicate_boxes() -> None:
    detections = [
        {
            "class_id": 0,
            "class_name": "person",
            "score": 0.95,
            "bbox_xyxy": [0, 0, 100, 100],
            "bbox_yxyx_1000": [0, 0, 500, 500],
            "area_ratio": 0.25,
        },
        {
            "class_id": 0,
            "class_name": "person",
            "score": 0.80,
            "bbox_xyxy": [1, 1, 100, 100],
            "bbox_yxyx_1000": [5, 5, 500, 500],
            "area_ratio": 0.245,
        },
        {
            "class_id": 2,
            "class_name": "car",
            "score": 0.29,
            "bbox_xyxy": [120, 20, 180, 80],
            "bbox_yxyx_1000": [100, 600, 400, 900],
            "area_ratio": 0.09,
        },
        {
            "class_id": 16,
            "class_name": "dog",
            "score": 0.90,
            "bbox_xyxy": [150, 80, 151, 81],
            "bbox_yxyx_1000": [400, 750, 405, 755],
            "area_ratio": 0.00001,
        },
    ]

    filtered = filter_detections(detections)

    assert [item["class_name"] for item in filtered] == ["person"]


def test_summary_tables_keep_frames_without_objects(tmp_path) -> None:
    output_dir = tmp_path / "objects"
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "video_id": "L21_V001",
        "shot_id": 1,
        "frame_id": 10,
        "pts_time": 0.4,
        "image_path": "/tmp/frame.jpg",
        "image_sha256": "hash",
        "image_width": 1920,
        "image_height": 1080,
        "model_id": "codetr-test",
        "run_signature": "signature",
        "detections": [],
        "processed_at": "2026-08-11T00:00:00+00:00",
    }
    (raw_dir / "frame_000000010.json").write_text(json.dumps(payload), encoding="utf-8")

    frames, detections, report = write_summary_tables(output_dir)

    assert frames["frame_id"].tolist() == [10]
    assert frames["filtered_detection_count"].tolist() == [0]
    assert detections.empty
    assert report["frames_with_objects"] == 0


def test_raw_payload_requires_normalized_detection_coordinates() -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "video_id": "L21_V001",
        "frame_id": 10,
        "image_path": "/tmp/frame.jpg",
        "image_sha256": "hash",
        "image_width": 1920,
        "image_height": 1080,
        "model_id": "codetr-test",
        "run_signature": "signature",
        "processed_at": "2026-08-11T00:00:00+00:00",
        "detections": [
            {
                "class_id": 0,
                "class_name": "person",
                "score": 0.9,
                "bbox_xyxy": [10.0, 20.0, 100.0, 200.0],
                "area_ratio": 0.01,
            }
        ],
    }

    assert not valid_raw_payload(payload)
    payload["detections"][0]["bbox_yxyx_1000"] = [19, 5, 185, 52]
    assert valid_raw_payload(payload)


def test_pipeline_processes_final_keyframes_and_resumes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = []
    for frame_id in (10, 20):
        image_path = image_dir / f"frame_{frame_id}.jpg"
        Image.new("RGB", (200, 100), "white").save(image_path)
        rows.append(
            {
                "video_id": "L21_V001",
                "shot_id": frame_id // 10,
                "frame_id": frame_id,
                "pts_time": frame_id / 25,
                "image_path": str(image_path),
                "image_exists": True,
            }
        )
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    config = tmp_path / "config.py"
    checkpoint = tmp_path / "model.pth"
    config.write_text("model = {}\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    output_dir = tmp_path / "output"

    class FakeDetector:
        calls = 0

        def predict(self, image_path):
            self.calls += 1
            return [
                {
                    "class_id": 0,
                    "class_name": "person",
                    "score": 0.9,
                    "bbox_xyxy": [10, 10, 100, 90],
                }
            ]

        def peak_vram_mb(self):
            return 123.0

    detector = FakeDetector()
    factory = lambda **kwargs: detector
    report = run_pipeline(
        video_id="L21_V001",
        manifest_path=manifest,
        model_id="codetr-test",
        config_path=config,
        checkpoint_path=checkpoint,
        codetr_root=tmp_path,
        output_dir=output_dir,
        device="cuda:0",
        fp16=False,
        raw_score_floor=0.05,
        score_threshold=0.30,
        min_area_ratio=0.0005,
        duplicate_iou=0.90,
        max_detections=100,
        hash_workers=1,
        checkpoint_every=10,
        progress_every=10,
        detector_factory=factory,
    )

    assert detector.calls == 2
    assert report["selected_successes"] == 2
    assert report["filtered_detections"] == 2
    assert (output_dir / "frames.parquet").is_file()
    assert (output_dir / "detections.parquet").is_file()

    resumed = run_pipeline(
        video_id="L21_V001",
        manifest_path=manifest,
        model_id="codetr-test",
        config_path=config,
        checkpoint_path=checkpoint,
        codetr_root=tmp_path,
        output_dir=output_dir,
        device="cuda:0",
        fp16=False,
        raw_score_floor=0.05,
        score_threshold=0.50,
        min_area_ratio=0.0005,
        duplicate_iou=0.90,
        max_detections=100,
        hash_workers=1,
        checkpoint_every=10,
        progress_every=10,
        detector_factory=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Detector should not load when every frame is reusable")
        ),
    )

    assert resumed["reusable_successes"] == 2
    assert resumed["selected_successes"] == 2
