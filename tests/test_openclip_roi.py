from pathlib import Path

import pandas as pd

from aic_pipeline.common.identity import ensure_frame_identity
from aic_pipeline.embeddings.openclip_roi import expand_bbox_xyxy, load_roi_jobs


def test_expand_bbox_adds_context_and_clips_to_image() -> None:
    assert expand_bbox_xyxy(
        (5, 10, 95, 90),
        image_width=100,
        image_height=100,
        padding_ratio=0.10,
    ) == (0, 2, 100, 98)


def test_load_roi_jobs_filters_and_assigns_stable_identity(tmp_path: Path) -> None:
    detections = pd.DataFrame(
        [
            {
                "video_id": "V",
                "frame_id": 10,
                "shot_id": 1,
                "pts_time": 0.4,
                "model_id": "codetr-test",
                "run_signature": "run-v1",
                "detection_id": 0,
                "class_id": 0,
                "class_name": "person",
                "score": 0.9,
                "x_min": 10.0,
                "y_min": 20.0,
                "x_max": 100.0,
                "y_max": 200.0,
                "x_min_1000": 5,
                "y_min_1000": 19,
                "x_max_1000": 52,
                "y_max_1000": 185,
                "area_ratio": 0.01,
            },
            {
                "video_id": "V",
                "frame_id": 10,
                "shot_id": 1,
                "pts_time": 0.4,
                "model_id": "codetr-test",
                "run_signature": "run-v1",
                "detection_id": 1,
                "class_id": 2,
                "class_name": "car",
                "score": 0.1,
                "x_min": 1.0,
                "y_min": 2.0,
                "x_max": 3.0,
                "y_max": 4.0,
                "x_min_1000": 1,
                "y_min_1000": 2,
                "x_max_1000": 3,
                "y_max_1000": 4,
                "area_ratio": 0.0001,
            },
        ]
    )
    detections_path = tmp_path / "detections.parquet"
    detections.to_parquet(detections_path, index=False)
    images = ensure_frame_identity(
        pd.DataFrame(
            [
                {
                    "video_id": "V",
                    "frame_id": 10,
                    "pts_time": 0.4,
                    "image_path": str(tmp_path / "frame.jpg"),
                }
            ]
        )
    )
    manifest_path = tmp_path / "manifest.parquet"
    images.to_parquet(manifest_path, index=False)

    jobs = load_roi_jobs(
        detections_path,
        manifest_path,
        video_id="V",
        min_score=0.3,
        min_area_ratio=0.0005,
    )

    assert len(jobs) == 1
    assert jobs.iloc[0]["class_name"] == "person"
    assert jobs.iloc[0]["roi_uid"].startswith("V:000000010:roi:")
