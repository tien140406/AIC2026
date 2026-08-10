import pandas as pd
from PIL import Image

from aic_pipeline.ocr.preprocess import (
    TextRegion,
    resize_for_screening,
    screen_image,
    screen_jobs,
)


class FakeDetector:
    def __init__(self, regions: list[TextRegion]) -> None:
        self.regions = regions
        self.calls = 0

    def detect(self, image: Image.Image) -> list[TextRegion]:
        self.calls += 1
        return self.regions


def test_resize_for_screening_preserves_aspect_ratio() -> None:
    image = Image.new("RGB", (1280, 720), "white")

    resized = resize_for_screening(image, max_side=960)

    assert resized.size == (960, 540)
    assert image.size == (1280, 720)


def test_screen_image_uses_the_full_screening_image() -> None:
    detector = FakeDetector(
        [
            TextRegion((30, 810, 120, 930), 0.99),
            TextRegion((700, 100, 760, 450), 0.95),
        ]
    )

    result = screen_image(
        Image.new("RGB", (1280, 720), "white"),
        detector=detector,
    )

    assert result.has_text
    assert [region.bbox for region in result.regions] == [
        (30, 810, 120, 930),
        (700, 100, 760, 450),
    ]


def test_screen_jobs_reuses_matching_manifest(tmp_path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (128, 72), "white").save(image_path)
    jobs = pd.DataFrame(
        [
            {
                "video_id": "L21_V001",
                "frame_id": 10,
                "image_path": str(image_path),
                "image_sha256": "hash-1",
            }
        ]
    )
    detector = FakeDetector([TextRegion((350, 100, 550, 400), 0.9)])

    first, first_report = screen_jobs(
        jobs,
        video_id="L21_V001",
        output_dir=tmp_path / "screening",
        detector=detector,
    )
    second_detector = FakeDetector([])
    second, second_report = screen_jobs(
        jobs,
        video_id="L21_V001",
        output_dir=tmp_path / "screening",
        detector=second_detector,
    )

    assert bool(first.loc[0, "screening_has_text"])
    assert first.loc[0, "screening_signature"] == second.loc[0, "screening_signature"]
    assert first_report["reused_frames"] == 0
    assert second_report["reused_frames"] == 1
    assert detector.calls == 1
    assert second_detector.calls == 0
