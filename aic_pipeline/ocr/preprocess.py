from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from PIL import Image, ImageOps

from ..common.io import atomic_write_json, atomic_write_parquet


PREPROCESS_VERSION = "gemini_screening_v2"
DEFAULT_MAX_SIDE = 960
DEFAULT_MIN_SCORE = 0.5
DEFAULT_MIN_BOX_HEIGHT = 8
DEFAULT_SCREENING_MODEL = "PP-OCRv5_mobile_det"


@dataclass(frozen=True)
class TextRegion:
    """A normalized [y_min, x_min, y_max, x_max] detector box."""

    bbox: tuple[int, int, int, int]
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {"bbox": list(self.bbox), "score": round(self.score, 6)}


@dataclass(frozen=True)
class ScreeningResult:
    has_text: bool
    regions: tuple[TextRegion, ...]
    image_width: int
    image_height: int

    def as_record(self) -> dict[str, Any]:
        return {
            "preprocess_version": PREPROCESS_VERSION,
            "screening_has_text": self.has_text,
            "screening_boxes_json": json.dumps(
                [region.as_dict() for region in self.regions], ensure_ascii=False
            ),
            "screening_width": self.image_width,
            "screening_height": self.image_height,
        }


class TextDetector(Protocol):
    def detect(self, image: Image.Image) -> list[TextRegion]:
        """Return text boxes in normalized [y_min, x_min, y_max, x_max] form."""


def resize_for_screening(
    image: Image.Image,
    max_side: int = DEFAULT_MAX_SIDE,
) -> Image.Image:
    if max_side <= 0:
        raise ValueError("max_side must be positive")
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    longest_side = max(width, height)
    if longest_side <= max_side:
        return image.copy()
    scale = max_side / longest_side
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def filter_text_regions(
    regions: list[TextRegion],
    *,
    min_score: float = DEFAULT_MIN_SCORE,
    min_box_height: int = DEFAULT_MIN_BOX_HEIGHT,
    image_height: int = 1000,
) -> list[TextRegion]:
    if not 0 <= min_score <= 1:
        raise ValueError("min_score must be between 0 and 1")
    if min_box_height < 0:
        raise ValueError("min_box_height must be non-negative")
    if image_height <= 0:
        raise ValueError("image_height must be positive")

    filtered: list[TextRegion] = []
    for region in regions:
        y_min, _x_min, y_max, _x_max = region.bbox
        box_height = (y_max - y_min) * image_height / 1000
        if region.score >= min_score and box_height >= min_box_height:
            filtered.append(region)
    return filtered


def _result_payload(result: Any) -> dict[str, Any]:
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if payload is None and isinstance(result, dict):
        payload = result
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported PaddleOCR result type: {type(result)!r}")
    nested = payload.get("res")
    return nested if isinstance(nested, dict) else payload


def _parse_paddle_regions(result: Any, width: int, height: int) -> list[TextRegion]:
    payload = _result_payload(result)
    polygons = payload.get("dt_polys")
    if polygons is None:
        polygons = payload.get("polys")
    if polygons is None:
        polygons = []
    scores = payload.get("dt_scores")
    if scores is None:
        scores = payload.get("scores")
    if scores is None:
        scores = []

    parsed: list[TextRegion] = []
    for index, polygon in enumerate(polygons):
        points = [
            (float(point[0]), float(point[1]))
            for point in polygon
            if len(point) >= 2
        ]
        if not points:
            continue
        x_min = max(0, min(width, round(min(point[0] for point in points))))
        y_min = max(0, min(height, round(min(point[1] for point in points))))
        x_max = max(0, min(width, round(max(point[0] for point in points))))
        y_max = max(0, min(height, round(max(point[1] for point in points))))
        bbox = (
            round(y_min * 1000 / height),
            round(x_min * 1000 / width),
            round(y_max * 1000 / height),
            round(x_max * 1000 / width),
        )
        score = float(scores[index]) if index < len(scores) else 1.0
        parsed.append(TextRegion(bbox=bbox, score=score))
    return parsed


class PaddleTextDetector:
    """Lazy PP-OCR detector wrapper; the model is created only when screening runs."""

    def __init__(
        self,
        model_name: str = DEFAULT_SCREENING_MODEL,
        device: str = "cpu",
    ) -> None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        try:
            from paddleocr import TextDetection
        except ImportError as error:  # pragma: no cover - depends on runtime extras.
            raise RuntimeError(
                "PP-OCR screening requires paddleocr and paddlepaddle. "
                "Run `uv sync --project environments/gemini`."
            ) from error

        kwargs: dict[str, Any] = {"model_name": model_name, "device": device}
        if device.startswith("cpu"):
            # Paddle 3.3 + PP-OCRv5 currently fails with MKL-DNN enabled.
            kwargs["enable_mkldnn"] = False
        self.model = TextDetection(**kwargs)

    def detect(self, image: Image.Image) -> list[TextRegion]:
        import numpy as np

        results = self.model.predict(np.asarray(image), batch_size=1)
        regions: list[TextRegion] = []
        for result in results:
            regions.extend(_parse_paddle_regions(result, image.width, image.height))
        return regions


def screen_image(
    image: Image.Image,
    *,
    detector: TextDetector,
    max_side: int = DEFAULT_MAX_SIDE,
    min_score: float = DEFAULT_MIN_SCORE,
    min_box_height: int = DEFAULT_MIN_BOX_HEIGHT,
) -> ScreeningResult:
    screening_image = resize_for_screening(image, max_side=max_side)
    detected = detector.detect(screening_image)
    regions = filter_text_regions(
        detected,
        min_score=min_score,
        min_box_height=min_box_height,
        image_height=screening_image.height,
    )
    return ScreeningResult(
        has_text=bool(regions),
        regions=tuple(regions),
        image_width=screening_image.width,
        image_height=screening_image.height,
    )


def _signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def screening_config_signature(
    *,
    video_id: str,
    model_name: str,
    device: str,
    max_side: int,
    min_score: float,
    min_box_height: int,
) -> str:
    return _signature(
        {
            "version": PREPROCESS_VERSION,
            "video_id": video_id,
            "model_name": model_name,
            "device": device,
            "max_side": max_side,
            "min_score": min_score,
            "min_box_height": min_box_height,
        }
    )


def screening_signature(
    *,
    image_sha256: str,
    config_signature: str,
    result: ScreeningResult,
) -> str:
    return _signature(
        {
            "image_sha256": image_sha256,
            "config_signature": config_signature,
            "result": result.as_record(),
        }
    )


def _cached_record_is_valid(
    record: dict[str, Any],
    *,
    image_sha256: str,
    config_signature: str,
) -> bool:
    return (
        record.get("preprocess_version") == PREPROCESS_VERSION
        and record.get("image_sha256") == image_sha256
        and record.get("screening_config_signature") == config_signature
        and isinstance(record.get("screening_signature"), str)
    )


def screen_jobs(
    jobs: pd.DataFrame,
    *,
    video_id: str,
    output_dir: Path,
    model_name: str = DEFAULT_SCREENING_MODEL,
    device: str = "cpu",
    max_side: int = DEFAULT_MAX_SIDE,
    min_score: float = DEFAULT_MIN_SCORE,
    min_box_height: int = DEFAULT_MIN_BOX_HEIGHT,
    force: bool = False,
    detector: TextDetector | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Screen manifest rows and persist a reusable local OCR screening manifest."""
    required = {"video_id", "frame_id", "image_path", "image_sha256"}
    missing = sorted(required - set(jobs.columns))
    if missing:
        raise ValueError(f"Screening manifest is missing columns: {missing}")
    if jobs.empty:
        raise ValueError("Cannot screen an empty manifest")
    if set(jobs["video_id"].astype(str)) != {video_id}:
        raise ValueError(f"Screening rows do not all belong to {video_id}")

    config_signature = screening_config_signature(
        video_id=video_id,
        model_name=model_name,
        device=device,
        max_side=max_side,
        min_score=min_score,
        min_box_height=min_box_height,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "screening.parquet"
    report_path = output_dir / "report.json"

    cached: dict[int, dict[str, Any]] = {}
    if manifest_path.is_file() and not force:
        try:
            cached_frame = pd.read_parquet(manifest_path)
            for record in cached_frame.to_dict(orient="records"):
                cached[int(record["frame_id"])] = record
        except (OSError, ValueError, KeyError):
            cached = {}

    records: list[dict[str, Any]] = []
    reused = 0
    active_detector = detector
    sort_columns = [column for column in ("pts_time", "frame_id") if column in jobs]
    ordered_jobs = jobs.sort_values(sort_columns, na_position="last")
    for row in ordered_jobs.to_dict(orient="records"):
        frame_id = int(row["frame_id"])
        image_sha256 = str(row["image_sha256"])
        old = cached.get(frame_id)
        if not force and old is not None and _cached_record_is_valid(
            old, image_sha256=image_sha256, config_signature=config_signature
        ):
            records.append(old)
            reused += 1
            continue

        if active_detector is None:
            active_detector = PaddleTextDetector(model_name=model_name, device=device)
        image_path = Path(str(row["image_path"]))
        with Image.open(image_path) as source:
            result = screen_image(
                source,
                detector=active_detector,
                max_side=max_side,
                min_score=min_score,
                min_box_height=min_box_height,
            )
        record = {
            "video_id": video_id,
            "frame_id": frame_id,
            "image_path": str(image_path),
            "image_sha256": image_sha256,
            "screening_config_signature": config_signature,
            **result.as_record(),
        }
        record["screening_signature"] = screening_signature(
            image_sha256=image_sha256,
            config_signature=config_signature,
            result=result,
        )
        records.append(record)

    screening = pd.DataFrame(records).sort_values("frame_id").reset_index(drop=True)
    current_ids = set(screening["frame_id"].astype(int))
    preserved = [
        record
        for frame_id, record in cached.items()
        if frame_id not in current_ids
        and record.get("preprocess_version") == PREPROCESS_VERSION
    ]
    persisted = (
        pd.DataFrame([*preserved, *records])
        .sort_values("frame_id")
        .reset_index(drop=True)
    )
    atomic_write_parquet(persisted, manifest_path)
    report = {
        "pipeline_version": PREPROCESS_VERSION,
        "video_id": video_id,
        "model_name": model_name,
        "device": device,
        "max_side": max_side,
        "min_score": min_score,
        "min_box_height": min_box_height,
        "input_frames": len(screening),
        "frames_with_text": int(screening["screening_has_text"].sum()),
        "frames_without_text": int((~screening["screening_has_text"]).sum()),
        "reused_frames": reused,
        "cached_manifest_frames": len(persisted),
        "output_manifest": str(manifest_path),
    }
    atomic_write_json(report, report_path)
    enriched = jobs.merge(
        screening[
            [
                "frame_id",
                "screening_has_text",
                "screening_boxes_json",
                "screening_signature",
                "screening_config_signature",
                "screening_width",
                "screening_height",
            ]
        ],
        on="frame_id",
        how="left",
        validate="one_to_one",
    )
    if enriched["screening_has_text"].isna().any():
        raise ValueError("Screening manifest did not cover every input frame")
    return enriched, report
