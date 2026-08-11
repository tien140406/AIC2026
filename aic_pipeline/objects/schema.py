from __future__ import annotations

from collections.abc import Sequence
from typing import Any

SCHEMA_VERSION = "codetr_frame_objects_v1"

COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def normalize_detection(
    *,
    class_id: int,
    class_name: str,
    score: float,
    bbox_xyxy: Sequence[float],
    image_width: int,
    image_height: int,
) -> dict[str, Any] | None:
    """Clip a detector box and add normalized coordinates used by the pipeline."""
    if image_width <= 0 or image_height <= 0 or len(bbox_xyxy) < 4:
        return None
    x_min, y_min, x_max, y_max = (float(value) for value in bbox_xyxy[:4])
    x_min = min(max(x_min, 0.0), float(image_width))
    x_max = min(max(x_max, 0.0), float(image_width))
    y_min = min(max(y_min, 0.0), float(image_height))
    y_max = min(max(y_max, 0.0), float(image_height))
    if x_max <= x_min or y_max <= y_min:
        return None

    area_ratio = ((x_max - x_min) * (y_max - y_min)) / (image_width * image_height)
    bbox_yxyx_1000 = [
        round(1000 * y_min / image_height),
        round(1000 * x_min / image_width),
        round(1000 * y_max / image_height),
        round(1000 * x_max / image_width),
    ]
    return {
        "class_id": int(class_id),
        "class_name": str(class_name),
        "score": float(score),
        "bbox_xyxy": [x_min, y_min, x_max, y_max],
        "bbox_yxyx_1000": bbox_yxyx_1000,
        "area_ratio": float(area_ratio),
    }


def valid_raw_payload(payload: dict[str, Any]) -> bool:
    if payload.get("schema_version") != SCHEMA_VERSION:
        return False
    required_payload = {
        "video_id",
        "frame_id",
        "image_path",
        "image_sha256",
        "image_width",
        "image_height",
        "model_id",
        "run_signature",
        "processed_at",
        "detections",
    }
    if not required_payload <= set(payload):
        return False
    if not isinstance(payload["video_id"], str) or not payload["video_id"]:
        return False
    if not isinstance(payload["frame_id"], int) or isinstance(
        payload["frame_id"], bool
    ):
        return False
    if not isinstance(payload["image_width"], int) or payload["image_width"] <= 0:
        return False
    if not isinstance(payload["image_height"], int) or payload["image_height"] <= 0:
        return False
    if not all(
        isinstance(payload[key], str) and payload[key]
        for key in (
            "image_path",
            "image_sha256",
            "model_id",
            "run_signature",
            "processed_at",
        )
    ):
        return False

    detections = payload["detections"]
    if not isinstance(detections, list):
        return False
    required_detection = {
        "class_id",
        "class_name",
        "score",
        "bbox_xyxy",
        "bbox_yxyx_1000",
        "area_ratio",
    }
    for item in detections:
        if not isinstance(item, dict) or not required_detection <= set(item):
            return False
        if not isinstance(item["class_id"], int) or isinstance(item["class_id"], bool):
            return False
        if not isinstance(item["class_name"], str) or not item["class_name"]:
            return False
        if not isinstance(item["score"], (int, float)) or isinstance(
            item["score"], bool
        ):
            return False
        if not isinstance(item["area_ratio"], (int, float)) or isinstance(
            item["area_ratio"], bool
        ):
            return False
        if not isinstance(item["bbox_xyxy"], list) or len(item["bbox_xyxy"]) != 4:
            return False
        if (
            not isinstance(item["bbox_yxyx_1000"], list)
            or len(item["bbox_yxyx_1000"]) != 4
        ):
            return False
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in item["bbox_xyxy"]
        ):
            return False
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in item["bbox_yxyx_1000"]
        ):
            return False
    return True
