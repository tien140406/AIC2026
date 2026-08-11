from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from .schema import COCO_CLASSES

DEFAULT_MODEL_ID = "co_dino_5scale_lsj_r50_3x_coco"
DEFAULT_CODE_REVISION = "26653521e47157fdd819e764b4bb949057ea9f3b"
DEFAULT_CONFIG_RELATIVE = Path(
    "projects/configs/co_dino/co_dino_5scale_lsj_r50_3x_coco.py"
)
DEFAULT_CHECKPOINT_NAME = f"{DEFAULT_MODEL_ID}.pth"
DEFAULT_CHECKPOINT_GDRIVE_ID = "1YqeQzEc0aS5x0s9lpUhnRU1k6c8v4_pz"


def flatten_codetr_result(
    result: Any,
    class_names: tuple[str, ...] = COCO_CLASSES,
) -> list[dict[str, Any]]:
    """Normalize MMDetection 2.x and 3.x result containers."""
    if hasattr(result, "pred_instances"):
        instances = result.pred_instances
        boxes = instances.bboxes.detach().cpu().numpy()
        scores = instances.scores.detach().cpu().numpy()
        labels = instances.labels.detach().cpu().numpy()
        return [
            {
                "class_id": int(label),
                "class_name": class_names[int(label)],
                "score": float(score),
                "bbox_xyxy": [float(value) for value in box[:4]],
            }
            for box, score, label in zip(boxes, scores, labels)
        ]

    if isinstance(result, tuple):
        result = result[0]
    if not isinstance(result, (list, tuple)):
        raise TypeError(f"Unsupported Co-DETR result type: {type(result)!r}")

    detections: list[dict[str, Any]] = []
    for class_id, class_boxes in enumerate(result):
        if class_id >= len(class_names):
            raise ValueError(f"Detection class ID {class_id} has no class name")
        boxes = np.asarray(class_boxes)
        if boxes.size == 0:
            continue
        if boxes.ndim != 2 or boxes.shape[1] < 5:
            raise ValueError(
                f"Expected Nx5 detection array for class {class_id}, got {boxes.shape}"
            )
        for row in boxes:
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "score": float(row[4]),
                    "bbox_xyxy": [float(value) for value in row[:4]],
                }
            )
    return detections


class CoDetrDetector:
    def __init__(
        self,
        *,
        codetr_root: Path,
        config_path: Path,
        checkpoint_path: Path,
        device: str = "cuda:0",
        fp16: bool = False,
    ) -> None:
        if not codetr_root.is_dir():
            raise FileNotFoundError(
                f"Co-DETR source is missing: {codetr_root}. Run scripts/setup_codetr.py."
            )
        if not config_path.is_file():
            raise FileNotFoundError(f"Co-DETR config is missing: {config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Co-DETR checkpoint is missing: {checkpoint_path}. "
                "Run scripts/setup_codetr.py."
            )

        root_text = str(codetr_root.resolve())
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        __import__("projects")
        from mmdet.apis import inference_detector, init_detector

        self._inference_detector = inference_detector
        self.model = init_detector(
            str(config_path.resolve()),
            str(checkpoint_path.resolve()),
            device=device,
        )
        if fp16:
            raise ValueError(
                "FP16 is disabled for the pinned MMCV runtime because "
                "MultiScaleDeformableAttention has no Half CUDA kernel"
            )
        classes = getattr(self.model, "CLASSES", None) or COCO_CLASSES
        self.class_names = tuple(str(name) for name in classes)

    def predict(self, image_path: Path) -> list[dict[str, Any]]:
        result = self._inference_detector(self.model, str(image_path))
        return flatten_codetr_result(result, self.class_names)

    def peak_vram_mb(self) -> float | None:
        try:
            import torch

            return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
        except (ImportError, RuntimeError):
            return None
