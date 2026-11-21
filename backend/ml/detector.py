"""Vehicle bounding-box detection abstraction and YOLOv8n implementation."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DetectedBox:
    """Detected or fallback vehicle crop coordinates within an image."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    area_ratio: float
    detected_by_yolo: bool
    mask: np.ndarray | None = field(default=None, compare=False, repr=False)


class VehicleDetector(Protocol):
    """Protocol for swappable vehicle localization backends."""

    def detect(self, image: Image.Image) -> list[DetectedBox]:
        ...


class YoloVehicleDetector:
    """YOLOv8n COCO vehicle detector/segmenter ([car, bus, truck]) with 85% center fallback."""

    def __init__(
        self,
        weights_path: Path,
        cache_dir: Path,
        conf: float = 0.25,
        min_area_ratio: float = 0.04,
        pad: float = 0.05,
        max_vehicles: int = 8,
        device: str = "cpu",
        seg_weights_path: Path | None = None,
    ) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(cache_dir))
        from ultralytics import YOLO

        if seg_weights_path is not None and seg_weights_path.is_file():
            self._model = YOLO(str(seg_weights_path))
            self._has_seg = True
        else:
            model_ref = str(weights_path) if weights_path.is_file() else "yolov8n.pt"
            self._model = YOLO(model_ref)
            self._has_seg = False
        self.conf = float(conf)
        self.min_area_ratio = float(min_area_ratio)
        self.pad = float(pad)
        self.max_vehicles = max(int(max_vehicles), 1)
        self.device = device

    def detect(self, image: Image.Image) -> list[DetectedBox]:
        img_w, img_h = image.size
        img_area = float(max(img_w * img_h, 1))
        predict_kwargs: dict[str, object] = {
            "classes": [2, 5, 7],
            "conf": self.conf,
            "device": self.device,
            "verbose": False,
        }
        if self._has_seg:
            predict_kwargs["retina_masks"] = True
        results = self._model.predict(image, **predict_kwargs)
        res = results[0]
        detected: list[DetectedBox] = []

        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            masks_data = res.masks.data if (self._has_seg and res.masks is not None) else None
            for idx, (box, conf_val) in enumerate(zip(xyxy, confs)):
                bx1, by1, bx2, by2 = [float(v) for v in box]
                bw, bh = max(bx2 - bx1, 0.0), max(by2 - by1, 0.0)
                ratio = min(max((bw * bh) / img_area, 0.0), 1.0)
                if ratio < self.min_area_ratio:
                    continue
                x1 = max(0, int(bx1 - self.pad * bw))
                y1 = max(0, int(by1 - self.pad * bh))
                x2 = min(img_w, max(x1 + 1, int(bx2 + self.pad * bw)))
                y2 = min(img_h, max(y1 + 1, int(by2 + self.pad * bh)))

                crop_mask: np.ndarray | None = None
                if masks_data is not None and idx < len(masks_data):
                    full_mask = masks_data[idx].cpu().numpy() >= 0.5
                    if full_mask.shape != (img_h, img_w):
                        full_mask = (
                            np.asarray(
                                Image.fromarray((full_mask.astype(np.uint8) * 255), mode="L").resize(
                                    (img_w, img_h), Image.Resampling.NEAREST
                                )
                            )
                            >= 128
                        )
                    sliced = full_mask[y1:y2, x1:x2]
                    if int(sliced.sum()) >= 64:
                        crop_mask = sliced

                detected.append(
                    DetectedBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=min(max(float(conf_val), 0.0), 1.0),
                        area_ratio=round(ratio, 4),
                        detected_by_yolo=True,
                        mask=crop_mask,
                    )
                )

        if detected:
            def _priority_key(b: DetectedBox) -> tuple[float, float]:
                cx = 0.5 * (b.x1 + b.x2) / max(img_w, 1)
                cy = 0.5 * (b.y1 + b.y2) / max(img_h, 1)
                dist = math.hypot(cx - 0.5, cy - 0.5) / 0.70710678
                center_prox = max(0.0, 1.0 - dist)
                return (-(b.area_ratio * (0.70 + 0.30 * center_prox)), -b.confidence)

            detected.sort(key=_priority_key)
            return detected[: self.max_vehicles]

        mw, mh = int(img_w * 0.075), int(img_h * 0.075)
        x1 = min(mw, max(img_w - 1, 0))
        y1 = min(mh, max(img_h - 1, 0))
        x2 = max(x1 + 1, img_w - mw)
        y2 = max(y1 + 1, img_h - mh)
        fb_ratio = min(max(((x2 - x1) * (y2 - y1)) / img_area, 0.0), 1.0)
        return [
            DetectedBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=0.0,
                area_ratio=round(fb_ratio, 4),
                detected_by_yolo=False,
            )
        ]
