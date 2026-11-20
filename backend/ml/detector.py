"""Vehicle bounding-box detection abstraction and YOLOv8n implementation."""

from __future__ import annotations

import os
from dataclasses import dataclass
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


class VehicleDetector(Protocol):
    """Protocol for swappable vehicle localization backends."""

    def detect(self, image: Image.Image) -> list[DetectedBox]:
        ...


class YoloVehicleDetector:
    """YOLOv8n COCO vehicle detector ([car, bus, truck]) with 85% center fallback."""

    def __init__(
        self,
        weights_path: Path,
        cache_dir: Path,
        conf: float = 0.25,
        min_area_ratio: float = 0.04,
        pad: float = 0.05,
        max_vehicles: int = 8,
        device: str = "cpu",
    ) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(cache_dir))
        from ultralytics import YOLO

        model_ref = str(weights_path) if weights_path.is_file() else "yolov8n.pt"
        self._model = YOLO(model_ref)
        self.conf = float(conf)
        self.min_area_ratio = float(min_area_ratio)
        self.pad = float(pad)
        self.max_vehicles = max(int(max_vehicles), 1)
        self.device = device

    def detect(self, image: Image.Image) -> list[DetectedBox]:
        img_w, img_h = image.size
        img_area = float(max(img_w * img_h, 1))
        results = self._model.predict(
            image,
            classes=[2, 5, 7],
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        res = results[0]
        detected: list[DetectedBox] = []

        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for box, conf_val in zip(xyxy, confs):
                bx1, by1, bx2, by2 = [float(v) for v in box]
                bw, bh = max(bx2 - bx1, 0.0), max(by2 - by1, 0.0)
                ratio = min(max((bw * bh) / img_area, 0.0), 1.0)
                if ratio < self.min_area_ratio:
                    continue
                x1 = max(0, int(bx1 - self.pad * bw))
                y1 = max(0, int(by1 - self.pad * bh))
                x2 = min(img_w, max(x1 + 1, int(bx2 + self.pad * bw)))
                y2 = min(img_h, max(y1 + 1, int(by2 + self.pad * bh)))
                detected.append(
                    DetectedBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=min(max(float(conf_val), 0.0), 1.0),
                        area_ratio=round(ratio, 4),
                        detected_by_yolo=True,
                    )
                )

        if detected:
            detected.sort(key=lambda b: (-b.area_ratio, -b.confidence))
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
