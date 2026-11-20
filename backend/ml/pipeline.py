"""Orchestration pipeline combining decoding, detection, classification, and retrieval."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from backend.config import BackendSettings
from backend.ml.classifier import AttributeClassifier, PyTorchAutoVisionClassifier
from backend.ml.detector import DetectedBox, VehicleDetector, YoloVehicleDetector
from backend.ml.retriever import NumpyCosineRetriever, VehicleRetriever
from backend.schemas import (
    AnalyzeResponse,
    BBoxNorm,
    BBoxXYXY,
    BodyTypeAttribute,
    CatalogColorItem,
    CatalogResponse,
    ColorAttribute,
    HealthResponse,
    ImageSize,
    InferenceTimingsMs,
    MakeAttribute,
    ModelAttribute,
    TopKPrediction,
    VehicleAttributes,
    VehicleDetectionResult,
    YearAttribute,
    format_display_name,
)
from backend.security import decode_uploaded_image


def _top_k_predictions(
    probs: np.ndarray,
    labels: list[str],
    k: int = 3,
    use_display: bool = True,
) -> list[TopKPrediction]:
    k_eff = min(k, len(labels), len(probs))
    top_idx = np.argsort(-probs)[:k_eff]
    return [
        TopKPrediction(
            label=labels[int(idx)],
            display_name=format_display_name(labels[int(idx)]) if use_display else labels[int(idx)],
            confidence=round(min(max(float(probs[int(idx)]), 0.0), 1.0), 4),
        )
        for idx in top_idx
    ]


class AutoVisionPipeline:
    """Stateless-per-request ML orchestration pipeline with swappable components."""

    def __init__(
        self,
        settings: BackendSettings,
        detector: VehicleDetector,
        classifier: AttributeClassifier,
        retriever: VehicleRetriever,
    ) -> None:
        self.settings = settings
        self.detector = detector
        self.classifier = classifier
        self.retriever = retriever

    @classmethod
    def from_settings(cls, settings: BackendSettings) -> "AutoVisionPipeline":
        classifier = PyTorchAutoVisionClassifier(
            models_dir=settings.models_dir,
            device=settings.device,
        )
        detector = YoloVehicleDetector(
            weights_path=settings.yolo_weights,
            cache_dir=settings.project_root / ".cache" / "ultralytics",
            conf=settings.yolo_conf,
            min_area_ratio=settings.yolo_min_area_ratio,
            pad=settings.yolo_pad,
            max_vehicles=settings.max_vehicles_per_image,
            device=classifier.device,
        )
        retriever = NumpyCosineRetriever(
            models_dir=settings.models_dir,
            manifest_path=settings.manifest_path,
            data_dir=settings.data_dir,
        )
        return cls(settings=settings, detector=detector, classifier=classifier, retriever=retriever)

    def _build_vehicle_result(
        self,
        vehicle_idx: int,
        box: DetectedBox,
        img_w: int,
        img_h: int,
        clf_out: Any,
        lmaps: dict[str, Any],
        top_k_similar: int,
    ) -> VehicleDetectionResult:
        make_classes: list[str] = lmaps["make_classes"]
        model_classes: list[str] = lmaps["model_classes"]
        body_classes: list[str] = lmaps["body_classes"]
        color_classes: list[str] = lmaps["color_classes"]
        color_groups: dict[str, str] = lmaps["color_groups_11"]
        color_hex: dict[str, str] = lmaps["color_hex"]
        year_min = int(lmaps["year_min"])
        year_max = int(lmaps["year_max"])

        mk_top3 = _top_k_predictions(clf_out.make_probs[vehicle_idx], make_classes, k=3, use_display=True)
        md_top3 = _top_k_predictions(clf_out.model_probs[vehicle_idx], model_classes, k=3, use_display=True)
        bd_top3 = _top_k_predictions(clf_out.body_probs[vehicle_idx], body_classes, k=3, use_display=False)
        cl_top3 = _top_k_predictions(clf_out.color_probs[vehicle_idx], color_classes, k=3, use_display=False)

        emb = clf_out.embeddings[vehicle_idx]
        few_shot = self.retriever.match_prototype(emb)
        similars = self.retriever.search_similar(emb, top_k=top_k_similar) if top_k_similar > 0 else []

        yr_val = float(clf_out.year_expected[vehicle_idx])
        yr_std = max(float(clf_out.year_std[vehicle_idx]), 0.0)
        yr_round = min(max(int(round(yr_val)), year_min), year_max)
        yr_low = max(year_min, min(yr_round, int(math.floor(yr_val - yr_std))))
        yr_high = min(year_max, max(yr_round, int(math.ceil(yr_val + yr_std))))

        top_color = cl_top3[0].label
        bbox_norm = BBoxNorm(
            x1=round(min(max(box.x1 / max(img_w, 1), 0.0), 1.0), 4),
            y1=round(min(max(box.y1 / max(img_h, 1), 0.0), 1.0), 4),
            x2=round(min(max(box.x2 / max(img_w, 1), 0.0), 1.0), 4),
            y2=round(min(max(box.y2 / max(img_h, 1), 0.0), 1.0), 4),
        )

        attrs = VehicleAttributes(
            make=MakeAttribute(
                label=mk_top3[0].label,
                display_name=mk_top3[0].display_name,
                confidence=mk_top3[0].confidence,
                top3=mk_top3,
            ),
            model=ModelAttribute(
                canonical_class=md_top3[0].label,
                display_name=md_top3[0].display_name,
                confidence=md_top3[0].confidence,
                top3=md_top3,
                few_shot_match=few_shot,
            ),
            year=YearAttribute(
                predicted=round(yr_val, 1),
                predicted_rounded=yr_round,
                range_low=yr_low,
                range_high=yr_high,
                std=round(yr_std, 2),
            ),
            body_type=BodyTypeAttribute(
                label=bd_top3[0].label,
                display_name=bd_top3[0].display_name,
                confidence=bd_top3[0].confidence,
                top3=bd_top3,
            ),
            color=ColorAttribute(
                label=top_color,
                display_name=top_color,
                group=color_groups.get(top_color, top_color),
                hex=color_hex.get(top_color, "#6B7280"),
                confidence=cl_top3[0].confidence,
                top3=cl_top3,
            ),
        )

        return VehicleDetectionResult(
            vehicle_index=vehicle_idx,
            detected_by_yolo=box.detected_by_yolo,
            detection_confidence=round(box.confidence, 4),
            area_ratio=round(box.area_ratio, 4),
            bbox_xyxy=BBoxXYXY(x1=float(box.x1), y1=float(box.y1), x2=float(box.x2), y2=float(box.y2)),
            bbox_norm=bbox_norm,
            attributes=attrs,
            similar_vehicles=similars,
        )

    def analyze_bytes(
        self,
        raw_bytes: bytes,
        top_k_similar: int | None = None,
        use_tta: bool = False,
    ) -> AnalyzeResponse:
        t_start = time.perf_counter()
        k_sim = min(
            max(0, top_k_similar if top_k_similar is not None else self.settings.default_top_k_similar),
            self.settings.max_top_k_similar,
        )

        rgb = decode_uploaded_image(raw_bytes, self.settings.max_upload_bytes, self.settings.max_image_pixels)
        crops = []
        try:
            img_w, img_h = rgb.size
            t_dec = time.perf_counter()

            boxes = self.detector.detect(rgb)
            t_det = time.perf_counter()

            crops = [rgb.crop((b.x1, b.y1, b.x2, b.y2)) for b in boxes]
            clf_out = self.classifier.classify_crops(crops, use_tta=use_tta)
            t_clf = time.perf_counter()

            lmaps = self.classifier.label_maps
            vehicles = [
                self._build_vehicle_result(idx, box, img_w, img_h, clf_out, lmaps, k_sim)
                for idx, box in enumerate(boxes)
            ]
            t_ret = time.perf_counter()
        finally:
            for c in crops:
                c.close()
            rgb.close()

        return AnalyzeResponse(
            model_version=self.settings.model_version,
            device=self.classifier.device,
            image_size=ImageSize(width=img_w, height=img_h),
            num_vehicles=len(vehicles),
            primary_vehicle_index=0,
            vehicles=vehicles,
            timings_ms=InferenceTimingsMs(
                decode_ms=round((t_dec - t_start) * 1000.0, 2),
                detect_ms=round((t_det - t_dec) * 1000.0, 2),
                classify_ms=round((t_clf - t_det) * 1000.0, 2),
                retrieve_ms=round((t_ret - t_clf) * 1000.0, 2),
                total_ms=round((t_ret - t_start) * 1000.0, 2),
            ),
        )

    def get_health(self) -> HealthResponse:
        lm = self.classifier.label_maps
        return HealthResponse(
            status="ok",
            model_version=self.settings.model_version,
            device=self.classifier.device,
            ready=True,
            num_makes=len(lm["make_classes"]),
            num_models=len(lm["model_classes"]),
            num_prototype_classes=len(self.retriever.prototype_classes),
            num_bodies=len(lm["body_classes"]),
            num_colors=len(lm["color_classes"]),
            gallery_size=self.retriever.gallery_size,
            yolo_available=True,
        )

    def get_catalog(self) -> CatalogResponse:
        lm = self.classifier.label_maps
        color_groups = lm["color_groups_11"]
        color_hex = lm["color_hex"]
        return CatalogResponse(
            model_version=self.settings.model_version,
            makes=list(lm["make_classes"]),
            models=list(lm["model_classes"]),
            prototype_classes=self.retriever.prototype_classes,
            body_types=list(lm["body_classes"]),
            colors=[
                CatalogColorItem(
                    label=c,
                    group=color_groups.get(c, c),
                    hex=color_hex.get(c, "#6B7280"),
                )
                for c in lm["color_classes"]
            ],
            year_min=int(lm["year_min"]),
            year_max=int(lm["year_max"]),
        )
