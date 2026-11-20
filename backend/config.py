"""Configuration loader for AutoVision FastAPI backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BackendSettings:
    """Immutable runtime settings for AutoVision inference and API limits."""

    project_root: Path
    models_dir: Path
    manifest_path: Path
    data_dir: Path
    yolo_weights: Path
    yolo_conf: float = 0.25
    yolo_min_area_ratio: float = 0.04
    yolo_pad: float = 0.05
    max_vehicles_per_image: int = 8
    max_upload_bytes: int = 15 * 1024 * 1024
    max_image_pixels: int = 25_000_000
    max_concurrent_inferences: int = 2
    default_top_k_similar: int = 5
    max_top_k_similar: int = 12
    thumbnail_max_width: int = 360
    thumbnail_jpeg_quality: int = 82
    device: str = "auto"
    model_version: str = "autovision-v1-vitb16-multitask"
    cors_origins: tuple[str, ...] = ("*",)


def load_settings(project_root: Path | None = None) -> BackendSettings:
    """Loads settings with optional AUTOVISION_* environment variable overrides."""
    root = project_root or Path(__file__).resolve().parent.parent

    default_manifest = root / "data_manifests" / "local_vmmrdb_multitask_manifest.csv"
    if not default_manifest.is_file():
        fallback_manifest = root / "data_manifests" / "local_vmmrdb_manifest.csv"
        if fallback_manifest.is_file():
            default_manifest = fallback_manifest

    cors_raw = os.environ.get("AUTOVISION_CORS_ORIGINS", "*").strip()
    cors_origins = tuple(o.strip() for o in cors_raw.split(",") if o.strip()) or ("*",)

    return BackendSettings(
        project_root=root,
        models_dir=Path(os.environ.get("AUTOVISION_MODELS_DIR", str(root / "models"))),
        manifest_path=Path(os.environ.get("AUTOVISION_MANIFEST_PATH", str(default_manifest))),
        data_dir=Path(os.environ.get("AUTOVISION_DATA_DIR", str(root.parent / "data"))),
        yolo_weights=Path(os.environ.get("AUTOVISION_YOLO_WEIGHTS", str(root / "yolov8n.pt"))),
        yolo_conf=float(os.environ.get("AUTOVISION_YOLO_CONF", "0.25")),
        yolo_min_area_ratio=float(os.environ.get("AUTOVISION_YOLO_MIN_AREA_RATIO", "0.04")),
        yolo_pad=float(os.environ.get("AUTOVISION_YOLO_PAD", "0.05")),
        max_vehicles_per_image=int(os.environ.get("AUTOVISION_MAX_VEHICLES", "8")),
        max_upload_bytes=int(os.environ.get("AUTOVISION_MAX_UPLOAD_BYTES", str(15 * 1024 * 1024))),
        max_image_pixels=int(os.environ.get("AUTOVISION_MAX_IMAGE_PIXELS", "25000000")),
        max_concurrent_inferences=int(os.environ.get("AUTOVISION_MAX_CONCURRENT_INFERENCES", "2")),
        default_top_k_similar=int(os.environ.get("AUTOVISION_DEFAULT_TOP_K_SIMILAR", "5")),
        max_top_k_similar=int(os.environ.get("AUTOVISION_MAX_TOP_K_SIMILAR", "12")),
        thumbnail_max_width=int(os.environ.get("AUTOVISION_THUMBNAIL_MAX_WIDTH", "360")),
        thumbnail_jpeg_quality=int(os.environ.get("AUTOVISION_THUMBNAIL_JPEG_QUALITY", "82")),
        device=os.environ.get("AUTOVISION_DEVICE", "auto").strip() or "auto",
        model_version=os.environ.get("AUTOVISION_MODEL_VERSION", "autovision-v1-vitb16-multitask"),
        cors_origins=cors_origins,
    )
