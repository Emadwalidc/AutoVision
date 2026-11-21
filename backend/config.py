"""Configuration loader for AutoVision FastAPI backend."""

from __future__ import annotations

import json
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
    color_models_dir: Path | None = None
    gallery_zip_path: Path | None = None
    yolo_seg_weights: Path | None = None
    eu_extension_dir: Path | None = None
    model_logit_temperature: float = 2.15
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


def _read_model_version(models_dir: Path) -> str | None:
    lmap_path = models_dir / "label_maps.json"
    if not lmap_path.is_file():
        return None
    try:
        data = json.loads(lmap_path.read_text(encoding="utf-8"))
        val = str(data.get("model_version", "")).strip()
        return val or None
    except Exception:
        return None


def load_settings(project_root: Path | None = None) -> BackendSettings:
    """Loads settings with optional AUTOVISION_* environment variable overrides."""
    root = project_root or Path(__file__).resolve().parent.parent

    models_env = os.environ.get("AUTOVISION_MODELS_DIR", "").strip()
    if models_env:
        models_dir = Path(models_env)
    elif (root / "models" / "v2_phase1" / "autovision_multitask_best.pt").is_file():
        models_dir = root / "models" / "v2_phase1"
    else:
        models_dir = root / "models"

    manifest_env = os.environ.get("AUTOVISION_MANIFEST_PATH", "").strip()
    if manifest_env:
        default_manifest = Path(manifest_env)
    else:
        candidates = (
            models_dir / "autovision_v2_manifest.csv",
            root / "data_manifests" / "local_vmmrdb_multitask_manifest.csv",
            root / "data_manifests" / "local_vmmrdb_manifest.csv",
        )
        default_manifest = next((p for p in candidates if p.is_file()), candidates[1])

    color_env = os.environ.get("AUTOVISION_COLOR_MODELS_DIR", "").strip()
    if color_env:
        color_models_dir: Path | None = Path(color_env)
    elif (root / "models" / "v2_color" / "autovision_color_expert_v2.pt").is_file():
        color_models_dir = root / "models" / "v2_color"
    else:
        color_models_dir = None

    gallery_zip_env = os.environ.get("AUTOVISION_GALLERY_ZIP_PATH", "").strip()
    if gallery_zip_env:
        gallery_zip_path: Path | None = Path(gallery_zip_env)
    elif (models_dir / "gallery_v2_thumbs.zip").is_file():
        gallery_zip_path = models_dir / "gallery_v2_thumbs.zip"
    else:
        gallery_zip_path = None

    seg_env = os.environ.get("AUTOVISION_YOLO_SEG_WEIGHTS", "").strip()
    if seg_env:
        yolo_seg_weights: Path | None = Path(seg_env)
    elif (root / "yolov8n-seg.pt").is_file():
        yolo_seg_weights = root / "yolov8n-seg.pt"
    else:
        yolo_seg_weights = None

    eu_env = os.environ.get("AUTOVISION_EU_EXTENSION_DIR", "").strip()
    if eu_env:
        eu_extension_dir: Path | None = Path(eu_env)
    elif (root / "models" / "v2_eu_extension" / "eu_prototypes.npz").is_file():
        eu_extension_dir = root / "models" / "v2_eu_extension"
    else:
        eu_extension_dir = None

    cors_raw = os.environ.get("AUTOVISION_CORS_ORIGINS", "*").strip()
    cors_origins = tuple(o.strip() for o in cors_raw.split(",") if o.strip()) or ("*",)
    model_version = (
        os.environ.get("AUTOVISION_MODEL_VERSION", "").strip()
        or _read_model_version(models_dir)
        or "autovision-v1-vitb16-multitask"
    )

    return BackendSettings(
        project_root=root,
        models_dir=models_dir,
        manifest_path=default_manifest,
        data_dir=Path(os.environ.get("AUTOVISION_DATA_DIR", str(root.parent / "data"))),
        yolo_weights=Path(os.environ.get("AUTOVISION_YOLO_WEIGHTS", str(root / "yolov8n.pt"))),
        color_models_dir=color_models_dir,
        gallery_zip_path=gallery_zip_path,
        yolo_seg_weights=yolo_seg_weights,
        eu_extension_dir=eu_extension_dir,
        model_logit_temperature=float(os.environ.get("AUTOVISION_MODEL_TEMPERATURE", "2.15")),
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
        model_version=model_version,
        cors_origins=cors_origins,
    )
