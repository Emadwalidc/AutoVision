"""AutoVision v2 (Phase 2) - Standalone Color Expert Training on Kaggle GPU (VCoR + DVM-CAR + YOLOv8n-seg).

Fixes dark/neutral reflection ambiguities (Black vs Brown, Silver vs Gray, White vs Silver) by:
  1. Extracting car-only pixel masks via YOLOv8n-seg (VCoR) and border flood-fill (DVM-CAR) -> neutral gray bg
  2. Combining ConvNeXt-Tiny visual features with a 16-D Masked CIE-Lab + Chroma paint physics branch
  3. Applying warm brick / cool sky illuminant reflection augmentation on achromatic classes (no hue shift!)
  4. Training jointly on real street VCoR + DVM-CAR Confirmed_fronts across 12 clean automotive colors

Kaggle Notebook settings:
  - Accelerator: GPU T4 x2 (recommended) or GPU P100
  - Internet: ON (auto-downloads DVM Confirmed_fronts.zip if not attached, plus yolov8n-seg & ConvNeXt weights)
  - Install if missing: `!pip install -q ultralytics`
  - Inputs: attach `landrykezebou/vcor-vehicle-color-recognition-dataset` (and optionally `Confirmed_fronts`)

Outputs (`/kaggle/working/autovision_v2_color/`):
  - autovision_color_expert_v2.pt
  - color_label_maps_v2.json
  - color_training_history.csv
  - color_training_summary.json
  - color_confusion_matrix_v2.png
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import random
import re
import shutil
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict, deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics_cfg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from tqdm.auto import tqdm
from ultralytics import YOLO

try:
    from scipy.ndimage import label as _scipy_cc_label
except Exception:
    _scipy_cc_label = None

# --- Hyperparameters & Paths ---
DVM_FIGSHARE_URL = "https://ndownloader.figshare.com/files/34792480"
DVM_DROPBOX_URL = "https://www.dropbox.com/s/xmyz2na20f8lth8/Confirmed_fronts.zip?dl=1"
CACHE_DIR = Path("/tmp/autovision_color_v2")
OUT_DIR = Path("/kaggle/working/autovision_v2_color")

SEED = 42
MASK_IMG_SIZE = 256
MODEL_IMG_SIZE = 224
MAX_DVM_PER_CLASS = 3500
VCOR_TRAIN_REPEAT = 3
WARMUP_EPOCHS = 2
FINETUNE_EPOCHS = 12
WARMUP_LR = 2e-3
BACKBONE_LR = 1e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.02
LABEL_SMOOTHING = 0.04
NUM_WORKERS = 2

# --- Embedded Constants (Parity with src/color_expert.py) ---
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
NEUTRAL_BG_RGB: tuple[int, int, int] = (128, 128, 128)

COLOR_CLASSES_V2: tuple[str, ...] = (
    "Black", "White", "Silver", "Gray", "Blue", "Red",
    "Brown", "Green", "Yellow", "Orange", "Beige", "Purple",
)

COLOR_HEX_V2: dict[str, str] = {
    "Black": "#1C1C1E", "White": "#F8FAFC", "Silver": "#C0C7CE", "Gray": "#6B7280",
    "Blue": "#2563EB", "Red": "#DC2626", "Brown": "#7C4A2D", "Green": "#16A34A",
    "Yellow": "#EAB308", "Orange": "#EA580C", "Beige": "#D8C39E", "Purple": "#7E22CE",
}

COLOR_V2_ALIASES: dict[str, str | None] = {
    **{c.lower(): c for c in COLOR_CLASSES_V2},
    "grey": "Gray", "tan": "Beige", "gold": "Beige", "pink": "Purple",
    "violet": "Purple", "cyan": "Blue", "bronze": "Brown", "maroon": "Red",
    "burgundy": "Red", "cream": "Beige", "champagne": "Beige", "navy": "Blue",
    "turquoise": "Blue", "magenta": "Purple", "indigo": "Purple",
    "unlisted": None, "multicolour": None, "multicolor": None, "unknown": None, "na": None, "": None,
}

ACHROMATIC_CLASSES: frozenset[str] = frozenset({"Black", "White", "Silver", "Gray"})


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_color_v2(raw_label: str) -> str | None:
    key = re.sub(r"[\s_-]+", "", (raw_label or "").strip().lower())
    if key in COLOR_V2_ALIASES:
        return COLOR_V2_ALIASES[key]
    return COLOR_V2_ALIASES.get((raw_label or "").strip().lower())


def parse_dvm_color_from_path(rel_path: str) -> str | None:
    norm_path = rel_path.replace("\\", "/").strip("/")
    parts = norm_path.split("/")
    filename = parts[-1]
    if "$$" in filename:
        tokens = filename.split("$$")
        if len(tokens) >= 4:
            mapped = normalize_color_v2(tokens[3])
            if mapped is not None:
                return mapped
    for folder in reversed(parts[:-1]):
        mapped = normalize_color_v2(folder)
        if mapped is not None:
            return mapped
    return None


def _erode_mask_numpy(mask_bool: np.ndarray, erode_px: int) -> np.ndarray:
    if erode_px <= 0:
        return mask_bool
    out = mask_bool.copy()
    for _ in range(int(erode_px)):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1]
            & padded[1:-1, :-2] & padded[1:-1, 2:]
        )
    return out


def apply_neutral_gray_mask(
    im: Image.Image,
    mask_bool: np.ndarray,
    erode_px: int = 2,
) -> tuple[Image.Image, np.ndarray]:
    rgb = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    h, w = rgb.shape[:2]
    m = np.asarray(mask_bool, dtype=bool)
    if m.shape != (h, w):
        m = np.asarray(
            Image.fromarray((m.astype(np.uint8) * 255), mode="L").resize((w, h), Image.Resampling.NEAREST)
        ) >= 128

    if erode_px > 0 and int(m.sum()) >= 64:
        eroded = _erode_mask_numpy(m, erode_px)
        if int(eroded.sum()) >= 32:
            m = eroded

    if int(m.sum()) == 0:
        m = np.zeros((h, w), dtype=bool)
        m[int(0.12 * h) : int(0.88 * h), int(0.12 * w) : int(0.88 * w)] = True

    rgb[~m] = np.asarray(NEUTRAL_BG_RGB, dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB"), m


def _border_flood_bfs(white_mask: np.ndarray) -> np.ndarray:
    h, w = white_mask.shape
    bg = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for x in range(w):
        if white_mask[0, x] and not bg[0, x]:
            bg[0, x] = True
            q.append((0, x))
        if white_mask[h - 1, x] and not bg[h - 1, x]:
            bg[h - 1, x] = True
            q.append((h - 1, x))
    for y in range(1, h - 1):
        if white_mask[y, 0] and not bg[y, 0]:
            bg[y, 0] = True
            q.append((y, 0))
        if white_mask[y, w - 1] and not bg[y, w - 1]:
            bg[y, w - 1] = True
            q.append((y, w - 1))
    while q:
        cy, cx = q.popleft()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and white_mask[ny, nx] and not bg[ny, nx]:
                bg[ny, nx] = True
                q.append((ny, nx))
    return bg


def remove_dvm_white_border_bg(im: Image.Image, white_thresh: int = 242) -> tuple[Image.Image, np.ndarray]:
    rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    is_white = np.all(rgb >= int(white_thresh), axis=-1)
    if _scipy_cc_label is not None:
        labeled, _ = _scipy_cc_label(is_white)
        border_ids = set(np.unique(np.concatenate([labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]])))
        border_ids.discard(0)
        bg_mask = np.isin(labeled, list(border_ids)) if border_ids else np.zeros((h, w), dtype=bool)
    else:
        sw, sh = min(w, 64), min(h, 64)
        small = np.asarray(
            Image.fromarray((is_white.astype(np.uint8) * 255), mode="L").resize((sw, sh), Image.Resampling.NEAREST)
        ) >= 128
        small_bg = _border_flood_bfs(small)
        bg_mask = np.asarray(
            Image.fromarray((small_bg.astype(np.uint8) * 255), mode="L").resize((w, h), Image.Resampling.NEAREST)
        ) >= 128
        bg_mask &= is_white
    return apply_neutral_gray_mask(Image.fromarray(rgb, mode="RGB"), ~bg_mask, erode_px=1)


def rgb_to_lab_numpy(rgb01: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(rgb01, dtype=np.float32), 0.0, 1.0)
    linear = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883
    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0
    def _f(t: np.ndarray) -> np.ndarray:
        return np.where(t > eps, np.cbrt(np.maximum(t, 0.0)), (kappa * t + 16.0) / 116.0)
    fx, fy, fz = _f(x), _f(y), _f(z)
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1).astype(np.float32)


def extract_masked_lab_features(rgb_uint8: np.ndarray, mask_bool: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(rgb_uint8, dtype=np.uint8)
    h, w = arr.shape[:2]
    if mask_bool is not None:
        m = np.asarray(mask_bool, dtype=bool)
        if m.shape != (h, w):
            m = np.asarray(
                Image.fromarray((m.astype(np.uint8) * 255), mode="L").resize((w, h), Image.Resampling.NEAREST)
            ) >= 128
        pixels = arr[m]
    else:
        interior = np.zeros((h, w), dtype=bool)
        interior[int(0.15 * h) : int(0.85 * h), int(0.15 * w) : int(0.85 * w)] = True
        pixels = arr[interior]
    if len(pixels) == 0:
        pixels = arr.reshape(-1, 3)
    if len(pixels) > 4096:
        pixels = pixels[:: max(1, len(pixels) // 4096)]

    lab = rgb_to_lab_numpy(pixels.astype(np.float32) / 255.0)
    l_vals, a_vals, b_vals = lab[:, 0], lab[:, 1], lab[:, 2]
    chroma = np.sqrt(a_vals * a_vals + b_vals * b_vals)
    lq15, lq50, lq85 = np.percentile(l_vals, [15, 50, 85])
    l_std = float(np.std(l_vals))
    aq25, aq50, aq75 = np.percentile(a_vals, [25, 50, 75])
    bq25, bq50, bq75 = np.percentile(b_vals, [25, 50, 75])
    cq25, cq50, cq75 = np.percentile(chroma, [25, 50, 75])
    return np.array([
        lq15 / 100.0, lq50 / 100.0, lq85 / 100.0, l_std / 50.0,
        aq25 / 60.0, aq50 / 60.0, aq75 / 60.0,
        bq25 / 60.0, bq50 / 60.0, bq75 / 60.0,
        cq25 / 60.0, cq50 / 60.0, cq75 / 60.0,
        float(np.mean((l_vals < 28.0) & (chroma < 10.0))),
        float(np.mean((l_vals > 76.0) & (chroma < 9.0))),
        float(np.mean((l_vals >= 18.0) & (l_vals <= 62.0) & (a_vals > 4.0) & (b_vals > 6.0) & (chroma > 12.0))),
    ], dtype=np.float32)


def apply_illuminant_reflection_cast(
    im: Image.Image,
    label: str,
    rng: random.Random,
    mask_bool: np.ndarray | None = None,
    prob: float = 0.45,
) -> Image.Image:
    if label not in ACHROMATIC_CLASSES or rng.random() > float(prob):
        return im
    arr = np.asarray(im.convert("RGB"), dtype=np.float32)
    lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]) / 255.0
    weight = 0.25 + 0.75 * np.clip(lum, 0.0, 1.0)
    strength = rng.uniform(10.0, 26.0)
    if rng.random() < 0.6:
        delta = np.stack([strength * weight, 0.42 * strength * weight, -0.30 * strength * weight], axis=-1)
    else:
        delta = np.stack([-0.30 * strength * weight, 0.15 * strength * weight, 0.90 * strength * weight], axis=-1)
    shifted = np.clip(arr + delta, 0.0, 255.0).astype(np.uint8)
    if mask_bool is not None and mask_bool.shape == shifted.shape[:2]:
        out = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
        out[mask_bool] = shifted[mask_bool]
        return Image.fromarray(out, mode="RGB")
    return Image.fromarray(shifted, mode="RGB")


def compute_soft_class_weights(
    counts: Sequence[int] | np.ndarray | torch.Tensor,
    power: float = 0.30,
    min_weight: float = 0.5,
    max_weight: float = 3.5,
) -> torch.Tensor:
    c = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    weights = (float(c.sum()) / c) ** float(power)
    weights = weights / max(float(weights.mean()), 1e-12)
    return torch.tensor(np.clip(weights, min_weight, max_weight), dtype=torch.float32)


class AutoVisionColorExpert(nn.Module):
    def __init__(self, num_colors: int = len(COLOR_CLASSES_V2), lab_dim: int = 16, pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        base = convnext_tiny(weights=weights)
        self.backbone_features = base.features
        self.backbone_avgpool = base.avgpool
        self.backbone_norm = nn.LayerNorm(768)
        self.lab_mlp = nn.Sequential(
            nn.LayerNorm(lab_dim), nn.Linear(lab_dim, 64), nn.GELU(), nn.Dropout(0.15), nn.Linear(64, 64), nn.GELU()
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(768 + 64), nn.Dropout(dropout), nn.Linear(768 + 64, num_colors)
        )
        self._mode = "finetune"

    def set_warmup_mode(self) -> None:
        self._mode = "warmup"
        for p in self.backbone_features.parameters():
            p.requires_grad = False
        for m in (self.backbone_norm, self.lab_mlp, self.classifier):
            for p in m.parameters():
                p.requires_grad = True
        self.train(self.training)

    def set_finetune_mode(self) -> None:
        self._mode = "finetune"
        for p in self.parameters():
            p.requires_grad = True
        self.train(self.training)

    def train(self, mode: bool = True) -> "AutoVisionColorExpert":
        super().train(mode)
        if mode and self._mode == "warmup":
            self.backbone_features.eval()
        return self

    def build_param_groups(self, backbone_lr: float = BACKBONE_LR, head_lr: float = HEAD_LR, weight_decay: float = WEIGHT_DECAY) -> list[dict[str, Any]]:
        buckets: dict[tuple[str, float, float], list[nn.Parameter]] = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            wd = 0.0 if (param.ndim < 2 or name.endswith(".bias")) else float(weight_decay)
            tier, lr = ("backbone", float(backbone_lr)) if name.startswith("backbone_features.") else ("head", float(head_lr))
            buckets.setdefault((tier, lr, wd), []).append(param)
        return [
            {"params": params, "lr": lr, "weight_decay": wd, "group_name": f"{tier}_wd{wd}"}
            for (tier, lr, wd), params in buckets.items()
        ]

    def forward(self, x: torch.Tensor, lab_feats: torch.Tensor) -> torch.Tensor:
        pooled = self.backbone_avgpool(self.backbone_features(x)).flatten(1)
        vis_feat = self.backbone_norm(pooled.float())
        lab_emb = self.lab_mlp(lab_feats.float())
        return self.classifier(torch.cat([vis_feat, lab_emb], dim=-1))


# --- Dataset Discovery & Caching (RGBA PNG keeps car mask aligned pixel-for-pixel) ---
def _save_rgba_masked_sample(rgb_im: Image.Image, car_mask: np.ndarray, dest_png: Path) -> None:
    rgb = rgb_im.convert("RGB")
    alpha = Image.fromarray((car_mask.astype(np.uint8) * 255), mode="L")
    rgba = rgb.copy()
    rgba.putalpha(alpha)
    cw, ch = rgba.size
    if cw != MASK_IMG_SIZE or ch != MASK_IMG_SIZE:
        rgba = rgba.resize((MASK_IMG_SIZE, MASK_IMG_SIZE), Image.Resampling.BILINEAR)
    dest_png.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(dest_png, format="PNG", compress_level=3)


def discover_vcor_root(search_root: Path = Path("/kaggle/input")) -> Path | None:
    if not search_root.is_dir():
        return None
    for root, dirs, _ in os.walk(search_root):
        if len(Path(root).relative_to(search_root).parts) > 4:
            dirs[:] = []
            continue
        p = Path(root)
        if (p / "train").is_dir() and (p / "test").is_dir() and ((p / "val").is_dir() or (p / "valid").is_dir()):
            return p
    return None


def prepare_vcor_with_yolo_seg(vcor_root: Path, cache_dir: Path, device: str) -> dict[str, list[dict[str, Any]]]:
    seg_model = YOLO("yolov8n-seg.pt")
    splits_out: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    split_dirs = {
        "train": vcor_root / "train",
        "val": vcor_root / "val" if (vcor_root / "val").is_dir() else vcor_root / "valid",
        "test": vcor_root / "test",
    }
    yolo_hits, total_count = 0, 0
    out_root = cache_dir / "vcor_rgba"

    for split_name, sdir in split_dirs.items():
        raw_items: list[tuple[Path, str, Path]] = []
        for cls_dir in sorted(d for d in sdir.iterdir() if d.is_dir()):
            color_lbl = normalize_color_v2(cls_dir.name)
            if color_lbl is None:
                continue
            for img_p in sorted(cls_dir.iterdir()):
                if img_p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    rel_id = f"{split_name}/{cls_dir.name}/{img_p.stem}.png"
                    raw_items.append((img_p, color_lbl, out_root / rel_id))

        pending = [it for it in raw_items if not it[2].is_file()]
        batch_sz = 32
        for i in tqdm(range(0, len(pending), batch_sz), desc=f"YOLOv8n-seg VCoR {split_name}"):
            chunk = pending[i : i + batch_sz]
            imgs, valid_chunk = [], []
            for src_p, lbl, dst_p in chunk:
                try:
                    with Image.open(src_p) as im:
                        imgs.append(im.convert("RGB"))
                    valid_chunk.append((src_p, lbl, dst_p))
                except Exception:
                    continue
            if not imgs:
                continue

            results = seg_model.predict(imgs, classes=[2, 5, 7], conf=0.25, retina_masks=True, device=device, verbose=False)
            for (src_p, lbl, dst_p), im, res in zip(valid_chunk, imgs, results):
                w, h = im.size
                img_area = float(max(w * h, 1))
                chosen_im = im
                chosen_mask: np.ndarray | None = None

                if res.boxes is not None and len(res.boxes) > 0 and res.masks is not None:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    areas = np.maximum(xyxy[:, 2] - xyxy[:, 0], 0.0) * np.maximum(xyxy[:, 3] - xyxy[:, 1], 0.0)
                    best_i = int(np.argmax(areas))
                    area_ratio = float(areas[best_i]) / img_area
                    if area_ratio > 0.12:
                        bx1, by1, bx2, by2 = [float(v) for v in xyxy[best_i]]
                        bw, bh = max(bx2 - bx1, 1.0), max(by2 - by1, 1.0)
                        pad = 0.05
                        cx1 = max(0, int(round(bx1 - pad * bw)))
                        cy1 = max(0, int(round(by1 - pad * bh)))
                        cx2 = min(w, int(round(bx2 + pad * bw)))
                        cy2 = min(h, int(round(by2 + pad * bh)))
                        full_mask = res.masks.data[best_i].cpu().numpy() >= 0.5
                        if full_mask.shape != (h, w):
                            full_mask = np.asarray(
                                Image.fromarray((full_mask.astype(np.uint8) * 255), mode="L").resize((w, h), Image.Resampling.NEAREST)
                            ) >= 128
                        crop_mask = full_mask[cy1:cy2, cx1:cx2]
                        if int(crop_mask.sum()) >= 64:
                            chosen_im = im.crop((cx1, cy1, cx2, cy2))
                            chosen_mask = crop_mask
                            yolo_hits += 1

                if chosen_mask is None:
                    # Fallback interior car mask for tight close-ups
                    chosen_mask = np.zeros((chosen_im.height, chosen_im.width), dtype=bool)
                    chosen_mask[int(0.12 * chosen_im.height) : int(0.88 * chosen_im.height),
                                int(0.12 * chosen_im.width) : int(0.88 * chosen_im.width)] = True

                masked_rgb, final_mask = apply_neutral_gray_mask(chosen_im, chosen_mask, erode_px=2)
                _save_rgba_masked_sample(masked_rgb, final_mask, dst_p)
                im.close()

        for _, lbl, dst_p in raw_items:
            if dst_p.is_file():
                total_count += 1
                splits_out[split_name].append({"path": str(dst_p), "color": lbl, "source": "vcor"})

    del seg_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[VCoR] Cached {total_count:,} car-masked samples (YOLOv8n-seg hit={yolo_hits:,} on uncached pass).")
    return splits_out


def _download_valid_zip(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AutoVision-Color-v2/1.0"})
        with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f, length=1024 * 1024)
        return zipfile.is_zipfile(dest)
    except Exception as exc:
        print(f"[DVM-CAR] Download failed from {url}: {exc}")
        return False


def _find_attached_dvm(search_root: Path = Path("/kaggle/input")) -> tuple[Path | None, Path | None]:
    if not search_root.is_dir():
        return None, None
    for root, dirs, files in os.walk(search_root):
        if len(Path(root).relative_to(search_root).parts) > 4:
            dirs[:] = []
            continue
        p = Path(root)
        if "confirmed_fronts" in root.lower() or "car-fronts" in root.lower() or "dvm" in root.lower():
            for fn in files:
                if fn.lower().endswith(".zip") and "front" in fn.lower():
                    return p / fn, None
            sample_jpgs = [fn for fn in files if fn.lower().endswith((".jpg", ".jpeg")) and "$$" in fn]
            if sample_jpgs:
                return None, search_root / Path(root).relative_to(search_root).parts[0]
    return None, None


def prepare_dvm_confirmed_fronts(cache_dir: Path, seed: int = SEED) -> dict[str, list[dict[str, Any]]]:
    out_root = cache_dir / "dvm_rgba"
    manifest_cache = cache_dir / "dvm_prepared_manifest.json"
    if manifest_cache.is_file():
        cached = json.loads(manifest_cache.read_text(encoding="utf-8"))
        if all(len(v) > 0 for v in cached.values()):
            print(f"[DVM-CAR] Reusing cached manifest ({sum(len(v) for v in cached.values()):,} samples).")
            return cached

    zip_path, folder_path = _find_attached_dvm()
    if zip_path is None and folder_path is None:
        zip_path = cache_dir / "Confirmed_fronts.zip"
        if not zipfile.is_zipfile(zip_path):
            print("[DVM-CAR] Downloading Confirmed_fronts.zip (~730 MB) from Figshare...")
            ok = _download_valid_zip(DVM_FIGSHARE_URL, zip_path)
            if not ok:
                print("[DVM-CAR] Figshare unavailable, trying official Dropbox mirror...")
                ok = _download_valid_zip(DVM_DROPBOX_URL, zip_path)
            if not ok:
                print("[DVM-CAR] Could not obtain Confirmed_fronts; continuing on VCoR only.")
                return {"train": [], "val": [], "test": []}

    rng = random.Random(seed)
    candidates_by_color: dict[str, list[tuple[str, str]]] = defaultdict(list)

    if zip_path is not None and zipfile.is_zipfile(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if not member.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                lbl = parse_dvm_color_from_path(member)
                if lbl is not None:
                    candidates_by_color[lbl].append(("zip", member))
    elif folder_path is not None:
        for p in folder_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                rel = str(p.relative_to(folder_path))
                lbl = parse_dvm_color_from_path(rel)
                if lbl is not None:
                    candidates_by_color[lbl].append(("fs", str(p)))

    selected_tasks: list[tuple[str, str, str, str, Path]] = []
    for lbl in COLOR_CLASSES_V2:
        items = candidates_by_color.get(lbl, [])
        rng.shuffle(items)
        chosen = items[:MAX_DVM_PER_CLASS]
        for mode, ref in chosen:
            key_hash = hashlib.sha1(ref.encode("utf-8")).hexdigest()[:16]
            stem = Path(ref).stem
            group_key = "$$".join(stem.split("$$")[:4]) if "$$" in stem else ref.rsplit("/", 1)[0]
            h_val = int(hashlib.md5(group_key.encode("utf-8")).hexdigest()[:8], 16) % 100
            split_name = "val" if h_val < 6 else ("test" if h_val < 12 else "train")
            dst_png = out_root / split_name / lbl / f"{key_hash}.png"
            selected_tasks.append((mode, ref, lbl, split_name, dst_png))

    print(f"[DVM-CAR] Extracting & masking {len(selected_tasks):,} balanced front views across {len(candidates_by_color)} colors...")

    if zip_path is not None and zipfile.is_zipfile(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            def _proc_zip(task: tuple[str, str, str, str, Path]) -> None:
                _, member, _, _, dst_png = task
                if dst_png.is_file():
                    return
                try:
                    from io import BytesIO
                    raw = zf.read(member)
                    with Image.open(BytesIO(raw)) as im:
                        masked_rgb, car_mask = remove_dvm_white_border_bg(im)
                    _save_rgba_masked_sample(masked_rgb, car_mask, dst_png)
                except Exception:
                    return

            with ThreadPoolExecutor(max_workers=4) as pool:
                list(tqdm(pool.map(_proc_zip, selected_tasks), total=len(selected_tasks), desc="Masking DVM-CAR"))
        if str(zip_path).startswith(str(cache_dir)):
            zip_path.unlink(missing_ok=True)
    else:
        def _proc_fs(task: tuple[str, str, str, str, Path]) -> None:
            _, src_p, _, _, dst_png = task
            if dst_png.is_file():
                return
            try:
                with Image.open(src_p) as im:
                    masked_rgb, car_mask = remove_dvm_white_border_bg(im)
                _save_rgba_masked_sample(masked_rgb, car_mask, dst_png)
            except Exception:
                return

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(tqdm(pool.map(_proc_fs, selected_tasks), total=len(selected_tasks), desc="Masking DVM-CAR"))

    splits_out: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for _, _, lbl, split_name, dst_png in selected_tasks:
        if dst_png.is_file():
            splits_out[split_name].append({"path": str(dst_png), "color": lbl, "source": "dvm"})
    manifest_cache.write_text(json.dumps(splits_out), encoding="utf-8")
    print(f"[DVM-CAR] Ready: tr={len(splits_out['train']):,} | va={len(splits_out['val']):,} | te={len(splits_out['test']):,}")
    return splits_out


class ColorExpertDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]], color_to_id: dict[str, int], is_train: bool, seed: int = SEED) -> None:
        self.samples = samples
        self.color_to_id = color_to_id
        self.is_train = is_train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        s = self.samples[idx]
        lbl = s["color"]
        with Image.open(s["path"]) as rgba_im:
            rgba = rgba_im.convert("RGBA")
        arr = np.asarray(rgba, dtype=np.uint8)
        rgb_pil = Image.fromarray(arr[:, :, :3], mode="RGB")
        mask_pil = Image.fromarray(arr[:, :, 3], mode="L")

        if self.is_train:
            rng = random.Random((self.seed * 1000003) ^ idx ^ hash(s["path"]))
            # Geometry crop on both RGB and alpha mask together
            scale = rng.uniform(0.68, 1.0)
            cw = max(64, int(round(rgb_pil.width * scale)))
            ch = max(64, int(round(rgb_pil.height * scale)))
            top = rng.randint(0, max(0, rgb_pil.height - ch))
            left = rng.randint(0, max(0, rgb_pil.width - cw))
            rgb_pil = TF.resized_crop(rgb_pil, top, left, ch, cw, [MODEL_IMG_SIZE, MODEL_IMG_SIZE], interpolation=TF.InterpolationMode.BICUBIC)
            mask_pil = TF.resized_crop(mask_pil, top, left, ch, cw, [MODEL_IMG_SIZE, MODEL_IMG_SIZE], interpolation=TF.InterpolationMode.NEAREST)
            if rng.random() < 0.5:
                rgb_pil = TF.hflip(rgb_pil)
                mask_pil = TF.hflip(mask_pil)

            mask_bool = np.asarray(mask_pil) >= 128
            if int(mask_bool.sum()) < 32:
                mask_bool = np.ones((MODEL_IMG_SIZE, MODEL_IMG_SIZE), dtype=bool)
            rgb_pil = apply_illuminant_reflection_cast(rgb_pil, lbl, rng, mask_bool=mask_bool, prob=0.45)

            # Mild exposure/contrast jitter on car only, NEVER hue shift!
            if rng.random() < 0.65:
                b_factor = rng.uniform(0.84, 1.18)
                c_factor = rng.uniform(0.85, 1.18)
                jittered = TF.adjust_contrast(TF.adjust_brightness(rgb_pil, b_factor), c_factor)
                j_arr = np.asarray(jittered, dtype=np.uint8)
                o_arr = np.asarray(rgb_pil, dtype=np.uint8).copy()
                o_arr[mask_bool] = j_arr[mask_bool]
                rgb_pil = Image.fromarray(o_arr, mode="RGB")
        else:
            rgb_pil = TF.resize(rgb_pil, [MODEL_IMG_SIZE, MODEL_IMG_SIZE], interpolation=TF.InterpolationMode.BICUBIC)
            mask_pil = TF.resize(mask_pil, [MODEL_IMG_SIZE, MODEL_IMG_SIZE], interpolation=TF.InterpolationMode.NEAREST)
            mask_bool = np.asarray(mask_pil) >= 128

        rgb_arr = np.asarray(rgb_pil, dtype=np.uint8)
        lab_vec = torch.from_numpy(extract_masked_lab_features(rgb_arr, mask_bool))
        x_tensor = TF.normalize(TF.to_tensor(rgb_pil), IMAGENET_MEAN, IMAGENET_STD)
        return x_tensor, lab_vec, self.color_to_id[lbl]


@torch.no_grad()
def evaluate_color_split(
    model: nn.Module,
    samples: list[dict[str, Any]],
    color_to_id: dict[str, int],
    device: str,
    use_tta: bool = True,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if not samples:
        return {"n_eval": 0, "top1": 0.0, "top3": 0.0, "macro_f1": 0.0}, np.empty((0, len(COLOR_CLASSES_V2))), np.empty((0,), dtype=np.int64)
    ds = ColorExpertDataset(samples, color_to_id, is_train=False)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.startswith("cuda"))
    model.eval()
    use_cuda = device.startswith("cuda")
    logits_all, targets_all = [], []
    try:
        for x, lab, y in loader:
            x = x.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                logits = model(x, lab).float()
                if use_tta:
                    logits = 0.5 * (logits + model(torch.flip(x, dims=[-1]), lab).float())
            logits_all.append(logits.cpu().numpy())
            targets_all.append(y.numpy())
    finally:
        del loader

    arr_logits = np.concatenate(logits_all, axis=0)
    arr_y = np.concatenate(targets_all, axis=0)
    preds = np.argmax(arr_logits, axis=1)
    top3_idx = np.argpartition(arr_logits, -3, axis=1)[:, -3:]
    top3 = float(np.mean(np.any(top3_idx == arr_y[:, None], axis=1)))
    macro_f1 = float(f1_score(arr_y, preds, average="macro", zero_division=0))
    return {
        "n_eval": int(len(arr_y)),
        "top1": round(float(np.mean(preds == arr_y)), 4),
        "top3": round(top3, 4),
        "macro_f1": round(macro_f1, 4),
    }, arr_logits, arr_y


def save_confusion_plot(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[str], out_png: Path) -> dict[str, int]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    idx = {c: i for i, c in enumerate(classes)}
    pair_report = {
        "Black_as_Brown": int(cm[idx["Black"], idx["Brown"]]),
        "Brown_as_Black": int(cm[idx["Brown"], idx["Black"]]),
        "Silver_as_Gray": int(cm[idx["Silver"], idx["Gray"]]),
        "Gray_as_Silver": int(cm[idx["Gray"], idx["Silver"]]),
        "White_as_Silver": int(cm[idx["White"], idx["Silver"]]),
    }

    cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (VCoR Test)")
    ax.set_title("AutoVision v2 Color Expert Confusion Matrix")
    for r in range(len(classes)):
        for c in range(len(classes)):
            val = cm_norm[r, c]
            if cm[r, c] > 0:
                ax.text(c, r, f"{val*100:.0f}%\n({cm[r,c]})", ha="center", va="center",
                        fontsize=7, color="white" if val > 0.55 else "#111111")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return pair_report


def main() -> None:
    t0 = time.perf_counter()
    set_seed(SEED)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = "cuda" if n_gpus > 0 else "cpu"
    batch_size = 128 if n_gpus >= 2 else 64
    pin = device == "cuda"
    print(f"[Env] Device={device} | GPUs={n_gpus} | Batch={batch_size}")

    vcor_root = discover_vcor_root()
    if vcor_root is None:
        raise FileNotFoundError("VCoR dataset not found under /kaggle/input/**. Please attach it.")

    vcor_splits = prepare_vcor_with_yolo_seg(vcor_root, CACHE_DIR, device)
    dvm_splits = prepare_dvm_confirmed_fronts(CACHE_DIR, SEED)

    # Oversample real-world VCoR train 3x relative to DVM dealer fronts.
    train_samples = vcor_splits["train"] * VCOR_TRAIN_REPEAT + dvm_splits["train"]
    random.Random(SEED).shuffle(train_samples)
    color_to_id = {c: i for i, c in enumerate(COLOR_CLASSES_V2)}
    print(
        f"[Dataset] Train={len(train_samples):,} (VCoR {len(vcor_splits['train']):,}x{VCOR_TRAIN_REPEAT} + DVM {len(dvm_splits['train']):,}) | "
        f"Val(VCoR={len(vcor_splits['val']):,}, DVM={len(dvm_splits['val']):,}) | "
        f"Test(VCoR={len(vcor_splits['test']):,}, DVM={len(dvm_splits['test']):,})"
    )

    tr_ds = ColorExpertDataset(train_samples, color_to_id, is_train=True, seed=SEED)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin)

    base_model = AutoVisionColorExpert(num_colors=len(COLOR_CLASSES_V2), pretrained=True).to(device)
    run_model: nn.Module = nn.DataParallel(base_model) if n_gpus >= 2 else base_model

    cnt = Counter(color_to_id[s["color"]] for s in train_samples)
    cls_weights = compute_soft_class_weights([cnt[i] for i in range(len(COLOR_CLASSES_V2))], power=0.30).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=pin)

    best_score = -1.0
    best_path = OUT_DIR / "autovision_color_expert_v2.pt"
    history: list[dict[str, Any]] = []

    def _run_phase(phase_name: str, epochs: int, optimizer: torch.optim.Optimizer, scheduler: Any) -> None:
        nonlocal best_score
        for ep in range(1, epochs + 1):
            base_model.train(True)
            loss_sum, steps = 0.0, 0
            for x, lab, y in tqdm(tr_loader, desc=f"{phase_name} Ep {ep}/{epochs}"):
                x = x.to(device, non_blocking=True)
                lab = lab.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=pin):
                    logits = run_model(x, lab)
                    loss = F.cross_entropy(logits.float(), y.long(), weight=cls_weights, label_smoothing=LABEL_SMOOTHING)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.item())
                steps += 1
            if scheduler is not None:
                scheduler.step()

            vcor_val, _, _ = evaluate_color_split(run_model, vcor_splits["val"], color_to_id, device, use_tta=True)
            dvm_val, _, _ = evaluate_color_split(run_model, dvm_splits["val"], color_to_id, device, use_tta=True)
            if dvm_val["n_eval"] > 0:
                score = 0.55 * vcor_val["macro_f1"] + 0.25 * vcor_val["top1"] + 0.20 * dvm_val["macro_f1"]
            else:
                score = 0.65 * vcor_val["macro_f1"] + 0.35 * vcor_val["top1"]

            row = {
                "phase": phase_name, "epoch": ep, "loss": round(loss_sum / max(steps, 1), 4),
                "vcor_val_top1": vcor_val["top1"], "vcor_val_f1": vcor_val["macro_f1"],
                "dvm_val_top1": dvm_val["top1"], "dvm_val_f1": dvm_val["macro_f1"],
                "composite_score": round(score, 4),
            }
            history.append(row)
            print(
                f"  [{phase_name} Ep {ep}] loss={row['loss']:.4f} | VCoR val Top1={row['vcor_val_top1']*100:.2f}% F1={row['vcor_val_f1']*100:.2f}% | "
                f"DVM val Top1={row['dvm_val_top1']*100:.2f}% score={score*100:.2f}%"
            )
            if score > best_score:
                best_score = score
                torch.save({
                    "model_state_dict": base_model.state_dict(),
                    "color_classes": list(COLOR_CLASSES_V2),
                    "color_hex": COLOR_HEX_V2,
                    "val_score": best_score,
                    "model_version": "autovision-color-expert-v2-convnext-tiny",
                }, best_path)
                print(f"  -> Saved new best Color Expert checkpoint ({best_score*100:.2f}%)")

    base_model.set_warmup_mode()
    opt_w = torch.optim.AdamW([p for p in base_model.parameters() if p.requires_grad], lr=WARMUP_LR, weight_decay=WEIGHT_DECAY)
    _run_phase("ColorWarmup", WARMUP_EPOCHS, opt_w, scheduler=None)

    base_model.set_finetune_mode()
    opt_f = torch.optim.AdamW(base_model.build_param_groups())
    sched_f = torch.optim.lr_scheduler.CosineAnnealingLR(opt_f, T_max=FINETUNE_EPOCHS, eta_min=1e-6)
    _run_phase("ColorFineTune", FINETUNE_EPOCHS, opt_f, scheduler=sched_f)

    del tr_loader
    gc.collect()

    # Final Evaluation on Best Checkpoint
    best_state = torch.load(best_path, map_location=device, weights_only=False)["model_state_dict"]
    base_model.load_state_dict(best_state, strict=True)
    eval_model: nn.Module = nn.DataParallel(base_model) if n_gpus >= 2 else base_model

    vcor_test, v_logits, v_targets = evaluate_color_split(eval_model, vcor_splits["test"], color_to_id, device, use_tta=True)
    dvm_test, _, _ = evaluate_color_split(eval_model, dvm_splits["test"], color_to_id, device, use_tta=True)
    pair_report = save_confusion_plot(v_targets, np.argmax(v_logits, axis=1), COLOR_CLASSES_V2, OUT_DIR / "color_confusion_matrix_v2.png")

    (OUT_DIR / "color_label_maps_v2.json").write_text(json.dumps({
        "model_version": "autovision-color-expert-v2-convnext-tiny",
        "color_classes": list(COLOR_CLASSES_V2),
        "color_hex": COLOR_HEX_V2,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "neutral_bg_rgb": NEUTRAL_BG_RGB,
    }, indent=2), encoding="utf-8")

    with open(OUT_DIR / "color_training_history.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    summary = {
        "model_version": "autovision-color-expert-v2-convnext-tiny",
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "best_val_score": best_score,
        "vcor_test_metrics": vcor_test,
        "dvm_test_metrics": dvm_test,
        "key_confusion_pairs_vcor_test": pair_report,
    }
    (OUT_DIR / "color_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== AUTOVISION v2 COLOR EXPERT COMPLETE -> {OUT_DIR} ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
