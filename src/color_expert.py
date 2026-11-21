"""Standalone vehicle color expert: taxonomy, car masking, CIE-Lab paint descriptor, and ConvNeXt-Tiny hybrid head."""

from __future__ import annotations

import random
import re
from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

try:
    from scipy.ndimage import label as _scipy_cc_label
except Exception:  # pragma: no cover - optional speedup fallback
    _scipy_cc_label = None

CLIP_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
NEUTRAL_BG_RGB: tuple[int, int, int] = (128, 128, 128)

COLOR_CLASSES_V2: tuple[str, ...] = (
    "Black",
    "White",
    "Silver",
    "Gray",
    "Blue",
    "Red",
    "Brown",
    "Green",
    "Yellow",
    "Orange",
    "Beige",
    "Purple",
)

COLOR_HEX_V2: dict[str, str] = {
    "Black": "#1C1C1E",
    "White": "#F8FAFC",
    "Silver": "#C0C7CE",
    "Gray": "#6B7280",
    "Blue": "#2563EB",
    "Red": "#DC2626",
    "Brown": "#7C4A2D",
    "Green": "#16A34A",
    "Yellow": "#EAB308",
    "Orange": "#EA580C",
    "Beige": "#D8C39E",
    "Purple": "#7E22CE",
}

COLOR_GROUPS_V2: dict[str, str] = {c: c for c in COLOR_CLASSES_V2}

COLOR_V2_ALIASES: dict[str, str | None] = {
    **{c.lower(): c for c in COLOR_CLASSES_V2},
    # VCoR aliases (consolidate noisy Tan/Gold/Pink ambiguities into clear automotive classes)
    "grey": "Gray",
    "tan": "Beige",
    "gold": "Beige",
    "pink": "Purple",
    "violet": "Purple",
    "cyan": "Blue",
    # DVM-CAR UK dealer aliases
    "bronze": "Brown",
    "maroon": "Red",
    "burgundy": "Red",
    "cream": "Beige",
    "champagne": "Beige",
    "navy": "Blue",
    "turquoise": "Blue",
    "magenta": "Purple",
    "indigo": "Purple",
    "unlisted": None,
    "multicolour": None,
    "multicolor": None,
    "unknown": None,
    "na": None,
    "": None,
}

ACHROMATIC_CLASSES: frozenset[str] = frozenset({"Black", "White", "Silver", "Gray"})


def normalize_color_v2(raw_label: str) -> str | None:
    """Normalizes raw VCoR or DVM-CAR color strings to canonical 12-class taxonomy (None if unlisted)."""
    key = re.sub(r"[\s_-]+", "", (raw_label or "").strip().lower())
    if key in COLOR_V2_ALIASES:
        return COLOR_V2_ALIASES[key]
    raw_key = (raw_label or "").strip().lower()
    return COLOR_V2_ALIASES.get(raw_key)


def parse_dvm_color_from_path(rel_path: str) -> str | None:
    """Extracts normalized color from DVM-CAR folder hierarchy or `$$`-delimited filenames."""
    norm_path = rel_path.replace("\\", "/").strip("/")
    parts = norm_path.split("/")
    filename = parts[-1]
    if "$$" in filename:
        tokens = filename.split("$$")
        if len(tokens) >= 4:
            mapped = normalize_color_v2(tokens[3])
            if mapped is not None:
                return mapped
    # Fallback to parent folders from deepest outward (Brand/Model/Year/Color/img.jpg)
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
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
    return out


def apply_neutral_gray_mask(
    im: Image.Image,
    mask_bool: np.ndarray,
    erode_px: int = 2,
    bg_rgb: tuple[int, int, int] = NEUTRAL_BG_RGB,
) -> tuple[Image.Image, np.ndarray]:
    """Replaces all non-vehicle background pixels with neutral gray to prevent reflection/wall leaks."""
    rgb = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    h, w = rgb.shape[:2]
    m = np.asarray(mask_bool, dtype=bool)
    if m.shape != (h, w):
        m_img = Image.fromarray((m.astype(np.uint8) * 255), mode="L").resize((w, h), Image.Resampling.NEAREST)
        m = np.asarray(m_img) >= 128

    if erode_px > 0 and int(m.sum()) >= 64:
        eroded = _erode_mask_numpy(m, erode_px)
        if int(eroded.sum()) >= 32:
            m = eroded

    if int(m.sum()) == 0:
        # Fallback to centered interior car box if mask completely vanished.
        m = np.zeros((h, w), dtype=bool)
        m[int(0.12 * h) : int(0.88 * h), int(0.12 * w) : int(0.88 * w)] = True

    rgb[~m] = np.asarray(bg_rgb, dtype=np.uint8)
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


def remove_dvm_white_border_bg(
    im: Image.Image,
    white_thresh: int = 242,
    bg_rgb: tuple[int, int, int] = NEUTRAL_BG_RGB,
) -> tuple[Image.Image, np.ndarray]:
    """Flood-fills only border-connected studio white background to neutral gray, preserving white paint."""
    rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    is_white = np.all(rgb >= int(white_thresh), axis=-1)

    if _scipy_cc_label is not None:
        labeled, _ = _scipy_cc_label(is_white)
        border_ids = set(np.unique(np.concatenate([labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]])))
        border_ids.discard(0)
        if border_ids:
            bg_mask = np.isin(labeled, list(border_ids))
        else:
            bg_mask = np.zeros((h, w), dtype=bool)
    else:
        small_w, small_h = min(w, 64), min(h, 64)
        small = np.asarray(
            Image.fromarray((is_white.astype(np.uint8) * 255), mode="L").resize((small_w, small_h), Image.Resampling.NEAREST)
        ) >= 128
        small_bg = _border_flood_bfs(small)
        bg_mask = np.asarray(
            Image.fromarray((small_bg.astype(np.uint8) * 255), mode="L").resize((w, h), Image.Resampling.NEAREST)
        ) >= 128
        bg_mask &= is_white

    car_mask = ~bg_mask
    return apply_neutral_gray_mask(Image.fromarray(rgb, mode="RGB"), car_mask, erode_px=1, bg_rgb=bg_rgb)


def rgb_to_lab_numpy(rgb01: np.ndarray) -> np.ndarray:
    """Converts sRGB float array in [0, 1] with shape (..., 3) into CIE L*a*b* (D65)."""
    arr = np.clip(np.asarray(rgb01, dtype=np.float32), 0.0, 1.0)
    linear = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)

    r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def _f(t: np.ndarray) -> np.ndarray:
        return np.where(t > eps, np.cbrt(np.maximum(t, 0.0)), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = _f(x), _f(y), _f(z)
    l_star = 116.0 * fy - 16.0
    a_star = 500.0 * (fx - fy)
    b_star = 200.0 * (fy - fz)
    return np.stack([l_star, a_star, b_star], axis=-1).astype(np.float32)


def extract_masked_lab_features(
    rgb_uint8: np.ndarray,
    mask_bool: np.ndarray | None = None,
) -> np.ndarray:
    """Computes a calibrated 16-D CIE-Lab & Chroma paint physics vector over vehicle pixels."""
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
        non_bg = np.any(np.abs(arr.astype(np.int16) - np.asarray(NEUTRAL_BG_RGB, dtype=np.int16)) > 2, axis=-1)
        cand = interior & non_bg
        pixels = arr[cand] if int(cand.sum()) >= 32 else arr[interior]

    if len(pixels) == 0:
        pixels = arr.reshape(-1, 3)

    if len(pixels) > 4096:
        stride = max(1, len(pixels) // 4096)
        pixels = pixels[::stride]

    lab = rgb_to_lab_numpy(pixels.astype(np.float32) / 255.0)
    l_vals = lab[:, 0]
    a_vals = lab[:, 1]
    b_vals = lab[:, 2]
    chroma = np.sqrt(a_vals * a_vals + b_vals * b_vals)

    lq15, lq50, lq85 = np.percentile(l_vals, [15, 50, 85])
    l_std = float(np.std(l_vals))
    aq25, aq50, aq75 = np.percentile(a_vals, [25, 50, 75])
    bq25, bq50, bq75 = np.percentile(b_vals, [25, 50, 75])
    cq25, cq50, cq75 = np.percentile(chroma, [25, 50, 75])

    dark_achrom_frac = float(np.mean((l_vals < 28.0) & (chroma < 10.0)))
    bright_achrom_frac = float(np.mean((l_vals > 76.0) & (chroma < 9.0)))
    warm_brown_frac = float(np.mean((l_vals >= 18.0) & (l_vals <= 62.0) & (a_vals > 4.0) & (b_vals > 6.0) & (chroma > 12.0)))

    feats = np.array(
        [
            lq15 / 100.0,
            lq50 / 100.0,
            lq85 / 100.0,
            l_std / 50.0,
            aq25 / 60.0,
            aq50 / 60.0,
            aq75 / 60.0,
            bq25 / 60.0,
            bq50 / 60.0,
            bq75 / 60.0,
            cq25 / 60.0,
            cq50 / 60.0,
            cq75 / 60.0,
            dark_achrom_frac,
            bright_achrom_frac,
            warm_brown_frac,
        ],
        dtype=np.float32,
    )
    return feats


def apply_illuminant_reflection_cast(
    im: Image.Image,
    label: str,
    rng: random.Random,
    mask_bool: np.ndarray | None = None,
    prob: float = 0.45,
) -> Image.Image:
    """Simulates warm brick-wall or cool sky glare on achromatic paints without altering true hue."""
    if label not in ACHROMATIC_CLASSES or rng.random() > float(prob):
        return im

    arr = np.asarray(im.convert("RGB"), dtype=np.float32)
    lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]) / 255.0
    # Specular reflections hit brighter/angled panels stronger than deep body shadows.
    weight = 0.25 + 0.75 * np.clip(lum, 0.0, 1.0)
    strength = rng.uniform(10.0, 26.0)

    if rng.random() < 0.6:
        # Warm brick / sunset cast
        delta = np.stack([strength * weight, 0.42 * strength * weight, -0.30 * strength * weight], axis=-1)
    else:
        # Cool sky / overcast cast
        delta = np.stack([-0.30 * strength * weight, 0.15 * strength * weight, 0.90 * strength * weight], axis=-1)

    shifted = np.clip(arr + delta, 0.0, 255.0).astype(np.uint8)
    if mask_bool is not None:
        m = np.asarray(mask_bool, dtype=bool)
        if m.shape == shifted.shape[:2]:
            orig = np.asarray(im.convert("RGB"), dtype=np.uint8)
            orig_copy = orig.copy()
            orig_copy[m] = shifted[m]
            return Image.fromarray(orig_copy, mode="RGB")
    return Image.fromarray(shifted, mode="RGB")


def compute_soft_class_weights(
    counts: Sequence[int] | np.ndarray | torch.Tensor,
    power: float = 0.30,
    min_weight: float = 0.5,
    max_weight: float = 3.5,
) -> torch.Tensor:
    c = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    total = float(c.sum())
    weights = (total / c) ** float(power)
    weights = weights / max(float(weights.mean()), 1e-12)
    weights = np.clip(weights, float(min_weight), float(max_weight))
    return torch.tensor(weights, dtype=torch.float32)


class AutoVisionColorExpert(nn.Module):
    """ConvNeXt-Tiny + 16-D Masked CIE-Lab hybrid classifier dedicated to automotive paint color."""

    def __init__(
        self,
        num_colors: int = len(COLOR_CLASSES_V2),
        lab_dim: int = 16,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        base = convnext_tiny(weights=weights)
        self.backbone_features = base.features
        self.backbone_avgpool = base.avgpool
        self.backbone_norm = nn.LayerNorm(768)

        self.lab_mlp = nn.Sequential(
            nn.LayerNorm(lab_dim),
            nn.Linear(lab_dim, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(768 + 64),
            nn.Dropout(dropout),
            nn.Linear(768 + 64, num_colors),
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

    def build_param_groups(
        self,
        backbone_lr: float = 1e-4,
        head_lr: float = 1e-3,
        weight_decay: float = 0.02,
    ) -> list[dict[str, Any]]:
        buckets: dict[tuple[str, float, float], list[nn.Parameter]] = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            wd = 0.0 if (param.ndim < 2 or name.endswith(".bias")) else float(weight_decay)
            if name.startswith("backbone_features."):
                tier, lr = "backbone", float(backbone_lr)
            else:
                tier, lr = "head", float(head_lr)
            buckets.setdefault((tier, lr, wd), []).append(param)
        return [
            {"params": params, "lr": lr, "weight_decay": wd, "group_name": f"{tier}_wd{wd}"}
            for (tier, lr, wd), params in buckets.items()
        ]

    def forward(self, x: torch.Tensor, lab_feats: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone_features(x)
        pooled = self.backbone_avgpool(feat_map).flatten(1)
        vis_feat = self.backbone_norm(pooled.float())
        lab_emb = self.lab_mlp(lab_feats.float())
        return self.classifier(torch.cat([vis_feat, lab_emb], dim=-1))
