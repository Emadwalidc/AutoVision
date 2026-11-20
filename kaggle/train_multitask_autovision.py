"""AutoVision - Multi-Task Fine-Tuning on Kaggle GPU (VMMRdb 89k + VCoR Color + ONNX Export).

Run in Kaggle Notebook (Accelerator: GPU P100 / T4, Internet: ON).
Install dependencies first if missing:
# !pip install -q open_clip_torch ultralytics onnx

Inputs (attach under /kaggle/input/**):
  1. autovision-vmmrdb-clean-89k (contains images/ + manifest CSV)
  2. Stage 2 output containing bboxes_yolo.npz (optional; computed automatically if missing)
  3. VCoR Vehicle Color Recognition dataset (landrykezebou/vcor-vehicle-color-recognition-dataset)

Outputs (/kaggle/working/autovision_trained/):
  - autovision_multitask_best.pt
  - autovision_multitask.onnx
  - label_maps.json
  - training_history.csv
  - training_summary.json
  - prototypes_342.npz
  - features_finetuned_openclip_vitb16.npz
  - vmmrdb_test_predictions.npz
  - vcor_test_predictions.npz
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics_cfg")

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from ultralytics import YOLO

# --- Hyperparameters & Paths ---
PHASE1_WARMUP_EPOCHS = 1
WARMUP_HEAD_LR = 3e-3
PHASE1_EPOCHS = 10
BACKBONE_LR = 2.5e-5
PROJ_LR = 1e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.02
PHASE2_COLOR_EPOCHS = 12
COLOR_LR = 1.5e-3
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128
NUM_WORKERS = 2
SEED = 42
OUT_DIR = Path("/kaggle/working/autovision_trained")

# --- Embedded Taxonomies & Model Parity Definitions ---
CLIP_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

YEAR_MIN: int = 1950
YEAR_MAX: int = 2016
NUM_YEAR_BINS: int = YEAR_MAX - YEAR_MIN + 1

BODY_TYPE_CLASSES: tuple[str, ...] = (
    "Sedan",
    "SUV",
    "Pickup",
    "Minivan",
    "Coupe",
    "Hatchback",
    "Wagon",
    "Convertible",
)

_BODY_GROUPS: dict[str, tuple[str, ...]] = {
    "SUV": (
        "mazda__tribute", "mercedes-benz__gl-class", "mercedes-benz__gla-class",
        "mercedes-benz__glk-class", "mercedes-benz__m-class", "mercury__mariner",
        "mercury__mountaineer", "mini__cooper_countryman", "mitsubishi__endeavor",
        "mitsubishi__montero", "mitsubishi__outlander", "nissan__armada",
        "nissan__juke", "nissan__murano", "nissan__pathfinder", "nissan__rogue",
        "nissan__x-trail", "nissan__xterra", "oldsmobile__bravada", "peugeot__3008",
        "plymouth__trailduster", "pontiac__aztek", "pontiac__torrent",
        "porsche__cayenne", "renault__captur", "saab__9-7", "saturn__outlook",
        "saturn__vue", "subaru__crosstrek", "subaru__forester", "subaru__tribeca",
        "suzuki__grand_vitara", "suzuki__samurai", "suzuki__sidekick",
        "suzuki__vitara", "suzuki__x90", "suzuki__xl7", "toyota__4runner",
        "toyota__fj40", "toyota__fjcruiser", "toyota__fortuner",
        "toyota__highlander", "toyota__landcruiser", "toyota__rav4",
        "toyota__sequoia", "toyota__venza", "volkswagen__tiguan",
        "volkswagen__touareg", "volvo__xc60", "volvo__xc90",
        "willys__cj2a", "willys__cj3b",
    ),
    "Pickup": (
        "mitsubishi__mighty max", "mitsubishi__raider", "nissan__720",
        "nissan__d21_hardbody", "nissan__frontier", "nissan__titan",
        "ram__1500", "ram__2500", "ram__3500", "subaru__baja", "subaru__brat",
        "suzuki__equator", "toyota__1-ton truck", "toyota__hilux",
        "toyota__pickup", "toyota__t100", "toyota__tacoma", "toyota__tundra",
    ),
    "Minivan": (
        "mercedes-benz__r-class", "mercedes-benz__sprinter", "mercury__villager",
        "mitsubishi__chariot", "mitsubishi__delica", "nissan__nv200",
        "nissan__presage", "nissan__quest", "oldsmobile__silhouette",
        "opel__vivaro", "plymouth__voyager", "pontiac__montana", "saturn__relay",
        "toyota__estima", "toyota__hiace", "toyota__previa", "toyota__sienna",
        "volkswagen__bus", "volkswagen__camper", "volkswagen__eurovan",
        "volkswagen__routan", "volkswagen__vanagon",
    ),
    "Wagon": (
        "peugeot__407_wagon", "saab__9-2", "subaru__outback",
        "volkswagen__squareback", "volvo__145_wagon", "volvo__245",
        "volvo__v40", "volvo__v50", "volvo__v70", "volvo__v90", "volvo__xc70",
    ),
    "Convertible": (
        "mercedes-benz__sl-class", "mercedes-benz__slk-class", "mg__mgb",
        "mg__midget", "plymouth__prowler", "pontiac__solstice", "porsche__914",
        "porsche__boxster", "rolls-royce__corniche", "saturn__sky",
        "triumph__spitfire", "triumph__tr", "volkswagen__cabrio",
        "volkswagen__dunebuggy", "volkswagen__eos",
    ),
    "Coupe": (
        "mazda__rx7", "mazda__rx8", "mercedes-benz__cl-class",
        "mercedes-benz__clk-class", "mercedes-benz__slr", "mercury__capri",
        "mercury__cougar", "mercury__cyclone", "mitsubishi__3000gt",
        "mitsubishi__eclipse", "mitsubishi__starion", "nash__metropolitan",
        "nissan__200sx", "nissan__240sx", "nissan__300zx", "nissan__350z",
        "nissan__370z", "nissan__gtr", "nissan__skyline", "oldsmobile__442",
        "oldsmobile__starfire", "oldsmobile__toronado", "opel__gt",
        "plymouth__barracuda", "plymouth__duster", "plymouth__gtx",
        "plymouth__laser", "plymouth__roadrunner", "plymouth__scamp",
        "pontiac__fiero", "pontiac__firebird", "pontiac__gto",
        "pontiac__transam", "porsche__911", "porsche__912", "porsche__917",
        "porsche__924", "porsche__928", "porsche__944", "porsche__cayman",
        "saturn__sc", "scion__frs", "scion__tc", "subaru__brz", "subaru__svx",
        "sunbeam__rapier", "toyota__celica", "toyota__mr2", "toyota__paseo",
        "toyota__solara", "toyota__supra", "triumph__gt6",
        "volkswagen__corrado", "volkswagen__karmannghia",
        "volkswagen__scirocco", "volvo__142", "volvo__242", "volvo__780",
        "volvo__c70",
    ),
    "Hatchback": (
        "mazda__speed3", "mercedes-benz__b-class", "mini__austin",
        "mini__cooper", "mitsubishi__colt", "nissan__cube", "nissan__leaf",
        "nissan__pulsar", "opel__astra", "plymouth__colt", "plymouth__horizon",
        "plymouth__sundance", "pontiac__g39", "pontiac__vibe", "pontiac__wave",
        "saab__9-6", "saturn__astra", "scion__iq", "scion__xa", "scion__xb",
        "scion__xd", "smart__fortwo", "subaru__360", "suzuki__reno",
        "suzuki__swift", "suzuki__sx4", "toyota__matrix", "toyota__prius",
        "toyota__starlet", "toyota__yaris", "volkswagen__beetle",
        "volkswagen__golf", "volkswagen__gti", "volkswagen__r32", "volvo__c30",
    ),
    "Sedan": (
        "mazda__speed6", "mercedes-benz__190-class", "mercedes-benz__220d",
        "mercedes-benz__240d", "mercedes-benz__300d", "mercedes-benz__c-class",
        "mercedes-benz__cla-class", "mercedes-benz__cls-class",
        "mercedes-benz__e-class", "mercedes-benz__s-class", "mercury__comet",
        "mercury__grandmarquis", "mercury__marauder", "mercury__milan",
        "mercury__montclair", "mercury__montego", "mercury__monterey",
        "mercury__mystique", "mercury__parklane", "mercury__sable",
        "mercury__topaz", "mercury__tracer", "mitsubishi__diamante",
        "mitsubishi__galant", "mitsubishi__lancer", "mitsubishi__mirage",
        "morris__minor", "nissan__altima", "nissan__cima", "nissan__maxima",
        "nissan__sentra", "nissan__stanza", "nissan__versa",
        "oldsmobile__achieva", "oldsmobile__alero", "oldsmobile__aurora",
        "oldsmobile__cutlass", "oldsmobile__eightyeight", "oldsmobile__intrigue",
        "oldsmobile__ninetyeight", "oldsmobile__omega", "opel__kadett",
        "packard__super8", "plymouth__acclaim", "plymouth__belvedere",
        "plymouth__breeze", "plymouth__cranbrook", "plymouth__deluxe",
        "plymouth__fury", "plymouth__neon", "plymouth__reliant",
        "plymouth__satellite", "plymouth__savoy", "plymouth__valiant",
        "plymouth__volare", "pontiac__bonneville", "pontiac__catalina",
        "pontiac__chieftain", "pontiac__g5", "pontiac__g6", "pontiac__g8",
        "pontiac__grandam", "pontiac__grandprix", "pontiac__grandville",
        "pontiac__lemans", "pontiac__parisienne", "pontiac__pursuit",
        "pontiac__starchief", "pontiac__sunbird", "pontiac__sunfire",
        "pontiac__tempest", "pontiac__ventura", "porsche__panamera",
        "rambler__american", "rambler__rogue", "rolls-royce__silvershadow",
        "rolls-royce__silverspur", "saab__9-3", "saab__9-5", "saab__900",
        "saturn__aura", "saturn__ion", "saturn__l-series", "saturn__s-series",
        "studebaker__champion", "studebaker__commander", "studebaker__lark",
        "subaru__gl10", "subaru__impreza", "subaru__legacy", "subaru__wrx",
        "suzuki__aerio", "suzuki__esteem", "suzuki__forenza", "suzuki__kizashi",
        "suzuki__verona", "tesla__s", "toyota__aristo", "toyota__avalon",
        "toyota__camry", "toyota__corolla", "toyota__cressida", "toyota__echo",
        "toyota__tercel", "toyota__tiara", "triumph__vitesse",
        "volkswagen__atlantic", "volkswagen__cc", "volkswagen__fox",
        "volkswagen__jetta", "volkswagen__passat", "volkswagen__phaeton",
        "volvo__164", "volvo__240", "volvo__244", "volvo__540", "volvo__670",
        "volvo__740", "volvo__760", "volvo__840", "volvo__850", "volvo__940",
        "volvo__960", "volvo__c40", "volvo__pv444", "volvo__s40", "volvo__s60", "volvo__s70",
        "volvo__s80", "volvo__s90",
    ),
}

MODEL_BODY_TYPE: dict[str, str] = {}
for _btype, _models in _BODY_GROUPS.items():
    for _cls in _models:
        if _cls in MODEL_BODY_TYPE:
            raise ValueError(f"Duplicate class in _BODY_GROUPS: {_cls}")
        MODEL_BODY_TYPE[_cls] = _btype


def resolve_body_type(
    canonical_class: str,
    raw_model: str,
    body_hint: str,
    raw_year: int | str,
    exclude_reason: str,
) -> str:
    """Resolves exact Body Type from explicit overrides and complete 342-class mapping."""
    if exclude_reason != "":
        return ""
    if body_hint in ("Wagon", "Convertible"):
        return body_hint
    if raw_model.strip().lower() in ("sw2", "lw200", "lw300"):
        return "Wagon"
    if canonical_class == "mercury__monterey" and int(raw_year) >= 2004:
        return "Minivan"
    return MODEL_BODY_TYPE[canonical_class]


COLOR_CLASSES: tuple[str, ...] = (
    "Beige",
    "Black",
    "Blue",
    "Brown",
    "Gold",
    "Gray",
    "Green",
    "Orange",
    "Pink",
    "Purple",
    "Red",
    "Silver",
    "Tan",
    "White",
    "Yellow",
)

COLOR_ALIASES: dict[str, str] = {
    **{c.lower(): c for c in COLOR_CLASSES},
    "grey": "Gray",
    "violet": "Purple",
    "cyan": "Blue",
}

COLOR_GROUPS_11: dict[str, str] = {
    "Beige": "Beige/Gold",
    "Black": "Black",
    "Blue": "Blue",
    "Brown": "Brown",
    "Gold": "Beige/Gold",
    "Gray": "Gray",
    "Green": "Green",
    "Orange": "Orange",
    "Pink": "Purple/Pink",
    "Purple": "Purple/Pink",
    "Red": "Red",
    "Silver": "Silver",
    "Tan": "Beige/Gold",
    "White": "White",
    "Yellow": "Yellow",
}

COLOR_HEX: dict[str, str] = {
    "Beige": "#E5D3B3",
    "Black": "#1C1C1E",
    "Blue": "#2563EB",
    "Brown": "#7C4A2D",
    "Gold": "#D4AF37",
    "Gray": "#6B7280",
    "Green": "#16A34A",
    "Orange": "#EA580C",
    "Pink": "#EC4899",
    "Purple": "#7E22CE",
    "Red": "#DC2626",
    "Silver": "#C0C7CE",
    "Tan": "#C8A47E",
    "White": "#F8FAFC",
    "Yellow": "#EAB308",
}


def normalize_color_label(raw: str) -> str:
    """Normalizes a raw color folder/class label into canonical COLOR_CLASSES."""
    key = raw.strip().lower()
    if key not in COLOR_ALIASES:
        raise KeyError(f"Unknown vehicle color label: {raw!r}")
    return COLOR_ALIASES[key]


def compute_soft_class_weights(
    counts: Sequence[int] | np.ndarray | torch.Tensor,
    power: float = 0.5,
    min_weight: float = 0.5,
    max_weight: float = 5.0,
) -> torch.Tensor:
    """Computes softened inverse-frequency class weights normalized around 1.0."""
    c = np.asarray(counts, dtype=np.float64)
    c = np.maximum(c, 1.0)
    total = float(c.sum())
    weights = (total / c) ** float(power)
    weights = weights / max(float(weights.mean()), 1e-12)
    weights = np.clip(weights, float(min_weight), float(max_weight))
    return torch.tensor(weights, dtype=torch.float32)


def year_to_gaussian_target(
    years: torch.Tensor,
    year_min: int = YEAR_MIN,
    year_max: int = YEAR_MAX,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Builds normalized Gaussian label distributions over [year_min, year_max]."""
    bins = torch.arange(year_min, year_max + 1, device=years.device, dtype=torch.float32).unsqueeze(0)
    y = years.float().clamp(min=float(year_min), max=float(year_max)).unsqueeze(1)
    diff = (bins - y) / float(sigma)
    weights = torch.exp(-0.5 * diff * diff)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def year_logits_to_expected(
    year_logits: torch.Tensor,
    year_min: int = YEAR_MIN,
    year_max: int = YEAR_MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes expected year and predictive standard deviation from bin logits."""
    logits_f = year_logits.float()
    probs = F.softmax(logits_f, dim=-1)
    bins = torch.arange(year_min, year_max + 1, device=logits_f.device, dtype=torch.float32)
    expected = (probs * bins).sum(dim=-1)
    var = (probs * (bins.unsqueeze(0) - expected.unsqueeze(1)).pow(2)).sum(dim=-1).clamp_min(0.0)
    return expected, torch.sqrt(var)


def dldl_year_loss(
    year_logits: torch.Tensor,
    target_years: torch.Tensor,
    valid_mask: torch.Tensor,
    year_min: int = YEAR_MIN,
    year_max: int = YEAR_MAX,
    sigma: float = 1.5,
    l1_weight: float = 0.2,
) -> torch.Tensor:
    """Computes masked DLDL Gaussian KL + SmoothL1 expectation loss for vehicle year."""
    mask = valid_mask.bool()
    if not torch.any(mask):
        return year_logits.sum() * 0.0

    logits_v = year_logits[mask].float()
    years_v = target_years[mask]
    target_probs = year_to_gaussian_target(years_v, year_min=year_min, year_max=year_max, sigma=sigma)
    log_probs = F.log_softmax(logits_v, dim=-1)
    kl_loss = -(target_probs * log_probs).sum(dim=-1).mean()
    expected_v, _ = year_logits_to_expected(logits_v, year_min=year_min, year_max=year_max)
    years_clamped = years_v.float().clamp(min=float(year_min), max=float(year_max))
    l1_loss = F.smooth_l1_loss(expected_v, years_clamped, beta=1.0)
    return kl_loss + float(l1_weight) * l1_loss


class CosFaceHead(nn.Module):
    """Cosine margin classifier for fine-grained vehicle model recognition."""

    def __init__(
        self,
        in_features: int = 512,
        out_features: int = 225,
        scale: float = 30.0,
        margin: float = 0.35,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = float(scale)
        self.margin = float(margin)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        x_norm = F.normalize(x.float(), dim=-1)
        w_norm = F.normalize(self.weight.float(), dim=-1)
        cosine = F.linear(x_norm, w_norm).clamp(-1.0, 1.0)
        if targets is not None:
            one_hot = F.one_hot(targets.long(), num_classes=self.out_features).to(cosine.dtype)
            cosine = cosine - self.margin * one_hot
        return cosine * self.scale


class SpatialAttentionColorHead(nn.Module):
    """Multi-scale paint-focused color head combining ConvRGB branch, ViT tokens, and RGB moments."""

    def __init__(
        self,
        vit_dim: int = 768,
        num_colors: int = len(COLOR_CLASSES),
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.register_buffer("clip_mean", torch.tensor(CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor(CLIP_STD, dtype=torch.float32).view(1, 3, 1, 1))

        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.attn_gate = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        fused_dim = 128 + vit_dim * 2 + 6
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_colors),
        )

    def forward(
        self,
        x: torch.Tensor,
        tok_mid: torch.Tensor,
        tok_late: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_rgb = (x.float() * self.clip_std + self.clip_mean).clamp(0.0, 1.0)
        conv_feat = self.conv_stem(x_rgb)
        bsz, _, gh, gw = conv_feat.shape
        conv_tokens = conv_feat.flatten(2).transpose(1, 2)

        attn_logits = self.attn_gate(tok_late.float()).squeeze(-1)
        attn = F.softmax(attn_logits, dim=-1)
        attn_unsq = attn.unsqueeze(-1)

        conv_pooled = (conv_tokens * attn_unsq).sum(dim=1)
        mid_pooled = (tok_mid.float() * attn_unsq).sum(dim=1)
        late_pooled = tok_late.float().mean(dim=1)

        rgb_14 = F.avg_pool2d(x_rgb, kernel_size=16, stride=16)
        rgb_patches = rgb_14.flatten(2).transpose(1, 2)
        rgb_mean = (rgb_patches * attn_unsq).sum(dim=1)
        rgb_var = ((rgb_patches - rgb_mean.unsqueeze(1)).pow(2) * attn_unsq).sum(dim=1).clamp_min(1e-6)
        rgb_std = torch.sqrt(rgb_var)

        fused = torch.cat([conv_pooled, mid_pooled, late_pooled, rgb_mean, rgb_std], dim=-1)
        logits = self.classifier(fused)
        return logits, attn.view(bsz, gh, gw)


def extract_vit_features_and_tokens(
    visual: nn.Module,
    x: torch.Tensor,
    indices: tuple[int, int] = (5, 11),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extracts ViT projection output alongside mid and late 196-patch token sequences."""
    if hasattr(visual, "forward_intermediates"):
        out = visual.forward_intermediates(
            x,
            indices=list(indices),
            normalize_intermediates=False,
            output_fmt="NLC",
        )
        intermediates = out["image_intermediates"]
        return out["image_features"], intermediates[0], intermediates[1]

    captured: dict[int, torch.Tensor] = {}
    handles = []
    bsz = x.shape[0]
    resblocks = visual.transformer.resblocks

    for blk_idx in indices:
        def _make_hook(i: int):
            def _hook(_mod, _inp, output):
                t = output[0] if isinstance(output, tuple) else output
                if t.dim() == 3 and t.shape[0] != bsz and t.shape[1] == bsz:
                    t = t.transpose(0, 1)
                captured[i] = t[:, 1:, :]
            return _hook

        handles.append(resblocks[blk_idx].register_forward_hook(_make_hook(blk_idx)))

    try:
        feat = visual(x)
    finally:
        for h in handles:
            h.remove()

    return feat, captured[indices[0]], captured[indices[1]]


def _set_module_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


class AutoVisionMultiTaskNet(nn.Module):
    """Unified ViT-B/16 multi-task architecture for Make, Model, Body Type, Year, and Color."""

    def __init__(
        self,
        num_makes: int = 22,
        num_models: int = 225,
        num_bodies: int = 8,
        num_years: int = NUM_YEAR_BINS,
        num_colors: int = len(COLOR_CLASSES),
        pretrained: str | None = "laion2b_s34b_b88k",
        cosface_scale: float = 30.0,
        cosface_margin: float = 0.35,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        clip_model = open_clip.create_model("ViT-B-16", pretrained=pretrained)
        self.visual = clip_model.visual
        embed_dim = int(getattr(self.visual, "output_dim", 512))
        vit_dim = int(getattr(getattr(self.visual, "transformer", None), "width", 768))

        self.make_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, num_makes),
        )
        self.model_head = CosFaceHead(
            embed_dim,
            num_models,
            scale=cosface_scale,
            margin=cosface_margin,
        )
        self.body_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, num_bodies),
        )
        self.year_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, num_years),
        )
        self.color_head = SpatialAttentionColorHead(
            vit_dim=vit_dim,
            num_colors=num_colors,
            hidden_dim=256,
            dropout=head_dropout,
        )
        self._phase_mode = "full"

    def set_phase1_warmup_mode(self) -> None:
        self._phase_mode = "phase1_warmup"
        _set_module_requires_grad(self.visual, False)
        for head in (self.make_head, self.model_head, self.body_head, self.year_head):
            _set_module_requires_grad(head, True)
        _set_module_requires_grad(self.color_head, False)
        self.train(self.training)

    def set_phase1_finetune_mode(self) -> None:
        self._phase_mode = "phase1_finetune"
        _set_module_requires_grad(self.visual, True)
        for head in (self.make_head, self.model_head, self.body_head, self.year_head):
            _set_module_requires_grad(head, True)
        _set_module_requires_grad(self.color_head, False)
        self.train(self.training)

    def set_phase2_color_mode(self) -> None:
        self._phase_mode = "phase2_color"
        _set_module_requires_grad(self.visual, False)
        for head in (self.make_head, self.model_head, self.body_head, self.year_head):
            _set_module_requires_grad(head, False)
        _set_module_requires_grad(self.color_head, True)
        self.train(self.training)

    def train(self, mode: bool = True) -> "AutoVisionMultiTaskNet":
        super().train(mode)
        if mode:
            if self._phase_mode == "phase1_warmup":
                self.visual.eval()
                self.color_head.eval()
            elif self._phase_mode == "phase1_finetune":
                self.color_head.eval()
            elif self._phase_mode == "phase2_color":
                self.visual.eval()
                self.make_head.eval()
                self.model_head.eval()
                self.body_head.eval()
                self.year_head.eval()
        return self

    def build_phase1_param_groups(
        self,
        backbone_lr: float = 2.5e-5,
        proj_lr: float = 1e-4,
        head_lr: float = 1e-3,
        weight_decay: float = 0.02,
    ) -> list[dict[str, Any]]:
        buckets: dict[tuple[str, float, float], list[nn.Parameter]] = {}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_no_decay = (
                param.ndim < 2
                or name.endswith(".bias")
                or "positional_embedding" in name
                or "class_embedding" in name
            )
            wd = 0.0 if is_no_decay else float(weight_decay)

            if name.startswith("visual."):
                is_proj = any(
                    k in name for k in ("visual.proj", "visual.ln_post", "visual.attn_pool")
                )
                lr = float(proj_lr if is_proj else backbone_lr)
                tier = "proj" if is_proj else "backbone"
            else:
                lr = float(head_lr)
                tier = "head"

            buckets.setdefault((tier, lr, wd), []).append(param)

        return [
            {"params": params, "lr": lr, "weight_decay": wd, "group_name": f"{tier}_wd{wd}"}
            for (tier, lr, wd), params in buckets.items()
        ]

    def forward(
        self,
        x: torch.Tensor,
        model_targets: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> dict[str, torch.Tensor]:
        raw_feat, tok_mid, tok_late = extract_vit_features_and_tokens(self.visual, x, indices=(5, 11))
        emb = F.normalize(raw_feat.float(), dim=-1)

        make_logits = self.make_head(emb)
        model_logits = self.model_head(emb, targets=model_targets)
        body_logits = self.body_head(emb)
        year_logits = self.year_head(emb)
        color_logits, color_attn = self.color_head(x, tok_mid.detach(), tok_late.detach())

        out = {
            "embedding": emb,
            "make_logits": make_logits,
            "model_logits": model_logits,
            "body_logits": body_logits,
            "year_logits": year_logits,
            "color_logits": color_logits,
        }
        if return_attn:
            out["color_attn"] = color_attn
        return out


class AutoVisionOnnxWrapper(nn.Module):
    """Tuple-output wrapper for clean ONNX export."""

    def __init__(self, core: AutoVisionMultiTaskNet) -> None:
        super().__init__()
        self.core = core

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.core(x, model_targets=None, return_attn=False)
        return (
            out["embedding"],
            out["make_logits"],
            out["model_logits"],
            out["body_logits"],
            out["year_logits"],
            out["color_logits"],
        )


class MultiTaskVmmrdbLoss(nn.Module):
    """Combined Phase-1 loss across Model (CosFace), Make, Body Type, and masked Year DLDL."""

    def __init__(
        self,
        make_weights: torch.Tensor | None = None,
        body_weights: torch.Tensor | None = None,
        w_model: float = 1.0,
        w_make: float = 0.5,
        w_body: float = 0.7,
        w_year: float = 0.7,
        label_smoothing: float = 0.05,
        year_min: int = YEAR_MIN,
        year_max: int = YEAR_MAX,
    ) -> None:
        super().__init__()
        self.w_model = float(w_model)
        self.w_make = float(w_make)
        self.w_body = float(w_body)
        self.w_year = float(w_year)
        self.label_smoothing = float(label_smoothing)
        self.year_min = int(year_min)
        self.year_max = int(year_max)

        self.register_buffer("make_weights", make_weights.float() if make_weights is not None else None)
        self.register_buffer("body_weights", body_weights.float() if body_weights is not None else None)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        make_y: torch.Tensor,
        model_y: torch.Tensor,
        body_y: torch.Tensor,
        year_y: torch.Tensor,
        year_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_model = F.cross_entropy(
            outputs["model_logits"].float(),
            model_y.long(),
            label_smoothing=self.label_smoothing,
        )
        loss_make = F.cross_entropy(
            outputs["make_logits"].float(),
            make_y.long(),
            weight=self.make_weights,
            label_smoothing=self.label_smoothing,
        )
        loss_body = F.cross_entropy(
            outputs["body_logits"].float(),
            body_y.long(),
            weight=self.body_weights,
            label_smoothing=self.label_smoothing,
        )
        loss_year = dldl_year_loss(
            outputs["year_logits"],
            year_y,
            year_valid,
            year_min=self.year_min,
            year_max=self.year_max,
        )

        total = (
            self.w_model * loss_model
            + self.w_make * loss_make
            + self.w_body * loss_body
            + self.w_year * loss_year
        )
        parts = {
            "total": float(total.detach().item()),
            "model": float(loss_model.detach().item()),
            "make": float(loss_make.detach().item()),
            "body": float(loss_body.detach().item()),
            "year": float(loss_year.detach().item()),
        }
        return total, parts


# --- Auto-Discovery & Data Preparation Helpers ---
def _bounded_walk_files(search_root: Path, filename: str, max_depth: int = 5) -> list[Path]:
    matches: list[Path] = []
    if not search_root.is_dir():
        return matches
    for root, dirs, files in os.walk(search_root):
        rel_parts = Path(root).relative_to(search_root).parts
        if len(rel_parts) > max_depth:
            dirs[:] = []
            continue
        if filename in files:
            matches.append(Path(root) / filename)
    return sorted(matches)


def find_vmmrdb_manifest_and_root(
    search_root: Path = Path("/kaggle/input"),
) -> tuple[Path, Path, list[dict[str, Any]]]:
    """Discovers multitask or Stage-2 VMMRdb manifest CSV and matching image root."""
    mt_cands = _bounded_walk_files(search_root, "local_vmmrdb_multitask_manifest.csv", max_depth=5)
    base_cands = _bounded_walk_files(search_root, "local_vmmrdb_manifest.csv", max_depth=5)
    cands = mt_cands or base_cands
    if not cands:
        raise FileNotFoundError(
            f"Neither local_vmmrdb_multitask_manifest.csv nor local_vmmrdb_manifest.csv found under {search_root}"
        )
    manifest_path = cands[0]

    rows: list[dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("is_clean", "")).strip().lower() != "true":
                continue
            orig_rel = row["rel_path"].replace("\\", "/")
            img_rel = (row.get("kaggle_rel_path") or orig_rel).replace("\\", "/")
            btype = (row.get("body_type") or "").strip()
            if not btype:
                btype = resolve_body_type(
                    canonical_class=row["canonical_class"],
                    raw_model=row.get("raw_model", ""),
                    body_hint=row.get("body_hint", ""),
                    raw_year=row["raw_year"],
                    exclude_reason=row.get("exclude_reason", ""),
                )
            if btype not in BODY_TYPE_CLASSES:
                raise ValueError(f"Invalid body_type {btype!r} for {orig_rel}")
            rows.append(
                {
                    "rel_path": orig_rel,
                    "img_rel_path": img_rel,
                    "make": row["make"],
                    "canonical_class": row["canonical_class"],
                    "raw_year": int(row["raw_year"]),
                    "year_valid": str(row.get("year_valid", "")).strip().lower() == "true",
                    "split": row["split"],
                    "body_type": btype,
                }
            )
    if not rows:
        raise ValueError(f"Zero clean rows found in {manifest_path}")

    sample_rels = [
        rows[0]["img_rel_path"],
        rows[len(rows) // 2]["img_rel_path"],
        rows[-1]["img_rel_path"],
    ]
    dir_cands = [manifest_path.parent, manifest_path.parent / "images"]
    for root, dirs, _ in os.walk(search_root):
        if len(Path(root).relative_to(search_root).parts) > 4:
            dirs[:] = []
            continue
        dir_cands.append(Path(root))

    seen: set[Path] = set()
    for cand in dir_cands:
        if cand in seen or not cand.is_dir():
            continue
        seen.add(cand)
        if all((cand / rp).is_file() for rp in sample_rels):
            return manifest_path, cand, rows

    raise FileNotFoundError(
        f"Could not locate VMMRdb image root under {search_root} matching {sample_rels}"
    )


def compute_vehicle_crop_box(
    img_w: int,
    img_h: int,
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    min_area_ratio: float = 0.08,
    pad: float = 0.05,
) -> tuple[tuple[int, int, int, int], list[float], bool]:
    """Selects largest YOLO vehicle box with padding, or 85% center fallback for VMMRdb."""
    img_area = float(max(img_w * img_h, 1))
    best_conf = 0.0
    best_ratio = 0.0

    if boxes_xyxy.ndim == 2 and len(boxes_xyxy) > 0:
        areas = np.maximum(boxes_xyxy[:, 2] - boxes_xyxy[:, 0], 0.0) * np.maximum(
            boxes_xyxy[:, 3] - boxes_xyxy[:, 1], 0.0
        )
        idx = int(np.argmax(areas))
        bx1, by1, bx2, by2 = [float(v) for v in boxes_xyxy[idx]]
        best_conf = float(confs[idx]) if len(confs) > idx else 0.0
        bw, bh = max(bx2 - bx1, 0.0), max(by2 - by1, 0.0)
        best_ratio = (bw * bh) / img_area

        if best_ratio > min_area_ratio:
            x1 = max(0, int(bx1 - pad * bw))
            y1 = max(0, int(by1 - pad * bh))
            x2 = min(img_w, int(bx2 + pad * bw))
            y2 = min(img_h, int(by2 + pad * bh))
            x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
            return (x1, y1, x2, y2), [float(x1), float(y1), float(x2), float(y2), best_conf, best_ratio], True

    mw, mh = int(img_w * 0.075), int(img_h * 0.075)
    x1, y1 = min(mw, max(img_w - 1, 0)), min(mh, max(img_h - 1, 0))
    x2, y2 = max(x1 + 1, img_w - mw), max(y1 + 1, img_h - mh)
    return (x1, y1, x2, y2), [float(x1), float(y1), float(x2), float(y2), best_conf, best_ratio], False


def compute_vcor_crop_box(
    img_w: int,
    img_h: int,
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    min_area_ratio: float = 0.15,
    pad: float = 0.05,
) -> tuple[tuple[int, int, int, int], list[float], bool]:
    """Selects YOLO box for VCoR (>15% area) or keeps full image to protect close-up crops."""
    img_area = float(max(img_w * img_h, 1))
    best_conf = 0.0
    best_ratio = 0.0

    if boxes_xyxy.ndim == 2 and len(boxes_xyxy) > 0:
        areas = np.maximum(boxes_xyxy[:, 2] - boxes_xyxy[:, 0], 0.0) * np.maximum(
            boxes_xyxy[:, 3] - boxes_xyxy[:, 1], 0.0
        )
        idx = int(np.argmax(areas))
        bx1, by1, bx2, by2 = [float(v) for v in boxes_xyxy[idx]]
        best_conf = float(confs[idx]) if len(confs) > idx else 0.0
        bw, bh = max(bx2 - bx1, 0.0), max(by2 - by1, 0.0)
        best_ratio = (bw * bh) / img_area

        if best_ratio > min_area_ratio:
            x1 = max(0, int(bx1 - pad * bw))
            y1 = max(0, int(by1 - pad * bh))
            x2 = min(img_w, int(bx2 + pad * bw))
            y2 = min(img_h, int(by2 + pad * bh))
            x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
            return (x1, y1, x2, y2), [float(x1), float(y1), float(x2), float(y2), best_conf, best_ratio], True

    return (0, 0, img_w, img_h), [0.0, 0.0, float(img_w), float(img_h), best_conf, best_ratio], False


class RawImagePathDataset(Dataset):
    def __init__(self, img_paths: list[Path]) -> None:
        self.img_paths = img_paths

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        with Image.open(self.img_paths[idx]) as im:
            rgb = im.convert("RGB")
        return idx, rgb


def _collate_raw(batch):
    indices, imgs = zip(*batch)
    return list(indices), list(imgs)


def _run_yolo_on_paths(
    img_paths: list[Path],
    rel_paths: list[str],
    out_path: Path,
    device: str,
    crop_fn,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    yolo = YOLO("yolov8n.pt")
    n = len(rel_paths)
    bboxes = np.zeros((n, 6), dtype=np.float32)
    used_yolo = np.zeros(n, dtype=bool)
    loader = DataLoader(
        RawImagePathDataset(img_paths),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=_collate_raw,
    )
    for indices, imgs in tqdm(loader, desc=desc):
        results = yolo.predict(imgs, classes=[2, 5, 7], conf=0.25, device=device, verbose=False)
        for idx, im, res in zip(indices, imgs, results):
            w, h = im.size
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
            else:
                xyxy = np.empty((0, 4), dtype=np.float32)
                confs = np.empty((0,), dtype=np.float32)
            _, row6, y_flag = crop_fn(w, h, xyxy, confs)
            bboxes[idx] = row6
            used_yolo[idx] = y_flag
            im.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, bboxes=bboxes, used_yolo=used_yolo, rel_paths=np.asarray(rel_paths, dtype=str))
    return bboxes, used_yolo


def _match_cached_bboxes(npz_path: Path, rel_paths: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    with np.load(npz_path, allow_pickle=False) as d:
        cached_rels = [str(p).replace("\\", "/") for p in d["rel_paths"]]
        if len(cached_rels) < len(rel_paths):
            return None
        lookup = {rp: i for i, rp in enumerate(cached_rels)}
        if not all(rp in lookup for rp in rel_paths):
            return None
        order = np.array([lookup[rp] for rp in rel_paths], dtype=np.int64)
        return d["bboxes"][order].astype(np.float32), d["used_yolo"][order].astype(bool)


def find_or_compute_vmmrdb_bboxes(
    rows: list[dict[str, Any]],
    img_root: Path,
    out_dir: Path,
    device: str,
    search_roots: tuple[Path, ...] = (Path("/kaggle/input"), Path("/kaggle/working")),
) -> tuple[np.ndarray, np.ndarray]:
    """Reuses attached bboxes_yolo.npz if 100% aligned, else computes via YOLOv8n."""
    rel_paths = [r["rel_path"] for r in rows]
    for sroot in search_roots:
        for cand in _bounded_walk_files(sroot, "bboxes_yolo.npz", max_depth=5):
            matched = _match_cached_bboxes(cand, rel_paths)
            if matched is not None:
                bboxes, used = matched
                print(f"[BBox] Reusing {cand} (matched {len(rel_paths):,}, yolo_hit={used.mean()*100:.2f}%)")
                return bboxes, used

    print("[BBox] Precomputed bboxes_yolo.npz not found; running YOLOv8n once...")
    img_paths = [img_root / r["img_rel_path"] for r in rows]
    return _run_yolo_on_paths(
        img_paths, rel_paths, out_dir / "bboxes_yolo.npz", device, compute_vehicle_crop_box, "YOLOv8n VMMRdb"
    )


def find_vcor_root(search_root: Path = Path("/kaggle/input")) -> Path | None:
    """Auto-discovers VCoR train/val/test directory under search_root."""
    if not search_root.is_dir():
        return None
    for root, dirs, _ in os.walk(search_root):
        rpath = Path(root)
        if len(rpath.relative_to(search_root).parts) > 4:
            dirs[:] = []
            continue
        train_dir = rpath / "train"
        val_dir = rpath / "val" if (rpath / "val").is_dir() else rpath / "valid"
        test_dir = rpath / "test"
        if not (train_dir.is_dir() and val_dir.is_dir() and test_dir.is_dir()):
            continue
        recognized = 0
        for sub in train_dir.iterdir():
            if not sub.is_dir() or sub.name.startswith("."):
                continue
            try:
                normalize_color_label(sub.name)
                recognized += 1
            except KeyError:
                pass
        if recognized >= 8:
            return rpath
    return None


def scan_vcor_split(split_dir: Path) -> list[dict[str, Any]]:
    """Scans a VCoR split folder and normalizes directory labels to canonical colors."""
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    samples: list[dict[str, Any]] = []
    for sub in sorted(split_dir.iterdir(), key=lambda p: p.name.lower()):
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        color_name = normalize_color_label(sub.name)
        for p in sorted(sub.iterdir(), key=lambda q: q.name.lower()):
            if p.is_file() and p.suffix.lower() in valid_exts:
                samples.append(
                    {
                        "path": p,
                        "rel_path": f"{split_dir.name}/{sub.name}/{p.name}",
                        "color": color_name,
                        "split": "val" if split_dir.name == "valid" else split_dir.name,
                    }
                )
    return samples


def run_vcor_yolo_stage(
    samples: list[dict[str, Any]],
    out_dir: Path,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes or reuses YOLO crop boxes for VCoR images (with full-image fallback)."""
    rel_paths = [s["rel_path"] for s in samples]
    cache_path = out_dir / "vcor_bboxes_yolo.npz"
    if cache_path.is_file():
        matched = _match_cached_bboxes(cache_path, rel_paths)
        if matched is not None:
            print(f"[VCoR BBox] Reusing cached {cache_path} (yolo_hit={matched[1].mean()*100:.2f}%)")
            return matched

    img_paths = [Path(s["path"]) for s in samples]
    return _run_yolo_on_paths(
        img_paths, rel_paths, cache_path, device, compute_vcor_crop_box, "YOLOv8n VCoR"
    )


# --- Datasets & Transforms ---
def build_eval_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def build_phase1_train_transform() -> T.Compose:
    return T.Compose(
        [
            T.RandomResizedCrop(
                224, scale=(0.82, 1.0), ratio=(0.85, 1.18), interpolation=T.InterpolationMode.BICUBIC
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.15, hue=0.03),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
            T.RandomErasing(p=0.15, scale=(0.02, 0.08)),
        ]
    )


def build_phase2_color_train_transform() -> T.Compose:
    """Hue-preserving augmentation (hue=0.0 strictly) for paint color learning."""
    return T.Compose(
        [
            T.RandomResizedCrop(
                224, scale=(0.85, 1.0), ratio=(0.85, 1.18), interpolation=T.InterpolationMode.BICUBIC
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.30, contrast=0.30, saturation=0.12, hue=0.0),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
            T.RandomErasing(p=0.10, scale=(0.02, 0.06)),
        ]
    )


def _crop_with_bbox(rgb: Image.Image, bbox_row: np.ndarray | None) -> Image.Image:
    if bbox_row is None:
        return rgb
    w, h = rgb.size
    x1, y1, x2, y2 = [int(v) for v in bbox_row[:4]]
    x1 = max(0, min(x1, max(w - 1, 0)))
    y1 = max(0, min(y1, max(h - 1, 0)))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return rgb.crop((x1, y1, x2, y2))


class VmmrdbMultiTaskDataset(Dataset):
    def __init__(
        self,
        img_root: Path,
        rows: list[dict[str, Any]],
        bboxes: np.ndarray,
        make_to_id: dict[str, int],
        model_to_id: dict[str, int],
        body_to_id: dict[str, int],
        transform: T.Compose,
    ) -> None:
        self.img_root = img_root
        self.rows = rows
        self.bboxes = bboxes
        self.make_to_id = make_to_id
        self.model_to_id = model_to_id
        self.body_to_id = body_to_id
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        with Image.open(self.img_root / r["img_rel_path"]) as im:
            rgb = im.convert("RGB")
        crop = _crop_with_bbox(rgb, self.bboxes[idx])
        tensor = self.transform(crop)
        return (
            tensor,
            self.make_to_id.get(r["make"], -1),
            self.model_to_id.get(r["canonical_class"], -1),
            self.body_to_id[r["body_type"]],
            r["raw_year"],
            1 if r["year_valid"] else 0,
        )


class VcorColorDataset(Dataset):
    def __init__(
        self,
        samples: list[dict[str, Any]],
        bboxes: np.ndarray,
        color_to_id: dict[str, int],
        transform: T.Compose,
    ) -> None:
        self.samples = samples
        self.bboxes = bboxes
        self.color_to_id = color_to_id
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        with Image.open(s["path"]) as im:
            rgb = im.convert("RGB")
        crop = _crop_with_bbox(rgb, self.bboxes[idx])
        return self.transform(crop), self.color_to_id[s["color"]]


# --- Training & Evaluation Logic ---
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cosine_lr_multiplier(step: int, warmup_steps: int, total_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    progress = min(max(progress, 0.0), 1.0)
    return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))


def _save_phase1_checkpoint(
    path: Path,
    model: AutoVisionMultiTaskNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_score: float,
    history: list[dict[str, Any]],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "history": history,
        },
        path,
    )


@torch.no_grad()
def evaluate_vmmrdb_split(
    model: AutoVisionMultiTaskNet,
    loader: DataLoader,
    num_bodies: int,
    device: str,
    use_tta: bool = True,
    collect_predictions: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    model.eval()
    all_make_logits, all_model_logits, all_body_logits, all_year_logits = [], [], [], []
    all_make_y, all_model_y, all_body_y, all_year_y, all_yvalid = [], [], [], [], []
    use_cuda = device.startswith("cuda")

    for x, mk_y, md_y, bd_y, yr_y, yr_v in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_cuda):
            out = model(x, model_targets=None)
            mk_l = out["make_logits"].float()
            md_l = out["model_logits"].float()
            bd_l = out["body_logits"].float()
            yr_l = out["year_logits"].float()
            if use_tta:
                out_f = model(torch.flip(x, dims=[-1]), model_targets=None)
                mk_l = 0.5 * (mk_l + out_f["make_logits"].float())
                md_l = 0.5 * (md_l + out_f["model_logits"].float())
                bd_l = 0.5 * (bd_l + out_f["body_logits"].float())
                yr_l = 0.5 * (yr_l + out_f["year_logits"].float())

        all_make_logits.append(mk_l.cpu())
        all_model_logits.append(md_l.cpu())
        all_body_logits.append(bd_l.cpu())
        all_year_logits.append(yr_l.cpu())
        all_make_y.append(mk_y)
        all_model_y.append(md_y)
        all_body_y.append(bd_y)
        all_year_y.append(yr_y)
        all_yvalid.append(yr_v)

    mk_logits = torch.cat(all_make_logits, dim=0)
    md_logits = torch.cat(all_model_logits, dim=0)
    bd_logits = torch.cat(all_body_logits, dim=0)
    yr_logits = torch.cat(all_year_logits, dim=0)

    mk_y = torch.cat(all_make_y, dim=0).numpy()
    md_y = torch.cat(all_model_y, dim=0).numpy()
    bd_y = torch.cat(all_body_y, dim=0).numpy()
    yr_y = torch.cat(all_year_y, dim=0).numpy()
    yr_v = torch.cat(all_yvalid, dim=0).numpy().astype(bool)

    mk_top5 = torch.topk(mk_logits, k=min(5, mk_logits.shape[1]), dim=1).indices.numpy()
    md_top5 = torch.topk(md_logits, k=min(5, md_logits.shape[1]), dim=1).indices.numpy()
    bd_top3 = torch.topk(bd_logits, k=min(3, bd_logits.shape[1]), dim=1).indices.numpy()

    mk_pred = mk_top5[:, 0]
    md_pred = md_top5[:, 0]
    bd_pred = bd_top3[:, 0]

    yr_exp, yr_std = year_logits_to_expected(yr_logits)
    yr_exp_np = yr_exp.numpy()
    yr_std_np = yr_std.numpy()
    yr_err = np.abs(yr_exp_np[yr_v] - yr_y[yr_v])

    metrics = {
        "n_eval": int(len(md_y)),
        "make_top1": round(float(np.mean(mk_pred == mk_y)), 4),
        "make_top5": round(float(np.mean([y in r for y, r in zip(mk_y, mk_top5)])), 4),
        "model_top1": round(float(np.mean(md_pred == md_y)), 4),
        "model_top5": round(float(np.mean([y in r for y, r in zip(md_y, md_top5)])), 4),
        "model_macro_f1": round(
            float(f1_score(md_y, md_pred, labels=list(range(md_logits.shape[1])), average="macro", zero_division=0)),
            4,
        ),
        "body_top1": round(float(np.mean(bd_pred == bd_y)), 4),
        "body_top3": round(float(np.mean([y in r for y, r in zip(bd_y, bd_top3)])), 4),
        "body_macro_f1": round(
            float(f1_score(bd_y, bd_pred, labels=list(range(num_bodies)), average="macro", zero_division=0)),
            4,
        ),
        "year_mae": round(float(yr_err.mean()), 4),
        "year_within2": round(float((yr_err <= 2.0).mean()), 4),
    }

    preds_bundle = None
    if collect_predictions:
        preds_bundle = {
            "make_probs": F.softmax(mk_logits, dim=-1).numpy().astype(np.float16),
            "model_logits": md_logits.numpy().astype(np.float16),
            "body_probs": F.softmax(bd_logits, dim=-1).numpy().astype(np.float16),
            "year_expected": yr_exp_np.astype(np.float32),
            "year_std": yr_std_np.astype(np.float32),
            "make_targets": mk_y.astype(np.int64),
            "model_targets": md_y.astype(np.int64),
            "body_targets": bd_y.astype(np.int64),
            "year_targets": yr_y.astype(np.int64),
            "year_valid": yr_v,
        }
    return metrics, preds_bundle


@torch.no_grad()
def extract_all_finetuned_embeddings(
    model: AutoVisionMultiTaskNet,
    img_root: Path,
    rows: list[dict[str, Any]],
    bboxes: np.ndarray,
    used_yolo: np.ndarray,
    make_to_id: dict[str, int],
    model_to_id: dict[str, int],
    body_to_id: dict[str, int],
    out_path: Path,
    device: str,
) -> np.ndarray:
    """Extracts 512-D Flip-TTA L2-normalized float16 embeddings for all 89,535 clean images."""
    model.eval()
    ds = VmmrdbMultiTaskDataset(
        img_root, rows, bboxes, make_to_id, model_to_id, body_to_id, build_eval_transform()
    )
    loader = DataLoader(
        ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.startswith("cuda")
    )
    n = len(rows)
    embs = np.zeros((n, 512), dtype=np.float16)
    pos = 0
    use_cuda = device.startswith("cuda")
    for x, *_ in tqdm(loader, desc="Extract finetuned 512-D embeddings"):
        bsz = x.shape[0]
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_cuda):
            e1 = model(x)["embedding"]
            e2 = model(torch.flip(x, dims=[-1]))["embedding"]
        comb = F.normalize(e1.float() + e2.float(), dim=-1)
        embs[pos : pos + bsz] = comb.cpu().numpy().astype(np.float16)
        pos += bsz

    meta = {
        "backbone": "finetuned_openclip_vitb16",
        "embed_dim": 512,
        "img_size": 224,
        "tta": "horizontal_flip",
        "dtype": "float16",
        "num_samples": n,
    }
    np.savez(
        out_path,
        embeddings=embs,
        bboxes=bboxes.astype(np.float32),
        used_yolo=used_yolo.astype(bool),
        rel_paths=np.asarray([r["rel_path"] for r in rows], dtype=str),
        metadata_json=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )
    print(f"[Export] Saved {out_path.name} {embs.shape}")
    return embs


def compute_prototypes_and_few_shot(
    embs_f16: np.ndarray,
    rows: list[dict[str, Any]],
    out_path: Path,
) -> dict[str, Any]:
    """Builds 342-class normalized prototypes and computes exact 576-query Few-Shot retrieval."""
    embs = embs_f16.astype(np.float32)
    by_split_cls: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    all_meta: dict[str, tuple[str, str]] = {}
    for i, r in enumerate(rows):
        cls_name = r["canonical_class"]
        all_meta.setdefault(cls_name, (r["make"], r["body_type"]))
        by_split_cls[r["split"]][cls_name].append(i)

    cls_list = sorted(all_meta)
    cls_to_id = {c: i for i, c in enumerate(cls_list)}
    protos_342 = np.zeros((len(cls_list), embs.shape[1]), dtype=np.float32)
    counts_342 = np.zeros(len(cls_list), dtype=np.int64)
    split_types: list[str] = []

    gallery_vecs: list[np.ndarray] = []
    gallery_labels: list[int] = []
    query_idxs: list[int] = []
    query_labels: list[int] = []
    eval_few_classes = 0

    for cid, cls_name in enumerate(cls_list):
        if cls_name in by_split_cls["train"]:
            idxs = by_split_cls["train"][cls_name]
            m = embs[idxs].mean(axis=0)
            p = m / max(float(np.linalg.norm(m)), 1e-12)
            protos_342[cid] = p
            counts_342[cid] = len(idxs)
            split_types.append("main_train")
            gallery_vecs.append(p)
            gallery_labels.append(cid)
        else:
            idxs = sorted(by_split_cls["few_shot"][cls_name], key=lambda idx: rows[idx]["rel_path"])
            m = embs[idxs].mean(axis=0)
            protos_342[cid] = m / max(float(np.linalg.norm(m)), 1e-12)
            counts_342[cid] = len(idxs)
            split_types.append("few_shot")

            support_idx = idxs[0]
            gallery_vecs.append(embs[support_idx])
            gallery_labels.append(cid)
            if len(idxs) >= 2:
                eval_few_classes += 1
                for q_idx in idxs[1:]:
                    query_idxs.append(q_idx)
                    query_labels.append(cid)

    np.savez(
        out_path,
        prototypes=protos_342.astype(np.float16),
        class_names=np.asarray(cls_list, dtype=str),
        makes=np.asarray([all_meta[c][0] for c in cls_list], dtype=str),
        body_types=np.asarray([all_meta[c][1] for c in cls_list], dtype=str),
        counts=counts_342,
        split_types=np.asarray(split_types, dtype=str),
    )

    G = np.vstack(gallery_vecs).astype(np.float32)
    g_lbls = np.array(gallery_labels, dtype=np.int64)
    Q = embs[query_idxs].astype(np.float32)
    y_q = np.array(query_labels, dtype=np.int64)
    sims = Q @ G.T
    top5 = g_lbls[np.argsort(-sims, axis=1)[:, : min(5, len(G))]]
    few_metrics = {
        "few_shot_classes_eval": int(eval_few_classes),
        "few_shot_queries": int(len(query_idxs)),
        "few_shot_gallery_classes": int(len(G)),
        "few_shot_top1": round(float(np.mean(top5[:, 0] == y_q)), 4),
        "few_shot_top5": round(float(np.mean([y in row for y, row in zip(y_q, top5)])), 4),
    }
    print(f"[Few-Shot] Queries={few_metrics['few_shot_queries']} | Top-1={few_metrics['few_shot_top1']*100:.2f}% | Top-5={few_metrics['few_shot_top5']*100:.2f}%")
    return few_metrics


def _score_color_predictions(
    logits: np.ndarray,
    targets: np.ndarray,
    color_classes: Sequence[str],
) -> dict[str, Any]:
    n_cls = len(color_classes)
    top3 = np.argsort(-logits, axis=1)[:, : min(3, n_cls)]
    preds = top3[:, 0]
    g_true = [COLOR_GROUPS_11[color_classes[int(y)]] for y in targets]
    g_pred = [COLOR_GROUPS_11[color_classes[int(p)]] for p in preds]
    return {
        "n_eval": int(len(targets)),
        "top1": round(float(np.mean(preds == targets)), 4),
        "top3": round(float(np.mean([y in row for y, row in zip(targets, top3)])), 4),
        "macro_f1": round(
            float(f1_score(targets, preds, labels=list(range(n_cls)), average="macro", zero_division=0)),
            4,
        ),
        "group11_top1": round(float(np.mean([a == b for a, b in zip(g_true, g_pred)])), 4),
    }


def eval_vcor_frozen_linear_baseline(
    model: AutoVisionMultiTaskNet,
    tr_loader: DataLoader,
    va_loader: DataLoader,
    te_loader: DataLoader,
    color_classes: Sequence[str],
    device: str,
) -> dict[str, Any]:
    """Ablation: trains a 20-epoch GPU Linear Probe on frozen Phase-1 z_id (512-D) embeddings."""
    model.eval()
    use_cuda = device.startswith("cuda")

    @torch.no_grad()
    def _extract(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                emb = model(x)["embedding"].float()
            xs.append(emb)
            ys.append(y.to(device, non_blocking=True))
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

    X_tr, y_tr = _extract(tr_loader)
    X_va, y_va = _extract(va_loader)
    X_te, y_te = _extract(te_loader)

    probe = nn.Linear(512, len(color_classes)).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=3e-3, weight_decay=1e-4)
    best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
    best_val_acc = -1.0

    for _ in range(20):
        probe.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(X_tr), y_tr)
        loss.backward()
        opt.step()

        probe.eval()
        with torch.no_grad():
            val_acc = float((probe(X_va).argmax(dim=1) == y_va).float().mean().item())
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

    probe.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    probe.eval()
    with torch.no_grad():
        te_logits = probe(X_te).cpu().numpy()
    metrics = _score_color_predictions(te_logits, y_te.cpu().numpy(), color_classes)
    print(
        f"[VCoR Baseline z_id LP] Top-1={metrics['top1']*100:.2f}% | "
        f"Macro-F1={metrics['macro_f1']*100:.2f}% | Group11={metrics['group11_top1']*100:.2f}%"
    )
    return metrics


@torch.no_grad()
def evaluate_vcor_split(
    model: AutoVisionMultiTaskNet,
    loader: DataLoader,
    color_classes: Sequence[str],
    device: str,
    use_tta: bool = True,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    use_cuda = device.startswith("cuda")
    logits_list, targets_list = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_cuda):
            logits = model(x)["color_logits"].float()
            if use_tta:
                logits = 0.5 * (logits + model(torch.flip(x, dims=[-1]))["color_logits"].float())
        logits_list.append(logits.cpu().numpy())
        targets_list.append(y.numpy())

    all_logits = np.concatenate(logits_list, axis=0)
    all_targets = np.concatenate(targets_list, axis=0)
    return _score_color_predictions(all_logits, all_targets, color_classes), all_logits, all_targets


def train_phase2_color(
    model: AutoVisionMultiTaskNet,
    vcor_root: Path,
    out_dir: Path,
    device: str,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Trains SpatialAttentionColorHead on VCoR with frozen Phase-1 ViT & 4 VMMRdb heads."""
    tr_samples = scan_vcor_split(vcor_root / "train")
    val_dir = vcor_root / "val" if (vcor_root / "val").is_dir() else vcor_root / "valid"
    va_samples = scan_vcor_split(val_dir)
    te_samples = scan_vcor_split(vcor_root / "test")

    all_samples = tr_samples + va_samples + te_samples
    all_bboxes, all_used = run_vcor_yolo_stage(all_samples, out_dir, device)
    n_tr, n_va = len(tr_samples), len(va_samples)
    tr_boxes = all_bboxes[:n_tr]
    va_boxes = all_bboxes[n_tr : n_tr + n_va]
    te_boxes = all_bboxes[n_tr + n_va :]

    color_to_id = {c: i for i, c in enumerate(COLOR_CLASSES)}
    eval_tf = build_eval_transform()
    tr_eval_ds = VcorColorDataset(tr_samples, tr_boxes, color_to_id, eval_tf)
    tr_train_ds = VcorColorDataset(tr_samples, tr_boxes, color_to_id, build_phase2_color_train_transform())
    va_ds = VcorColorDataset(va_samples, va_boxes, color_to_id, eval_tf)
    te_ds = VcorColorDataset(te_samples, te_boxes, color_to_id, eval_tf)

    pin = device.startswith("cuda")
    tr_eval_loader = DataLoader(tr_eval_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)
    tr_loader = DataLoader(tr_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)
    te_loader = DataLoader(te_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)

    # Baseline ablation on frozen Phase-1 z_id (512-D)
    frozen_lp_metrics = eval_vcor_frozen_linear_baseline(
        model, tr_eval_loader, va_loader, te_loader, COLOR_CLASSES, device
    )

    # Train SpatialAttentionColorHead
    cnt = Counter(color_to_id[s["color"]] for s in tr_samples)
    color_weights = compute_soft_class_weights(
        [cnt[i] for i in range(len(COLOR_CLASSES))], power=0.35, min_weight=0.5, max_weight=3.0
    ).to(device)

    model.set_phase2_color_mode()
    optimizer = torch.optim.AdamW(model.color_head.parameters(), lr=COLOR_LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_COLOR_EPOCHS, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=pin)

    best_score = -1.0
    best_color_path = out_dir / "phase2_color_best.pt"
    t0 = time.perf_counter()

    for epoch in range(1, PHASE2_COLOR_EPOCHS + 1):
        model.train(True)
        run_loss, steps = 0.0, 0
        for x, y in tqdm(tr_loader, desc=f"Phase 2 Color Ep {epoch}/{PHASE2_COLOR_EPOCHS}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=pin):
                logits = model(x)["color_logits"]
                loss = F.cross_entropy(logits.float(), y.long(), weight=color_weights, label_smoothing=0.05)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.color_head.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            run_loss += float(loss.item())
            steps += 1
        scheduler.step()

        va_m, _, _ = evaluate_vcor_split(model, va_loader, COLOR_CLASSES, device, use_tta=True)
        score = 0.5 * va_m["top1"] + 0.5 * va_m["macro_f1"]
        print(
            f"  [Phase 2 Ep {epoch:02d}] loss={run_loss/max(steps,1):.4f} | "
            f"val_top1={va_m['top1']*100:.2f}% | val_macro_f1={va_m['macro_f1']*100:.2f}% | "
            f"group11={va_m['group11_top1']*100:.2f}%"
        )
        history.append(
            {
                "phase": "phase2_color",
                "epoch": epoch,
                "train_loss": round(run_loss / max(steps, 1), 4),
                "color_val_top1": va_m["top1"],
                "color_val_macro_f1": va_m["macro_f1"],
                "color_val_group11": va_m["group11_top1"],
            }
        )
        if score > best_score:
            best_score = score
            torch.save({"color_head": model.color_head.state_dict(), "val_metrics": va_m}, best_color_path)

    if best_color_path.is_file():
        ckpt = torch.load(best_color_path, map_location=device, weights_only=True)
        model.color_head.load_state_dict(ckpt["color_head"])

    te_m, te_logits, te_targets = evaluate_vcor_split(model, te_loader, COLOR_CLASSES, device, use_tta=True)
    te_probs = F.softmax(torch.from_numpy(te_logits), dim=-1).numpy().astype(np.float16)
    np.savez(
        out_dir / "vcor_test_predictions.npz",
        probs=te_probs,
        preds=te_logits.argmax(axis=1).astype(np.int64),
        targets=te_targets.astype(np.int64),
        group11_top1=np.asarray(te_m["group11_top1"], dtype=np.float32),
        rel_paths=np.asarray([s["rel_path"] for s in te_samples], dtype=str),
    )
    print(
        f"[VCoR Final SpatialAttn Test] Top-1={te_m['top1']*100:.2f}% | "
        f"Top-3={te_m['top3']*100:.2f}% | Macro-F1={te_m['macro_f1']*100:.2f}% | "
        f"Group11={te_m['group11_top1']*100:.2f}%"
    )
    return {
        "vcor_root": str(vcor_root),
        "counts": {"train": len(tr_samples), "val": len(va_samples), "test": len(te_samples)},
        "yolo_hit_rate": round(float(all_used.mean()), 4),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "frozen_zid_linear_probe": frozen_lp_metrics,
        "spatial_attention_test": te_m,
    }


def export_onnx_model(model: AutoVisionMultiTaskNet, out_path: Path) -> bool:
    try:
        import onnx  # noqa: F401
    except ImportError:
        print("[ONNX] 'onnx' package not installed on Kaggle; skipping ONNX export (checkpoint .pt saved safely).")
        return False

    try:
        model_cpu = AutoVisionMultiTaskNet(
            num_makes=model.make_head[-1].out_features,
            num_models=model.model_head.out_features,
            num_bodies=model.body_head[-1].out_features,
            num_years=model.year_head[-1].out_features,
            num_colors=model.color_head.classifier[-1].out_features,
            pretrained=None,
        ).cpu().eval()
        model_cpu.load_state_dict(model.state_dict(), strict=True)
        wrapper = AutoVisionOnnxWrapper(model_cpu).eval()
        dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        torch.onnx.export(
            wrapper,
            dummy,
            str(out_path),
            input_names=["image"],
            output_names=[
                "embedding",
                "make_logits",
                "model_logits",
                "body_logits",
                "year_logits",
                "color_logits",
            ],
            dynamic_axes={
                "image": {0: "batch"},
                "embedding": {0: "batch"},
                "make_logits": {0: "batch"},
                "model_logits": {0: "batch"},
                "body_logits": {0: "batch"},
                "year_logits": {0: "batch"},
                "color_logits": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"[ONNX] Exported cleanly to {out_path} ({out_path.stat().st_size / (1024**2):.1f} MB)")
        return True
    except Exception as exc:
        print(f"[ONNX] Export skipped due to error (checkpoint .pt preserved): {exc}")
        return False


def main() -> None:
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = device.startswith("cuda")
    print(f"=== AutoVision Multi-Task Training (device={device.upper()}) ===")

    manifest_path, img_root, rows = find_vmmrdb_manifest_and_root(Path("/kaggle/input"))
    bboxes, used_yolo = find_or_compute_vmmrdb_bboxes(rows, img_root, OUT_DIR, device)
    print(f"Manifest: {manifest_path} | Image root: {img_root} | Clean rows: {len(rows):,}")

    tr_idx = [i for i, r in enumerate(rows) if r["split"] == "train"]
    va_idx = [i for i, r in enumerate(rows) if r["split"] == "val"]
    te_idx = [i for i, r in enumerate(rows) if r["split"] == "test"]

    make_classes = sorted({rows[i]["make"] for i in tr_idx})
    model_classes = sorted({rows[i]["canonical_class"] for i in tr_idx})
    body_classes = list(BODY_TYPE_CLASSES)
    make_to_id = {c: i for i, c in enumerate(make_classes)}
    model_to_id = {c: i for i, c in enumerate(model_classes)}
    body_to_id = {c: i for i, c in enumerate(body_classes)}

    make_cnt = Counter(make_to_id[rows[i]["make"]] for i in tr_idx)
    body_cnt = Counter(body_to_id[rows[i]["body_type"]] for i in tr_idx)
    make_weights = compute_soft_class_weights([make_cnt[i] for i in range(len(make_classes))], power=0.35)
    body_weights = compute_soft_class_weights([body_cnt[i] for i in range(len(body_classes))], power=0.50)

    tr_rows = [rows[i] for i in tr_idx]
    va_rows = [rows[i] for i in va_idx]
    te_rows = [rows[i] for i in te_idx]
    tr_boxes, va_boxes, te_boxes = bboxes[tr_idx], bboxes[va_idx], bboxes[te_idx]

    eval_tf = build_eval_transform()
    tr_ds = VmmrdbMultiTaskDataset(img_root, tr_rows, tr_boxes, make_to_id, model_to_id, body_to_id, build_phase1_train_transform())
    va_ds = VmmrdbMultiTaskDataset(img_root, va_rows, va_boxes, make_to_id, model_to_id, body_to_id, eval_tf)
    te_ds = VmmrdbMultiTaskDataset(img_root, te_rows, te_boxes, make_to_id, model_to_id, body_to_id, eval_tf)

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=use_cuda, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=use_cuda)
    te_loader = DataLoader(te_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=use_cuda)

    model = AutoVisionMultiTaskNet(
        num_makes=len(make_classes),
        num_models=len(model_classes),
        num_bodies=len(body_classes),
        num_years=NUM_YEAR_BINS,
        num_colors=len(COLOR_CLASSES),
        pretrained="laion2b_s34b_b88k",
    ).to(device)
    criterion = MultiTaskVmmrdbLoss(make_weights=make_weights, body_weights=body_weights).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    history: list[dict[str, Any]] = []
    phase1_latest = OUT_DIR / "phase1_latest.pt"
    phase1_best = OUT_DIR / "phase1_best.pt"
    t_phase1 = time.perf_counter()

    # Phase 1A: LP-FT Warmup (only if starting fresh)
    if not phase1_latest.is_file():
        print(f"\n[Phase 1A] LP-FT Head Warmup ({PHASE1_WARMUP_EPOCHS} epoch, frozen ViT)...")
        model.set_phase1_warmup_mode()
        warm_opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=WARMUP_HEAD_LR, weight_decay=WEIGHT_DECAY
        )
        for ep in range(1, PHASE1_WARMUP_EPOCHS + 1):
            model.train(True)
            loss_sum, steps = 0.0, 0
            for x, mk_y, md_y, bd_y, yr_y, yr_v in tqdm(tr_loader, desc=f"Warmup Ep {ep}"):
                x = x.to(device, non_blocking=True)
                mk_y, md_y = mk_y.to(device, non_blocking=True), md_y.to(device, non_blocking=True)
                bd_y = bd_y.to(device, non_blocking=True)
                yr_y, yr_v = yr_y.to(device, non_blocking=True), yr_v.to(device, non_blocking=True)
                warm_opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_cuda):
                    out = model(x, model_targets=md_y)
                    loss, _ = criterion(out, mk_y, md_y, bd_y, yr_y, yr_v)
                scaler.scale(loss).backward()
                scaler.unscale_(warm_opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(warm_opt)
                scaler.update()
                loss_sum += float(loss.item())
                steps += 1
            va_m, _ = evaluate_vmmrdb_split(model, va_loader, len(body_classes), device, use_tta=True)
            print(
                f"  [Warmup Ep {ep}] loss={loss_sum/max(steps,1):.4f} | "
                f"val model={va_m['model_top1']*100:.2f}% | make={va_m['make_top1']*100:.2f}% | "
                f"bodyF1={va_m['body_macro_f1']*100:.2f}% | yrMAE={va_m['year_mae']:.2f}"
            )
            history.append({"phase": "phase1_warmup", "epoch": ep, "train_loss": round(loss_sum / max(steps, 1), 4), **va_m})

    # Phase 1B: Differential Fine-Tuning
    print(f"\n[Phase 1B] Differential Fine-Tuning ({PHASE1_EPOCHS} epochs)...")
    model.set_phase1_finetune_mode()
    param_groups = model.build_phase1_param_groups(BACKBONE_LR, PROJ_LR, HEAD_LR, WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(param_groups)
    total_steps = PHASE1_EPOCHS * max(len(tr_loader), 1)
    warmup_steps = int(0.5 * max(len(tr_loader), 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda st: _cosine_lr_multiplier(st, warmup_steps, total_steps)
    )

    start_epoch = 1
    best_val_score = -1.0
    if phase1_latest.is_file():
        ckpt = torch.load(phase1_latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_score = float(ckpt["best_score"])
        history = list(ckpt.get("history", []))
        print(f"  [Resume Phase 1B] Resuming from epoch {start_epoch} (best_score={best_val_score:.4f})")

    for epoch in range(start_epoch, PHASE1_EPOCHS + 1):
        model.train(True)
        loss_sum, steps = 0.0, 0
        for x, mk_y, md_y, bd_y, yr_y, yr_v in tqdm(tr_loader, desc=f"Phase 1B Ep {epoch}/{PHASE1_EPOCHS}"):
            x = x.to(device, non_blocking=True)
            mk_y, md_y = mk_y.to(device, non_blocking=True), md_y.to(device, non_blocking=True)
            bd_y = bd_y.to(device, non_blocking=True)
            yr_y, yr_v = yr_y.to(device, non_blocking=True), yr_v.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                out = model(x, model_targets=md_y)
                loss, _ = criterion(out, mk_y, md_y, bd_y, yr_y, yr_v)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss.item())
            steps += 1

        va_m, _ = evaluate_vmmrdb_split(model, va_loader, len(body_classes), device, use_tta=True)
        val_score = (
            0.50 * va_m["model_top1"]
            + 0.25 * va_m["body_macro_f1"]
            + 0.15 * va_m["make_top1"]
            + 0.10 * va_m["year_within2"]
        )
        print(
            f"  [Phase 1B Ep {epoch:02d}] loss={loss_sum/max(steps,1):.4f} | score={val_score:.4f} | "
            f"model={va_m['model_top1']*100:.2f}% | make={va_m['make_top1']*100:.2f}% | "
            f"body={va_m['body_top1']*100:.2f}% (F1={va_m['body_macro_f1']*100:.2f}%) | "
            f"yrMAE={va_m['year_mae']:.2f} (±2g={va_m['year_within2']*100:.1f}%)"
        )
        history.append(
            {"phase": "phase1_finetune", "epoch": epoch, "train_loss": round(loss_sum / max(steps, 1), 4), "val_score": round(val_score, 4), **va_m}
        )
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save({"model_state_dict": model.state_dict(), "val_metrics": va_m, "val_score": val_score}, phase1_best)
        _save_phase1_checkpoint(phase1_latest, model, optimizer, scheduler, scaler, epoch, best_val_score, history)

    if phase1_best.is_file():
        best_ckpt = torch.load(phase1_best, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
    phase1_sec = time.perf_counter() - t_phase1

    # Phase 2: VCoR Color Learning (strictly frozen Phase 1 parameters)
    vcor_root = find_vcor_root(Path("/kaggle/input"))
    vcor_summary: dict[str, Any] = {"found": vcor_root is not None}
    if vcor_root is not None:
        print(f"\n[Phase 2] Found VCoR at {vcor_root}. Running frozen ablation + Color Head training...")
        vcor_summary.update(train_phase2_color(model, vcor_root, OUT_DIR, device, history))
    else:
        print("\n[Phase 2] VCoR dataset not attached under /kaggle/input; skipping color training.")

    # Final Evaluation on VMMRdb Test (verifies zero regression after Phase 2)
    print("\n[Eval] Evaluating final model on VMMRdb test=8,885 with Flip TTA...")
    test_metrics, test_preds = evaluate_vmmrdb_split(
        model, te_loader, len(body_classes), device, use_tta=True, collect_predictions=True
    )
    assert test_preds is not None
    np.savez(
        OUT_DIR / "vmmrdb_test_predictions.npz",
        **test_preds,
        rel_paths=np.asarray([r["rel_path"] for r in te_rows], dtype=str),
    )

    # Full 89,535 Embedding Extraction + 342-Class Prototypes + Exact Few-Shot Benchmark
    embs_f16 = extract_all_finetuned_embeddings(
        model, img_root, rows, bboxes, used_yolo, make_to_id, model_to_id, body_to_id,
        OUT_DIR / "features_finetuned_openclip_vitb16.npz", device
    )
    few_shot_metrics = compute_prototypes_and_few_shot(embs_f16, rows, OUT_DIR / "prototypes_342.npz")

    label_maps = {
        "make_classes": make_classes,
        "model_classes": model_classes,
        "body_classes": body_classes,
        "color_classes": list(COLOR_CLASSES),
        "color_groups_11": COLOR_GROUPS_11,
        "color_hex": COLOR_HEX,
        "year_min": YEAR_MIN,
        "year_max": YEAR_MAX,
    }
    (OUT_DIR / "label_maps.json").write_text(json.dumps(label_maps, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save training history CSV
    if history:
        fieldnames = sorted({k for row in history for k in row.keys()})
        with open(OUT_DIR / "training_history.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)

    # Clean final checkpoint + ONNX export
    onnx_exported = export_onnx_model(model, OUT_DIR / "autovision_multitask.onnx")
    summary = {
        "num_clean_images": len(rows),
        "num_makes_train": len(make_classes),
        "num_models_train": len(model_classes),
        "num_bodies": len(body_classes),
        "num_colors": len(COLOR_CLASSES),
        "phase1_elapsed_sec": round(phase1_sec, 2),
        "best_phase1_val_score": round(best_val_score, 4),
        "vmmrdb_test_metrics": test_metrics,
        "few_shot_metrics": few_shot_metrics,
        "vcor_summary": vcor_summary,
        "onnx_exported": onnx_exported,
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_maps": label_maps,
            "summary": summary,
        },
        OUT_DIR / "autovision_multitask_best.pt",
    )
    (OUT_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== TRAINING & EXPORT COMPLETE -> {OUT_DIR} ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
