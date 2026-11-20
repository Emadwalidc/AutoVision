"""Unified multi-task vehicle attribute model, taxonomies, and losses for AutoVision."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

YEAR_MIN: int = 1950
YEAR_MAX: int = 2016
NUM_YEAR_BINS: int = YEAR_MAX - YEAR_MIN + 1

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
        conv_feat = self.conv_stem(x_rgb)  # [B, 128, 14, 14]
        bsz, ch, gh, gw = conv_feat.shape
        conv_tokens = conv_feat.flatten(2).transpose(1, 2)  # [B, 196, 128]

        attn_logits = self.attn_gate(tok_late.float()).squeeze(-1)  # [B, 196]
        attn = F.softmax(attn_logits, dim=-1)
        attn_unsq = attn.unsqueeze(-1)

        conv_pooled = (conv_tokens * attn_unsq).sum(dim=1)
        mid_pooled = (tok_mid.float() * attn_unsq).sum(dim=1)
        late_pooled = tok_late.float().mean(dim=1)

        rgb_14 = F.avg_pool2d(x_rgb, kernel_size=16, stride=16)  # [B, 3, 14, 14]
        rgb_patches = rgb_14.flatten(2).transpose(1, 2)  # [B, 196, 3]
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
        """Builds differential LR and weight-decay parameter groups for Phase 1 training."""
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
    """Tuple-output wrapper for clean ONNX export of all 5 attribute heads and 512-D embedding."""

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
