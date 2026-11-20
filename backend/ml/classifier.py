"""Multi-task vehicle attribute classifier abstraction and PyTorch implementation."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from multitask_model import (  # noqa: E402
    CLIP_MEAN,
    CLIP_STD,
    COLOR_CLASSES,
    COLOR_GROUPS_11,
    COLOR_HEX,
    YEAR_MAX,
    YEAR_MIN,
    AutoVisionMultiTaskNet,
    year_logits_to_expected,
)


@dataclass
class ClassifierBatchOutput:
    """Batched outputs across 512-D embedding and all 5 attribute heads."""

    embeddings: np.ndarray
    make_probs: np.ndarray
    model_probs: np.ndarray
    body_probs: np.ndarray
    year_expected: np.ndarray
    year_std: np.ndarray
    color_probs: np.ndarray


class AttributeClassifier(Protocol):
    """Protocol for swappable multi-task attribute classifiers (PyTorch / ONNX)."""

    @property
    def label_maps(self) -> dict[str, Any]:
        ...

    @property
    def device(self) -> str:
        ...

    def classify_crops(self, crops: list[Image.Image], use_tta: bool = False) -> ClassifierBatchOutput:
        ...


def resolve_device(requested: str = "auto") -> str:
    if requested.lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


class PyTorchAutoVisionClassifier:
    """Loads fine-tuned AutoVisionMultiTaskNet weights cleanly for CPU/CUDA inference."""

    def __init__(self, models_dir: Path, device: str = "auto") -> None:
        lmap_path = models_dir / "label_maps.json"
        ckpt_path = models_dir / "autovision_multitask_best.pt"
        if not lmap_path.is_file():
            raise FileNotFoundError(f"Missing label_maps.json: {lmap_path}")
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing model checkpoint: {ckpt_path}")

        self._label_maps: dict[str, Any] = json.loads(lmap_path.read_text(encoding="utf-8"))
        self._label_maps.setdefault("color_classes", list(COLOR_CLASSES))
        self._label_maps.setdefault("color_groups_11", COLOR_GROUPS_11)
        self._label_maps.setdefault("color_hex", COLOR_HEX)
        self._label_maps.setdefault("year_min", YEAR_MIN)
        self._label_maps.setdefault("year_max", YEAR_MAX)

        self._device = resolve_device(device)
        year_min = int(self._label_maps["year_min"])
        year_max = int(self._label_maps["year_max"])
        self._net = AutoVisionMultiTaskNet(
            num_makes=len(self._label_maps["make_classes"]),
            num_models=len(self._label_maps["model_classes"]),
            num_bodies=len(self._label_maps["body_classes"]),
            num_years=year_max - year_min + 1,
            num_colors=len(self._label_maps["color_classes"]),
            pretrained=None,
        ).to(self._device)

        ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=False)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self._net.load_state_dict(state, strict=True)
        self._net.eval()

        self._tf = T.Compose(
            [
                T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(CLIP_MEAN, CLIP_STD),
            ]
        )

    @property
    def label_maps(self) -> dict[str, Any]:
        return self._label_maps

    @property
    def device(self) -> str:
        return self._device

    def classify_crops(self, crops: list[Image.Image], use_tta: bool = False) -> ClassifierBatchOutput:
        if not crops:
            raise ValueError("classify_crops requires at least one vehicle crop.")

        x = torch.stack([self._tf(c) for c in crops]).to(self._device)
        year_min = int(self._label_maps["year_min"])
        year_max = int(self._label_maps["year_max"])
        use_cuda = self._device.startswith("cuda")

        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=use_cuda):
                out = self._net(x, model_targets=None)
                emb = out["embedding"].float()
                mk_l = out["make_logits"].float()
                md_l = out["model_logits"].float()
                bd_l = out["body_logits"].float()
                yr_l = out["year_logits"].float()
                cl_l = out["color_logits"].float()

                if use_tta:
                    out_f = self._net(torch.flip(x, dims=[-1]), model_targets=None)
                    emb = F.normalize(emb + out_f["embedding"].float(), dim=-1)
                    mk_l = 0.5 * (mk_l + out_f["make_logits"].float())
                    md_l = 0.5 * (md_l + out_f["model_logits"].float())
                    bd_l = 0.5 * (bd_l + out_f["body_logits"].float())
                    yr_l = 0.5 * (yr_l + out_f["year_logits"].float())
                    cl_l = 0.5 * (cl_l + out_f["color_logits"].float())

            yr_exp, yr_std = year_logits_to_expected(yr_l, year_min=year_min, year_max=year_max)
            return ClassifierBatchOutput(
                embeddings=emb.cpu().numpy().astype(np.float32),
                make_probs=F.softmax(mk_l, dim=-1).cpu().numpy().astype(np.float32),
                model_probs=F.softmax(md_l, dim=-1).cpu().numpy().astype(np.float32),
                body_probs=F.softmax(bd_l, dim=-1).cpu().numpy().astype(np.float32),
                year_expected=yr_exp.cpu().numpy().astype(np.float32),
                year_std=yr_std.cpu().numpy().astype(np.float32),
                color_probs=F.softmax(cl_l, dim=-1).cpu().numpy().astype(np.float32),
            )
