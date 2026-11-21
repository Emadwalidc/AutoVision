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
import torchvision.transforms.functional as TF
from PIL import Image

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from color_expert import (  # noqa: E402
    AutoVisionColorExpert,
    COLOR_CLASSES_V2,
    COLOR_GROUPS_V2,
    COLOR_HEX_V2,
    IMAGENET_MEAN,
    IMAGENET_STD,
    apply_neutral_gray_mask,
    extract_masked_lab_features,
)
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
    model_cosines: np.ndarray | None = None


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


class PyTorchColorExpert:
    """Standalone ConvNeXt-Tiny + 16-D Masked CIE-Lab color classifier."""

    def __init__(self, color_models_dir: Path, device: str = "cpu") -> None:
        lmap_path = color_models_dir / "color_label_maps_v2.json"
        ckpt_path = color_models_dir / "autovision_color_expert_v2.pt"
        if not lmap_path.is_file():
            raise FileNotFoundError(f"Missing color_label_maps_v2.json: {lmap_path}")
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing color expert checkpoint: {ckpt_path}")

        clm = json.loads(lmap_path.read_text(encoding="utf-8"))
        self.color_classes: list[str] = list(clm.get("color_classes") or COLOR_CLASSES_V2)
        self.color_hex: dict[str, str] = dict(clm.get("color_hex") or COLOR_HEX_V2)
        self.color_groups: dict[str, str] = {
            c: COLOR_GROUPS_V2.get(c, c) for c in self.color_classes
        }
        self.imagenet_mean = tuple(float(v) for v in clm.get("imagenet_mean", IMAGENET_MEAN))
        self.imagenet_std = tuple(float(v) for v in clm.get("imagenet_std", IMAGENET_STD))
        bg = clm.get("neutral_bg_rgb", (128, 128, 128))
        self.neutral_bg_rgb = (int(bg[0]), int(bg[1]), int(bg[2]))
        self._device = device

        self._net = AutoVisionColorExpert(
            num_colors=len(self.color_classes),
            lab_dim=16,
            pretrained=False,
        ).to(self._device)
        ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=False)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self._net.load_state_dict(state, strict=True)
        self._net.eval()

    def classify_crops(
        self,
        crops: list[Image.Image],
        masks: list[np.ndarray | None] | None = None,
        use_tta: bool = False,
    ) -> np.ndarray:
        x_list: list[torch.Tensor] = []
        lab_list: list[torch.Tensor] = []

        for idx, crop in enumerate(crops):
            rgb_crop = crop.convert("RGB")
            masked_rgb: Image.Image | None = None
            rgb_224: Image.Image | None = None
            mask_pil: Image.Image | None = None
            mask_224_pil: Image.Image | None = None
            try:
                mask = masks[idx] if (masks is not None and idx < len(masks)) else None
                if mask is None:
                    h, w = rgb_crop.height, rgb_crop.width
                    mask = np.zeros((h, w), dtype=bool)
                    mask[int(0.12 * h) : int(0.88 * h), int(0.12 * w) : int(0.88 * w)] = True

                masked_rgb, final_mask = apply_neutral_gray_mask(
                    rgb_crop,
                    mask,
                    erode_px=2,
                    bg_rgb=self.neutral_bg_rgb,
                )
                rgb_224 = TF.resize(
                    masked_rgb, [224, 224], interpolation=TF.InterpolationMode.BICUBIC
                )
                mask_pil = Image.fromarray((final_mask.astype(np.uint8) * 255), mode="L")
                mask_224_pil = TF.resize(
                    mask_pil, [224, 224], interpolation=TF.InterpolationMode.NEAREST
                )
                mask_224 = np.asarray(mask_224_pil) >= 128
                if int(mask_224.sum()) < 32:
                    mask_224 = np.ones((224, 224), dtype=bool)

                lab_vec = torch.from_numpy(
                    extract_masked_lab_features(np.asarray(rgb_224, dtype=np.uint8), mask_224)
                )
                x_t = TF.normalize(TF.to_tensor(rgb_224), self.imagenet_mean, self.imagenet_std)
                x_list.append(x_t)
                lab_list.append(lab_vec)
            finally:
                for tmp in (mask_224_pil, mask_pil, rgb_224, masked_rgb):
                    if tmp is not None:
                        tmp.close()
                if rgb_crop is not crop:
                    rgb_crop.close()

        x = torch.stack(x_list).to(self._device)
        lab = torch.stack(lab_list).to(self._device)
        use_cuda = self._device.startswith("cuda")

        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=use_cuda):
                logits = self._net(x, lab).float()
                if use_tta:
                    logits = 0.5 * (logits + self._net(torch.flip(x, dims=[-1]), lab).float())
            return F.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)


class PyTorchAutoVisionClassifier:
    """Loads fine-tuned AutoVisionMultiTaskNet (+ optional Phase 2 Color Expert) for inference."""

    def __init__(
        self,
        models_dir: Path,
        device: str = "auto",
        color_models_dir: Path | None = None,
        model_temperature: float = 2.15,
    ) -> None:
        self._model_temperature = max(float(model_temperature), 1e-3)
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

        self._color_expert: PyTorchColorExpert | None = None
        if (
            color_models_dir is not None
            and (color_models_dir / "autovision_color_expert_v2.pt").is_file()
            and (color_models_dir / "color_label_maps_v2.json").is_file()
        ):
            self._color_expert = PyTorchColorExpert(color_models_dir, device=self._device)
            self._label_maps["color_classes"] = list(self._color_expert.color_classes)
            self._label_maps["color_groups_11"] = dict(self._color_expert.color_groups)
            self._label_maps["color_hex"] = dict(self._color_expert.color_hex)

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

    @property
    def supports_masks(self) -> bool:
        return True

    def classify_crops(
        self,
        crops: list[Image.Image],
        use_tta: bool = False,
        masks: list[np.ndarray | None] | None = None,
    ) -> ClassifierBatchOutput:
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
            cos_scale = float(getattr(self._net.model_head, "scale", 30.0))
            model_cosines = (md_l / max(cos_scale, 1e-6)).cpu().numpy().astype(np.float32)
            color_probs = (
                self._color_expert.classify_crops(crops, masks=masks, use_tta=use_tta)
                if self._color_expert is not None
                else F.softmax(cl_l, dim=-1).cpu().numpy().astype(np.float32)
            )
            return ClassifierBatchOutput(
                embeddings=emb.cpu().numpy().astype(np.float32),
                make_probs=F.softmax(mk_l, dim=-1).cpu().numpy().astype(np.float32),
                model_probs=F.softmax(md_l / self._model_temperature, dim=-1).cpu().numpy().astype(np.float32),
                body_probs=F.softmax(bd_l, dim=-1).cpu().numpy().astype(np.float32),
                year_expected=yr_exp.cpu().numpy().astype(np.float32),
                year_std=yr_std.cpu().numpy().astype(np.float32),
                color_probs=color_probs,
                model_cosines=model_cosines,
            )
