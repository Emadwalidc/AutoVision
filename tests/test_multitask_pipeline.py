"""Automated unit and parity tests for AutoVision Multi-Task model and Kaggle training script."""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
KAGGLE_DIR = BASE_DIR / "kaggle"
CACHE_ULTRALYTICS = BASE_DIR / ".cache" / "ultralytics"
CACHE_ULTRALYTICS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_ULTRALYTICS))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import build_attributes_manifest as bam  # noqa: E402
from multitask_model import (  # noqa: E402
    COLOR_CLASSES,
    COLOR_GROUPS_11,
    COLOR_HEX,
    NUM_YEAR_BINS,
    YEAR_MAX,
    YEAR_MIN,
    AutoVisionMultiTaskNet,
    AutoVisionOnnxWrapper,
    CosFaceHead,
    MultiTaskVmmrdbLoss,
    compute_soft_class_weights,
    dldl_year_loss,
    normalize_color_label,
    year_logits_to_expected,
    year_to_gaussian_target,
)


def _load_kaggle_module():
    script_path = KAGGLE_DIR / "train_multitask_autovision.py"
    spec = importlib.util.spec_from_file_location("train_multitask_autovision", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestTaxonomiesAndLosses(unittest.TestCase):
    def test_color_taxonomy_and_normalization(self) -> None:
        self.assertEqual(len(COLOR_CLASSES), 15)
        self.assertEqual(normalize_color_label("grey"), "Gray")
        self.assertEqual(normalize_color_label("GRAY"), "Gray")
        self.assertEqual(normalize_color_label("violet"), "Purple")
        self.assertEqual(normalize_color_label(" silver "), "Silver")
        with self.assertRaises(KeyError):
            normalize_color_label("unknown_metallic_x")

        self.assertEqual(set(COLOR_GROUPS_11.keys()), set(COLOR_CLASSES))
        self.assertEqual(COLOR_GROUPS_11["Beige"], "Beige/Gold")
        self.assertEqual(COLOR_GROUPS_11["Tan"], "Beige/Gold")
        self.assertEqual(COLOR_GROUPS_11["Gold"], "Beige/Gold")
        self.assertEqual(COLOR_GROUPS_11["Pink"], "Purple/Pink")
        self.assertEqual(COLOR_GROUPS_11["Purple"], "Purple/Pink")
        self.assertEqual(COLOR_GROUPS_11["Red"], "Red")

        self.assertEqual(set(COLOR_HEX.keys()), set(COLOR_CLASSES))
        hex_re = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for color, code in COLOR_HEX.items():
            self.assertRegex(code, hex_re, f"Invalid hex for {color}: {code}")

        w = compute_soft_class_weights([1000, 100, 10], power=0.5, min_weight=0.5, max_weight=5.0)
        self.assertEqual(w.dtype, torch.float32)
        self.assertTrue(bool((w >= 0.5).all() and (w <= 5.0).all()))
        self.assertGreater(float(w[2]), float(w[0]))

    def test_year_dldl_target_and_masked_loss(self) -> None:
        years = torch.tensor([2005, 1950, 2016])
        probs = year_to_gaussian_target(years)
        self.assertEqual(probs.shape, (3, NUM_YEAR_BINS))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-5))
        self.assertEqual(probs.argmax(dim=-1).tolist(), [55, 0, 66])

        logits_exact = torch.log(probs.clamp_min(1e-12))
        bins = torch.arange(YEAR_MIN, YEAR_MAX + 1, dtype=torch.float32)
        exp_exact, std_yr = year_logits_to_expected(logits_exact)
        self.assertTrue(torch.allclose(exp_exact, (probs * bins).sum(dim=-1), atol=1e-4))
        self.assertTrue(bool((std_yr > 0.0).all()))

        peaky_logits = F.one_hot(torch.tensor([55, 0, 66]), num_classes=NUM_YEAR_BINS).float() * 20.0
        exp_peaky, std_peaky = year_logits_to_expected(peaky_logits)
        self.assertTrue(torch.allclose(exp_peaky, torch.tensor([2005.0, 1950.0, 2016.0]), atol=0.01))
        self.assertTrue(bool((std_peaky < 0.05).all()))

        logits = torch.randn(2, NUM_YEAR_BINS, requires_grad=True)
        all_false = torch.tensor([False, False])
        zero_loss = dldl_year_loss(logits, torch.tensor([1900, 1938]), all_false)
        self.assertTrue(torch.isfinite(zero_loss))
        self.assertAlmostEqual(float(zero_loss.item()), 0.0, places=6)
        zero_loss.backward()

        logits2 = logits.detach().clone().requires_grad_(True)
        mask = torch.tensor([True, False])
        l_bad_1900 = dldl_year_loss(logits2, torch.tensor([2008, 1900]), mask)
        l_bad_9999 = dldl_year_loss(logits2, torch.tensor([2008, 9999]), mask)
        self.assertTrue(torch.allclose(l_bad_1900, l_bad_9999, atol=1e-6))

    def test_cosface_head_train_vs_eval(self) -> None:
        torch.manual_seed(0)
        head = CosFaceHead(in_features=512, out_features=225, scale=30.0, margin=0.35)
        emb = F.normalize(torch.randn(4, 512), dim=-1)
        targets = torch.tensor([0, 17, 104, 224])

        eval_logits = head(emb, targets=None)
        train_logits = head(emb, targets=targets)
        self.assertTrue(torch.isfinite(eval_logits).all() and torch.isfinite(train_logits).all())

        for row, tgt in enumerate(targets.tolist()):
            self.assertAlmostEqual(
                float(train_logits[row, tgt].item()),
                float(eval_logits[row, tgt].item()) - 30.0 * 0.35,
                places=4,
            )
            other = (tgt + 1) % 225
            self.assertAlmostEqual(
                float(train_logits[row, other].item()),
                float(eval_logits[row, other].item()),
                places=4,
            )


class TestMultiTaskArchitectureAndParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(42)
        cls.net = AutoVisionMultiTaskNet(pretrained=None)

    def test_phase1_and_phase2_forward_backward_invariants(self) -> None:
        net = self.net
        x = torch.randn(2, 3, 224, 224)
        mk_y = torch.tensor([0, 5])
        md_y = torch.tensor([10, 100])
        bd_y = torch.tensor([0, 1])
        yr_y = torch.tensor([2005, 1900])
        yr_v = torch.tensor([True, False])
        cl_y = torch.tensor([1, 10])
        crit = MultiTaskVmmrdbLoss()

        # 1. Phase 1 Warmup: ONLY VMMRdb heads train; visual + color_head stay eval & grad=None
        net.set_phase1_warmup_mode()
        net.train(True)
        self.assertFalse(net.visual.training)
        self.assertFalse(net.color_head.training)
        self.assertTrue(net.make_head.training)

        net.zero_grad(set_to_none=True)
        out = net(x, model_targets=md_y, return_attn=True)
        norms = out["embedding"].norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones(2), atol=1e-5))
        self.assertEqual(out["color_attn"].shape, (2, 14, 14))
        self.assertTrue(torch.allclose(out["color_attn"].sum(dim=(1, 2)), torch.ones(2), atol=1e-5))

        loss_w, _ = crit(out, mk_y, md_y, bd_y, yr_y, yr_v)
        loss_w.backward()
        self.assertTrue(all(p.grad is None for p in net.visual.parameters()))
        self.assertTrue(all(p.grad is None for p in net.color_head.parameters()))
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in net.model_head.parameters()))
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in net.year_head.parameters()))

        # 2. Phase 1 Finetune: visual + VMMRdb heads train; color_head stays eval & grad=None
        net.set_phase1_finetune_mode()
        net.train(True)
        self.assertTrue(net.visual.training)
        self.assertFalse(net.color_head.training)
        groups = net.build_phase1_param_groups(2.5e-5, 1e-4, 1e-3, 0.02)
        lrs = {g["lr"] for g in groups}
        self.assertEqual(lrs, {2.5e-5, 1e-4, 1e-3})

        net.zero_grad(set_to_none=True)
        out_ft = net(x, model_targets=md_y)
        loss_ft, _ = crit(out_ft, mk_y, md_y, bd_y, yr_y, yr_v)
        loss_ft.backward()
        self.assertTrue(all(p.grad is None for p in net.color_head.parameters()))
        self.assertTrue(any(p.grad is not None for p in net.visual.parameters()))

        # 3. Phase 2 Color: ONLY color_head trains; visual + 4 VMMRdb heads stay eval & grad=None
        net.set_phase2_color_mode()
        net.train(True)
        self.assertFalse(net.visual.training)
        self.assertFalse(net.make_head.training)
        self.assertFalse(net.model_head.training)
        self.assertTrue(net.color_head.training)

        net.zero_grad(set_to_none=True)
        out_c = net(x)
        loss_c = F.cross_entropy(out_c["color_logits"], cl_y)
        loss_c.backward()
        self.assertTrue(all(p.grad is None for p in net.visual.parameters()))
        self.assertTrue(all(p.grad is None for p in net.make_head.parameters()))
        self.assertTrue(all(p.grad is None for p in net.model_head.parameters()))
        self.assertTrue(all(p.grad is None for p in net.body_head.parameters()))
        self.assertTrue(all(p.grad is None for p in net.year_head.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.color_head.parameters()))

        # 4. ONNX wrapper output tuple shapes
        wrapper = AutoVisionOnnxWrapper(net.eval())
        with torch.no_grad():
            tup = wrapper(x)
        self.assertEqual(len(tup), 6)
        self.assertEqual([tuple(t.shape) for t in tup], [(2, 512), (2, 22), (2, 225), (2, 8), (2, 67), (2, 15)])

    def test_kaggle_script_strict_parity_and_checkpoint_load(self) -> None:
        kag = _load_kaggle_module()

        self.assertEqual(kag.BODY_TYPE_CLASSES, bam.BODY_TYPE_CLASSES)
        self.assertEqual(kag.MODEL_BODY_TYPE, bam.MODEL_BODY_TYPE)
        self.assertEqual(len(kag.MODEL_BODY_TYPE), 342)
        for cls_name, raw_model, hint, yr in [
            ("volkswagen__passat", "passat_wagon", "Wagon", 2005),
            ("toyota__solara", "solara_convertable", "Convertible", 2005),
            ("saturn__s-series", "sw2", "", 2000),
            ("saturn__l-series", "lw300", "", 2002),
            ("mercury__monterey", "monterey", "", 2005),
            ("mercury__monterey", "monterey", "", 1955),
        ]:
            self.assertEqual(
                kag.resolve_body_type(cls_name, raw_model, hint, yr, ""),
                bam.resolve_body_type(cls_name, raw_model, hint, yr, ""),
            )

        self.assertEqual(kag.COLOR_CLASSES, COLOR_CLASSES)
        self.assertEqual(kag.COLOR_GROUPS_11, COLOR_GROUPS_11)
        self.assertEqual(kag.YEAR_MIN, YEAR_MIN)
        self.assertEqual(kag.YEAR_MAX, YEAR_MAX)

        kag_net = kag.AutoVisionMultiTaskNet(pretrained=None).eval()
        src_net = AutoVisionMultiTaskNet(pretrained=None).eval()
        src_net.load_state_dict(kag_net.state_dict(), strict=True)

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out_kag = kag_net(x)
            out_src = src_net(x)
        for key in ("embedding", "make_logits", "model_logits", "body_logits", "year_logits", "color_logits"):
            self.assertTrue(
                torch.allclose(out_kag[key], out_src[key], atol=1e-6),
                f"Mismatch on output {key}",
            )

    def test_vcor_discovery_loader_and_crop_fallback(self) -> None:
        kag = _load_kaggle_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "kaggle_in" / "vcor_ds"
            folders = ["beige", "black", "blue", "brown", "green", "grey", "red", "silver", "white"]
            for split in ("train", "val", "test"):
                for fld in folders:
                    d = root / split / fld
                    d.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (64, 48), (120, 120, 120)).save(d / f"{split}_{fld}.png")
                (root / split / ".ipynb_checkpoints").mkdir(exist_ok=True)

            found = kag.find_vcor_root(Path(tmp) / "kaggle_in")
            self.assertEqual(found, root)
            tr_samples = kag.scan_vcor_split(root / "train")
            self.assertEqual(len(tr_samples), 9)
            by_rel = {s["rel_path"]: s["color"] for s in tr_samples}
            self.assertEqual(by_rel["train/grey/train_grey.png"], "Gray")
            self.assertEqual(by_rel["train/silver/train_silver.png"], "Silver")

        # Large box > 15% area -> padded YOLO box (used=True)
        box_big, row_big, used_big = kag.compute_vcor_crop_box(
            400, 300, np.array([[50.0, 50.0, 300.0, 220.0]]), np.array([0.88])
        )
        self.assertTrue(used_big)
        self.assertEqual(box_big, (37, 41, 312, 228))
        self.assertGreater(row_big[5], 0.15)

        # Tiny box <= 15% area -> full image fallback (0, 0, w, h)
        box_small, _, used_small = kag.compute_vcor_crop_box(
            400, 300, np.array([[10.0, 10.0, 50.0, 40.0]]), np.array([0.75])
        )
        self.assertFalse(used_small)
        self.assertEqual(box_small, (0, 0, 400, 300))

        # Empty box -> full image fallback (0, 0, w, h)
        box_empty, _, used_empty = kag.compute_vcor_crop_box(
            400, 300, np.empty((0, 4)), np.empty((0,))
        )
        self.assertFalse(used_empty)
        self.assertEqual(box_empty, (0, 0, 400, 300))


if __name__ == "__main__":
    unittest.main()
