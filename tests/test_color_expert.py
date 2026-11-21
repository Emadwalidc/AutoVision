"""Unit and parity tests for AutoVision v2 standalone Color Expert."""

from __future__ import annotations

import ast
import random
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from color_expert import (  # noqa: E402
    COLOR_CLASSES_V2,
    COLOR_HEX_V2,
    NEUTRAL_BG_RGB,
    AutoVisionColorExpert,
    apply_illuminant_reflection_cast,
    apply_neutral_gray_mask,
    extract_masked_lab_features,
    normalize_color_v2,
    parse_dvm_color_from_path,
    remove_dvm_white_border_bg,
)


class TestColorExpertV2(unittest.TestCase):
    def test_color_normalization_and_dvm_path_parsing(self) -> None:
        self.assertEqual(normalize_color_v2("Grey"), "Gray")
        self.assertEqual(normalize_color_v2("Bronze"), "Brown")
        self.assertEqual(normalize_color_v2("Maroon"), "Red")
        self.assertEqual(normalize_color_v2("Gold"), "Beige")
        self.assertEqual(normalize_color_v2("Tan"), "Beige")
        self.assertEqual(normalize_color_v2("Pink"), "Purple")
        self.assertIsNone(normalize_color_v2("Unlisted"))
        self.assertIsNone(normalize_color_v2("Multicolour"))

        self.assertEqual(
            parse_dvm_color_from_path("Abarth$$595$$2018$$White$$18_1$$99.jpg"),
            "White",
        )
        self.assertEqual(
            parse_dvm_color_from_path("BMW/3 Series/2020/Grey/front_01.jpg"),
            "Gray",
        )
        for c in COLOR_CLASSES_V2:
            self.assertIn(c, COLOR_HEX_V2)

    def test_white_border_flood_fill_preserves_interior_white(self) -> None:
        arr = np.full((64, 64, 3), 255, dtype=np.uint8)
        # Draw a dark bumper frame enclosing a pure white painted hood panel in center
        arr[16:48, 16:48] = (20, 20, 22)
        arr[22:42, 22:42] = (252, 252, 252)

        masked_im, car_mask = remove_dvm_white_border_bg(Image.fromarray(arr))
        out = np.asarray(masked_im)
        # Exterior border must turn neutral gray
        self.assertEqual(tuple(out[2, 2]), NEUTRAL_BG_RGB)
        self.assertFalse(bool(car_mask[2, 2]))
        # Interior white hood must remain white inside the vehicle mask
        self.assertGreater(int(out[32, 32, 0]), 240)
        self.assertTrue(bool(car_mask[32, 32]))

    def test_lab_descriptor_separates_black_brown_and_white(self) -> None:
        mask = np.ones((48, 48), dtype=bool)
        black_img = np.full((48, 48, 3), (18, 18, 20), dtype=np.uint8)
        brown_img = np.full((48, 48, 3), (112, 66, 36), dtype=np.uint8)
        white_img = np.full((48, 48, 3), (242, 242, 245), dtype=np.uint8)

        f_black = extract_masked_lab_features(black_img, mask)
        f_brown = extract_masked_lab_features(brown_img, mask)
        f_white = extract_masked_lab_features(white_img, mask)

        self.assertEqual(f_black.shape, (16,))
        self.assertGreater(f_black[13], 0.9)   # dark_achrom_frac
        self.assertLess(f_black[15], 0.05)     # warm_brown_frac
        self.assertGreater(f_brown[15], 0.9)   # warm_brown_frac
        self.assertGreater(f_white[14], 0.9)   # bright_achrom_frac

    def test_illuminant_cast_only_affects_achromatic_within_mask(self) -> None:
        arr = np.full((40, 40, 3), (30, 30, 32), dtype=np.uint8)
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:30, 10:30] = True
        masked_im, final_mask = apply_neutral_gray_mask(Image.fromarray(arr), mask, erode_px=0)

        rng = random.Random(123)
        aug = apply_illuminant_reflection_cast(masked_im, "Black", rng, mask_bool=final_mask, prob=1.0)
        aug_arr = np.asarray(aug)
        # Background pixels remain exact neutral gray
        self.assertEqual(tuple(aug_arr[2, 2]), NEUTRAL_BG_RGB)

        red_im = Image.fromarray(np.full((40, 40, 3), (210, 20, 20), dtype=np.uint8))
        red_unchanged = apply_illuminant_reflection_cast(red_im, "Red", rng, prob=1.0)
        self.assertTrue(np.array_equal(np.asarray(red_im), np.asarray(red_unchanged)))

    def test_color_expert_forward_modes_and_kaggle_script_parity(self) -> None:
        model = AutoVisionColorExpert(num_colors=len(COLOR_CLASSES_V2), pretrained=False)
        model.set_warmup_mode()
        self.assertTrue(all(not p.requires_grad for p in model.backbone_features.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))

        model.set_finetune_mode()
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

        x = torch.randn(2, 3, 224, 224)
        lab = torch.randn(2, 16)
        logits = model(x, lab)
        self.assertEqual(tuple(logits.shape), (2, len(COLOR_CLASSES_V2)))
        logits.sum().backward()

        kaggle_script = ROOT / "kaggle" / "train_autovision_v2_color.py"
        src_text = kaggle_script.read_text(encoding="utf-8")
        compile(src_text, str(kaggle_script), "exec")
        tree = ast.parse(src_text)
        assigns = {
            node.target.id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "COLOR_CLASSES_V2"
            and node.value is not None
        }
        self.assertEqual(assigns.get("COLOR_CLASSES_V2"), COLOR_CLASSES_V2)


if __name__ == "__main__":
    unittest.main()
