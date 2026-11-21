"""Unit and parity tests for AutoVision v2 taxonomy, interval DLDL loss, and Kaggle trainer."""

from __future__ import annotations

import ast
import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from multitask_model import (  # noqa: E402
    AutoVisionMultiTaskNet,
    dldl_interval_year_loss,
    dldl_year_loss,
    year_logits_to_expected,
)
from v2_taxonomy import (  # noqa: E402
    BODY_TYPE_CLASSES,
    NUM_YEAR_BINS_V2,
    YEAR_MAX_V2,
    YEAR_MIN_V2,
    canonicalize_make_model,
    compute_padded_crop_box,
    parse_timdwms_year_interval,
    refine_body_type_from_text,
    resolve_default_body_type,
)


class V2TaxonomyAndLossTests(unittest.TestCase):
    def test_canonicalize_unifies_vmmrdb_aliases_and_displays(self) -> None:
        self.assertEqual(
            canonicalize_make_model("Mercedes-Benz", "SL")[1],
            "mercedes-benz__sl-class",
        )
        self.assertEqual(
            canonicalize_make_model("Dodge", "RAM_1500")[:2],
            ("ram", "ram__1500"),
        )
        self.assertEqual(
            canonicalize_make_model("Toyota", "Landcruiser")[:2],
            ("toyota", "toyota__land_cruiser"),
        )
        mk, cls_name, mk_disp, md_disp = canonicalize_make_model("Mazda", "MX5_MIATA")
        self.assertEqual((mk, cls_name, mk_disp, md_disp), ("mazda", "mazda__mx-5_miata", "Mazda", "MX-5 Miata"))

        _, cls_p2, mk_p2, md_p2 = canonicalize_make_model("Polestar", "Polestar 2")
        self.assertEqual((cls_p2, mk_p2, md_p2), ("polestar__polestar_2", "Polestar", "Polestar 2"))

    def test_body_type_defaults_and_wikimedia_refinement(self) -> None:
        self.assertEqual(resolve_default_body_type("bmw__3_series"), "Sedan")
        self.assertEqual(resolve_default_body_type("bmw__x5"), "SUV")
        self.assertEqual(resolve_default_body_type("tesla__cybertruck"), "Pickup")
        self.assertEqual(resolve_default_body_type("toyota__sienna"), "Minivan")
        self.assertEqual(resolve_default_body_type("ford__mustang"), "Coupe")
        self.assertEqual(resolve_default_body_type("volkswagen__golf"), "Hatchback")
        self.assertEqual(resolve_default_body_type("subaru__outback"), "Wagon")
        self.assertEqual(resolve_default_body_type("honda__s2000"), "Convertible")

        self.assertEqual(
            refine_body_type_from_text("Sedan", "BMW 3 Series (F31) Touring"),
            "Wagon",
        )
        self.assertEqual(
            refine_body_type_from_text("Coupe", "Porsche 911 (992) Cabriolet"),
            "Convertible",
        )
        self.assertEqual(
            refine_body_type_from_text("Sedan", "BMW 4 Series Gran Coupe (F36)"),
            "Sedan",
        )
        self.assertEqual(
            refine_body_type_from_text("SUV", "BMW X6 Coupe SUV"),
            "SUV",
        )

    def test_parse_year_interval_phase_and_photo_date_clamp(self) -> None:
        lo_pre, hi_pre, ok = parse_timdwms_year_interval("2014-2024", phase_str="pre", date_taken_str="2016-08-10")
        self.assertTrue(ok)
        self.assertEqual(lo_pre, 2014.0)
        self.assertLessEqual(hi_pre, 2017.0)

        lo_fl, hi_fl, ok_fl = parse_timdwms_year_interval("2014-2024", phase_str="facelift")
        self.assertTrue(ok_fl)
        self.assertGreater(lo_fl, 2014.0)
        self.assertEqual(hi_fl, 2024.0)

        self.assertEqual(parse_timdwms_year_interval("unknown"), (0.0, 0.0, False))

    def test_compute_padded_crop_box_clamps_cleanly(self) -> None:
        box = compute_padded_crop_box("100 100 500 300", img_w=600, img_h=400, pad=0.05)
        self.assertEqual(box, (80, 90, 520, 310))
        self.assertIsNone(compute_padded_crop_box("bad box", img_w=600, img_h=400))
        self.assertIsNone(compute_padded_crop_box("10 10 20 20", img_w=600, img_h=400))

    def test_interval_dldl_matches_point_gaussian_and_gradients(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(4, NUM_YEAR_BINS_V2, requires_grad=True)
        years = torch.tensor([2005.0, 2016.0, 2023.0, 1998.0])
        valid = torch.tensor([True, True, True, False])

        loss_int = dldl_interval_year_loss(
            logits, years, years, valid, year_min=YEAR_MIN_V2, year_max=YEAR_MAX_V2, sigma=1.5
        )
        loss_pt = dldl_year_loss(
            logits, years, valid, year_min=YEAR_MIN_V2, year_max=YEAR_MAX_V2, sigma=1.5
        )
        self.assertAlmostEqual(float(loss_int.item()), float(loss_pt.item()), places=5)
        loss_int.backward()
        self.assertIsNotNone(logits.grad)

        exp, std = year_logits_to_expected(logits.detach(), YEAR_MIN_V2, YEAR_MAX_V2)
        self.assertTrue(torch.all((exp >= YEAR_MIN_V2) & (exp <= YEAR_MAX_V2)))
        self.assertTrue(torch.all(std >= 0.0))

    def test_multitask_net_v2_dims_and_kaggle_script_parity(self) -> None:
        kaggle_script = ROOT / "kaggle" / "train_autovision_v2_phase1.py"
        src_text = kaggle_script.read_text(encoding="utf-8")
        compile(src_text, str(kaggle_script), "exec")

        net = AutoVisionMultiTaskNet(
            num_makes=49,
            num_models=560,
            num_bodies=len(BODY_TYPE_CLASSES),
            num_years=NUM_YEAR_BINS_V2,
            pretrained=None,
        )
        x = torch.randn(2, 3, 224, 224)
        out = net(x, model_targets=torch.tensor([0, 1]))
        self.assertEqual(tuple(out["year_logits"].shape), (2, NUM_YEAR_BINS_V2))
        self.assertEqual(tuple(out["make_logits"].shape), (2, 49))
        self.assertEqual(tuple(out["model_logits"].shape), (2, 560))

        # Check static AST constants parity in Kaggle script
        tree = ast.parse(kaggle_script.read_text(encoding="utf-8"))
        assigns = {
            node.target.id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in ("YEAR_MIN", "YEAR_MAX", "BODY_TYPE_CLASSES")
            and node.value is not None
        }
        self.assertEqual(assigns.get("YEAR_MIN"), YEAR_MIN_V2)
        self.assertEqual(assigns.get("YEAR_MAX"), YEAR_MAX_V2)
        self.assertEqual(assigns.get("BODY_TYPE_CLASSES"), BODY_TYPE_CLASSES)

    def test_audit_timdwms_csv_if_present(self) -> None:
        audit_csv = (
            Path(r"C:\Users\Eldar Semenov\.gemini\antigravity\brain\d3f4976b-9417-496c-a205-2fedbd2218ed\scratch\v2_audit")
            / "timdwms_metadata.csv"
        )
        if not audit_csv.is_file():
            self.skipTest("Scratch timdwms_metadata.csv not present")
        with open(audit_csv, "r", encoding="utf-8", newline="") as f:
            seen_classes = set()
            for r in csv.DictReader(f):
                if r["filter"] == "ok" and r["zip"] != "_Multiple cars.zip" and r["make"].strip() and r["model"].strip():
                    _, cls_name, _, _ = canonicalize_make_model(r["make"], r["model"])
                    bt = resolve_default_body_type(cls_name)
                    self.assertIn(bt, BODY_TYPE_CLASSES)
                    seen_classes.add(cls_name)
            self.assertGreaterEqual(len(seen_classes), 550)

    def test_discover_and_load_vmmrdb_handles_kaggle_rel_path_and_6col_bboxes(self) -> None:
        kaggle_script = ROOT / "kaggle" / "train_autovision_v2_phase1.py"
        spec = importlib.util.spec_from_file_location("train_autovision_v2_phase1", kaggle_script)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_root = tmp_path / "input"
            cache_dir = tmp_path / "cache"

            # Decoy sibling VCoR dataset folder + AutoVision VMMRdb Clean folder
            (input_root / "vcor" / "images").mkdir(parents=True)
            vmmr_root = input_root / "autovision-vmmrdb-clean"
            img_file = vmmr_root / "packed" / "mercedes-benz_s-class_2007" / "car1.jpg"
            img_file.parent.mkdir(parents=True)
            Image.new("RGB", (200, 150), color=(20, 20, 25)).save(img_file, format="JPEG")

            manifest = vmmr_root / "local_vmmrdb_multitask_manifest.csv"
            with open(manifest, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "rel_path", "kaggle_rel_path", "make", "canonical_class",
                        "raw_year", "year_valid", "split", "body_type", "is_clean",
                    ],
                )
                w.writeheader()
                w.writerow({
                    "rel_path": "mercedes-benz/s-class/2007/car1.jpg",
                    "kaggle_rel_path": "packed/mercedes-benz_s-class_2007/car1.jpg",
                    "make": "Mercedes-Benz",
                    "canonical_class": "mercedes-benz__s-class",
                    "raw_year": "2007",
                    "year_valid": "true",
                    "split": "train",
                    "body_type": "Sedan",
                    "is_clean": "true",
                })

            bbox_file = input_root / "bboxes-yolo" / "bboxes_yolo.npz"
            bbox_file.parent.mkdir(parents=True)
            np.savez(
                bbox_file,
                rel_paths=np.array(["mercedes-benz/s-class/2007/car1.jpg"]),
                bboxes=np.array([[20.0, 15.0, 180.0, 135.0, 0.95, 0.64]], dtype=np.float32),
                used_yolo=np.array([True]),
            )

            samples = mod.discover_and_load_vmmrdb(cache_dir, input_root=input_root)
            self.assertEqual(len(samples), 1)
            s = samples[0]
            self.assertEqual(s["canonical_class"], "mercedes-benz__s-class")
            self.assertEqual(s["body_type"], "Sedan")
            self.assertEqual((s["year_low"], s["year_high"], s["year_valid"]), (2007.0, 2007.0, True))
            self.assertTrue(Path(s["abs_path"]).is_file())
            with Image.open(s["abs_path"]) as cropped:
                self.assertEqual(cropped.size, (160, 120))


class TestAutoVisionV2DatasetFewShot(unittest.TestCase):
    def test_unseen_few_shot_make_and_body_fallback(self) -> None:
        import importlib.util
        import tempfile
        from pathlib import Path
        from PIL import Image
        import torch

        script_path = Path(__file__).resolve().parents[1] / "kaggle" / "train_autovision_v2_phase1.py"
        spec = importlib.util.spec_from_file_location("train_autovision_v2_phase1", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "morris_minor.jpg"
            Image.new("RGB", (64, 64), color=(120, 130, 140)).save(img_path, format="JPEG")

            samples = [
                {
                    "abs_path": str(img_path),
                    "make": "morris",
                    "canonical_class": "morris__minor",
                    "body_type": "Sedan",
                    "year_low": 1962.0,
                    "year_high": 1962.0,
                    "year_valid": True,
                },
                {
                    "abs_path": str(img_path),
                    "make": "morris",
                    "canonical_class": "morris__minor",
                    "body_type": "UnknownBody",
                    "year_low": 1950.0,
                    "year_high": 2026.0,
                    "year_valid": False,
                },
            ]
            ds = mod.AutoVisionV2Dataset(
                samples,
                {"ford": 0},
                {"ford__mustang": 0},
                {"Sedan": 0, "SUV": 1},
                mod.build_eval_transform(),
            )
            img_t, mk_y, md_y, bd_y, y_lo, y_hi, y_val = ds[0]
            self.assertIsInstance(img_t, torch.Tensor)
            self.assertEqual(img_t.ndim, 3)
            self.assertEqual(img_t.shape[0], 3)
            self.assertEqual((mk_y, md_y, bd_y), (-1, -1, 0))
            self.assertEqual((y_lo, y_hi, y_val), (1962.0, 1962.0, 1))

            _, mk_y2, md_y2, bd_y2, _, _, y_val2 = ds[1]
            self.assertEqual((mk_y2, md_y2, bd_y2, y_val2), (-1, -1, 0, 0))


if __name__ == "__main__":
    unittest.main()
