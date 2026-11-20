"""Automated unit and end-to-end tests for feature extraction and evaluation pipeline."""

from __future__ import annotations

import csv
import os
import py_compile
import re
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CACHE_ULTRALYTICS = BASE_DIR / ".cache" / "ultralytics"
CACHE_ULTRALYTICS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_ULTRALYTICS))

from evaluate_features_baseline import evaluate_all  # noqa: E402
from extract_features import (  # noqa: E402
    compute_vehicle_crop_box,
    load_feature_bundle,
    save_feature_bundle,
)
from pack_kaggle_dataset import KAGGLE_SAFE_RE, pack_kaggle_dataset  # noqa: E402


class TestCropAndBundleIntegrity(unittest.TestCase):
    def test_compute_vehicle_crop_box_variants(self) -> None:
        # 1. Large valid box gets 5% padding clamped to image bounds
        crop_box, row6, used = compute_vehicle_crop_box(
            600, 450, np.array([[100.0, 100.0, 400.0, 300.0]]), np.array([0.92])
        )
        self.assertTrue(used)
        self.assertEqual(crop_box, (85, 90, 415, 310))
        self.assertAlmostEqual(row6[4], 0.92, places=3)
        self.assertGreater(row6[5], 0.08)

        # 2. Tiny background box (<= 8% area) triggers 85% center fallback
        crop_small, row6_small, used_small = compute_vehicle_crop_box(
            600, 450, np.array([[10.0, 10.0, 40.0, 40.0]]), np.array([0.75])
        )
        self.assertFalse(used_small)
        self.assertEqual(crop_small, (45, 33, 555, 417))
        self.assertLessEqual(row6_small[5], 0.08)

        # 3. Empty detections triggers 85% center fallback
        crop_empty, row6_empty, used_empty = compute_vehicle_crop_box(
            600, 450, np.empty((0, 4)), np.empty((0,))
        )
        self.assertFalse(used_empty)
        self.assertEqual(crop_empty, (45, 33, 555, 417))
        self.assertEqual(row6_empty[4], 0.0)

    def test_real_yolo_smoke_on_two_images(self) -> None:
        weights = BASE_DIR / "yolov8n.pt"
        data_dir = Path(r"D:\Projects\bitirme\data")
        sample_imgs = [
            data_dir / "toyota_camry_2007" / "00000_6J7PfFKy1oa_600x450.jpg",
            data_dir / "mercedes benz_c300_2012" / "2012 Mercedes c300 4matic_00J0J_aOwoYr5uBeI_600x450.jpg",
        ]
        if not weights.exists() or not all(p.is_file() for p in sample_imgs):
            self.skipTest("Real sample images or yolov8n.pt unavailable")

        from ultralytics import YOLO

        model = YOLO(str(weights))
        imgs = [Image.open(p).convert("RGB") for p in sample_imgs]
        try:
            results = model.predict(imgs, classes=[2, 5, 7], conf=0.25, verbose=False)
            for im, res in zip(imgs, results):
                xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
                confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.empty((0,))
                crop_box, _, used = compute_vehicle_crop_box(im.size[0], im.size[1], xyxy, confs)
                self.assertTrue(used)
                self.assertGreater(crop_box[2], crop_box[0])
                self.assertGreater(crop_box[3], crop_box[1])
        finally:
            for im in imgs:
                im.close()

    def test_bundle_save_load_and_validation(self) -> None:
        rng = np.random.default_rng(7)
        vecs = rng.normal(size=(10, 64)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        boxes = np.tile(np.array([10, 10, 200, 150, 0.9, 0.5], dtype=np.float32), (10, 1))
        used = np.ones(10, dtype=bool)
        rels = [f"make_model_2005/img_{i}.jpg" for i in range(10)]
        meta = {"backbone": "openclip_vitb16", "embed_dim": 64}

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bundle.npz"
            save_feature_bundle(p, vecs, boxes, used, rels, meta)
            loaded = load_feature_bundle(p)
            self.assertEqual(loaded["embeddings"].shape, (10, 64))
            self.assertEqual(loaded["metadata"]["backbone"], "openclip_vitb16")

            bad_nan = vecs.copy()
            bad_nan[0, 0] = np.nan
            with self.assertRaises(ValueError):
                save_feature_bundle(p, bad_nan, boxes, used, rels, meta)

            bad_zero = vecs.copy()
            bad_zero[3, :] = 0.0
            with self.assertRaises(ValueError):
                save_feature_bundle(p, bad_zero, boxes, used, rels, meta)


class TestPackAndEvaluateEndToEnd(unittest.TestCase):
    def test_pack_kaggle_dry_run_on_real_manifest(self) -> None:
        manifest = BASE_DIR / "data_manifests" / "local_vmmrdb_manifest.csv"
        data_dir = Path(r"D:\Projects\bitirme\data")
        if not manifest.is_file() or not data_dir.is_dir():
            self.skipTest("Local manifest or data_dir unavailable")
        summary = pack_kaggle_dataset(
            manifest_path=manifest,
            data_dir=data_dir,
            out_zip=BASE_DIR / "kaggle" / "vmmrdb_clean_89k.zip",
            dry_run=True,
        )
        self.assertEqual(summary["clean_images"], 89535)
        self.assertEqual(summary["missing_images"], 0)
        self.assertGreater(summary["total_gb"], 3.0)

    def test_evaluate_features_baseline_end_to_end(self) -> None:
        rng = np.random.default_rng(42)
        dim = 64
        n_main_classes = 18
        n_few_classes = 4
        makes = ["toyota", "nissan", "volkswagen", "subaru", "mazda", "porsche"]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            data_dir = tmp_dir / "data"
            out_dir = tmp_dir / "reports"
            manifest_path = tmp_dir / "manifest.csv"

            rows = []
            embs_clean = []
            rels = []
            used_yolo = []
            bboxes = []

            ablation_embs = []
            ablation_rels = []

            class_centers = {}
            for c_idx in range(n_main_classes + n_few_classes):
                center = rng.normal(size=dim).astype(np.float32)
                center /= np.linalg.norm(center)
                class_centers[c_idx] = center

            # 18 main classes x 20 images (14 train, 3 val, 3 test)
            for c_idx in range(n_main_classes):
                make = makes[c_idx % len(makes)]
                model = f"model{c_idx}"
                cls_name = f"{make}__{model}"
                color = tuple(int(v) for v in rng.integers(30, 230, size=3))

                for j in range(20):
                    year = 2000 + (j % 12)
                    split = "train" if j < 14 else ("val" if j < 17 else "test")
                    rel = f"{make}_{model}_{year}/img_{c_idx}_{j}.png"
                    img_file = data_dir / rel
                    img_file.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (96, 72), color=color).save(img_file)

                    # Inject year signal in dims 0..1 alongside class center
                    vec = class_centers[c_idx].copy()
                    vec[0] += (year - 2005) * 0.15
                    vec += rng.normal(scale=0.05, size=dim).astype(np.float32)
                    vec /= np.linalg.norm(vec)

                    embs_clean.append(vec)
                    rels.append(rel)
                    used_yolo.append(True)
                    bboxes.append([4.0, 4.0, 92.0, 68.0, 0.9, 0.8])
                    rows.append(
                        {
                            "rel_path": rel,
                            "make": make,
                            "model": model,
                            "canonical_class": cls_name,
                            "raw_year": year,
                            "year_valid": "True",
                            "is_clean": "True",
                            "split": split,
                        }
                    )

                    if split == "test":
                        raw_vec = vec + rng.normal(scale=0.25, size=dim).astype(np.float32)
                        raw_vec /= np.linalg.norm(raw_vec)
                        ablation_embs.append(raw_vec)
                        ablation_rels.append(rel)

            # 4 few-shot classes x 3 images
            for f_i in range(n_few_classes):
                c_idx = n_main_classes + f_i
                make = makes[f_i % len(makes)]
                model = f"rare{f_i}"
                cls_name = f"{make}__{model}"
                for j in range(3):
                    year = 2008 + j
                    rel = f"{make}_{model}_{year}/rare_{f_i}_{j}.png"
                    img_file = data_dir / rel
                    img_file.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (96, 72), color=(120, 140, 160)).save(img_file)

                    vec = class_centers[c_idx] + rng.normal(scale=0.04, size=dim).astype(np.float32)
                    vec /= np.linalg.norm(vec)
                    embs_clean.append(vec)
                    rels.append(rel)
                    used_yolo.append(True)
                    bboxes.append([4.0, 4.0, 92.0, 68.0, 0.9, 0.8])
                    rows.append(
                        {
                            "rel_path": rel,
                            "make": make,
                            "model": model,
                            "canonical_class": cls_name,
                            "raw_year": year,
                            "year_valid": "True",
                            "is_clean": "True",
                            "split": "few_shot",
                        }
                    )

            with open(manifest_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "rel_path",
                        "make",
                        "model",
                        "canonical_class",
                        "raw_year",
                        "year_valid",
                        "is_clean",
                        "split",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)

            feat_path = tmp_dir / "features_openclip_vitb16.npz"
            save_feature_bundle(
                feat_path,
                np.vstack(embs_clean),
                np.array(bboxes, dtype=np.float32),
                np.array(used_yolo, dtype=bool),
                rels,
                {"backbone": "openclip_vitb16", "yolo_hit_rate": 1.0},
            )

            abl_path = tmp_dir / "ablation_raw_test_openclip_vitb16.npz"
            save_feature_bundle(
                abl_path,
                np.vstack(ablation_embs),
                np.tile(np.array([0.0, 0.0, 96.0, 72.0, 0.0, 1.0], dtype=np.float32), (len(ablation_rels), 1)),
                np.zeros(len(ablation_rels), dtype=bool),
                ablation_rels,
                {"backbone": "openclip_vitb16", "no_yolo": True},
            )

            res = evaluate_all(
                manifest_path=manifest_path,
                feature_paths=[feat_path],
                ablation_paths=[abl_path],
                data_dir=data_dir,
                out_dir=out_dir,
                epochs=15,
                seed=42,
            )

            self.assertTrue(Path(res["metrics_csv"]).is_file())
            self.assertTrue(Path(res["report_md"]).is_file())
            self.assertTrue(res["montage_created"])

            for fig_name in (
                "feature_pca_variance.png",
                "feature_tsne_models.png",
                "feature_year_regression.png",
                "feature_retrieval_montage.png",
            ):
                fig_p = out_dir / "figures" / fig_name
                self.assertTrue(fig_p.is_file(), fig_name)
                self.assertGreater(fig_p.stat().st_size, 1000, fig_name)

            with open(res["metrics_csv"], encoding="utf-8", newline="") as f:
                m_rows = list(csv.DictReader(f))
            lp_model = next(
                r for r in m_rows if r["task"] == "Model" and r["method"] == "Linear Probe"
            )
            self.assertGreater(float(lp_model["top1"]), 0.90)

            # Also verify --fuse across two feature bundles + raw ablation bundles
            feat2_path = tmp_dir / "features_dinov2_vitb14.npz"
            save_feature_bundle(
                feat2_path,
                np.vstack(embs_clean),
                np.array(bboxes, dtype=np.float32),
                np.array(used_yolo, dtype=bool),
                rels,
                {"backbone": "dinov2_vitb14", "yolo_hit_rate": 1.0},
            )
            abl2_path = tmp_dir / "ablation_raw_test_dinov2_vitb14.npz"
            save_feature_bundle(
                abl2_path,
                np.vstack(ablation_embs),
                np.tile(np.array([0.0, 0.0, 96.0, 72.0, 0.0, 1.0], dtype=np.float32), (len(ablation_rels), 1)),
                np.zeros(len(ablation_rels), dtype=bool),
                ablation_rels,
                {"backbone": "dinov2_vitb14", "no_yolo": True},
            )
            fused_out = tmp_dir / "reports_fused"
            res_fused = evaluate_all(
                manifest_path=manifest_path,
                feature_paths=[feat_path, feat2_path],
                ablation_paths=[abl_path, abl2_path],
                data_dir=data_dir,
                out_dir=fused_out,
                epochs=10,
                seed=42,
                fuse=True,
            )
            with open(res_fused["metrics_csv"], encoding="utf-8", newline="") as f:
                fused_rows = list(csv.DictReader(f))
            fused_backbones = {r["backbone"] for r in fused_rows}
            self.assertIn("fusion_openclip_dinov2", fused_backbones)
            fused_abl = [
                r for r in fused_rows
                if r["backbone"] == "fusion_openclip_dinov2" and r["split"] == "ablation_test"
            ]
            self.assertEqual(len(fused_abl), 2)

    def test_kaggle_script_compiles(self) -> None:
        kaggle_script = BASE_DIR / "kaggle" / "extract_vmmrdb_full_kaggle.py"
        py_compile.compile(str(kaggle_script), doraise=True)

    def test_pack_safe_ascii_paths_and_repack(self) -> None:
        special_rels = [
            "toyota_yaris_2013/2013 Toyota Yair's (1)+test,$5.jpg",
            "willys_cj3b_1954/1954 Willy's, CJ3B.png",
        ]
        excluded_rel = "suzuki_boulevard_2006/excluded bike.jpg"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            data_dir = tmp_dir / "data"
            for rel, col in [(special_rels[0], (200, 30, 30)), (special_rels[1], (30, 120, 200)), (excluded_rel, (0, 0, 0))]:
                p = data_dir / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (64, 48), color=col).save(p)

            manifest_path = tmp_dir / "local_vmmrdb_manifest.csv"
            with open(manifest_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["rel_path", "file_size", "is_clean", "split"])
                writer.writeheader()
                for rel in special_rels:
                    writer.writerow({"rel_path": rel, "file_size": str((data_dir / rel).stat().st_size), "is_clean": "True", "split": "train"})
                writer.writerow({"rel_path": excluded_rel, "file_size": "100", "is_clean": "False", "split": "excluded"})

            safe_zip = tmp_dir / "safe.zip"
            res = pack_kaggle_dataset(manifest_path, data_dir, safe_zip)
            self.assertEqual(res["clean_images"], 2)

            with zipfile.ZipFile(safe_zip, "r") as zf:
                names = zf.namelist()
                self.assertEqual(len(names), 3)
                for name in names:
                    self.assertIsNotNone(KAGGLE_SAFE_RE.fullmatch(name), f"Unsafe ZIP name: {name}")
                with zf.open("local_vmmrdb_manifest.csv") as mf:
                    m_rows = list(csv.DictReader(line.decode("utf-8") for line in mf))
                self.assertEqual(len(m_rows), 2)
                self.assertEqual([r["rel_path"] for r in m_rows], special_rels)
                self.assertEqual([r["kaggle_rel_path"] for r in m_rows], ["images/000/000000.jpg", "images/000/000001.png"])
                for r in m_rows:
                    self.assertIn(r["kaggle_rel_path"], names)

            # Test --repack-from-zip starting from a legacy-named ZIP
            legacy_zip = tmp_dir / "legacy.zip"
            with zipfile.ZipFile(legacy_zip, "w", compression=zipfile.ZIP_STORED) as zf_leg:
                zf_leg.write(manifest_path, arcname="local_vmmrdb_manifest.csv")
                for rel in special_rels:
                    zf_leg.write(data_dir / rel, arcname=f"images/{rel}")

            pack_kaggle_dataset(manifest_path, data_dir, legacy_zip, repack_from_zip=legacy_zip)
            with zipfile.ZipFile(legacy_zip, "r") as zf_repacked:
                names_rep = zf_repacked.namelist()
                self.assertEqual(sorted(names_rep), ["images/000/000000.jpg", "images/000/000001.png", "local_vmmrdb_manifest.csv"])
                for img_entry in ("images/000/000000.jpg", "images/000/000001.png"):
                    with zf_repacked.open(img_entry) as imf:
                        with Image.open(imf) as im:
                            self.assertEqual(im.size, (64, 48))


if __name__ == "__main__":
    unittest.main()
