"""Unit and end-to-end tests for Stage 5 evaluation and Before-vs-After reporting."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_attributes_manifest import BODY_TYPE_CLASSES  # noqa: E402
from evaluate_multitask_stage5 import evaluate_stage5, load_baseline_reference  # noqa: E402
from extract_features import save_feature_bundle  # noqa: E402
from multitask_model import COLOR_CLASSES, COLOR_GROUPS_11, YEAR_MAX, YEAR_MIN  # noqa: E402


def _make_synthetic_stage5_inputs(root: Path, include_optional: bool = True) -> dict[str, Path]:
    models_dir = root / "models"
    features_dir = root / "features"
    reports_dir = root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    makes = ["mazda", "nissan", "toyota"]
    models = ["mazda__rx7", "mazda__tribute", "nissan__altima", "nissan__frontier", "toyota__camry", "toyota__rav4"]
    bodies = list(BODY_TYPE_CLASSES)
    rng = np.random.default_rng(123)

    manifest_path = root / "manifest.csv"
    all_rows = []
    rel_paths = []
    splits = []
    cls_list = []
    make_y, model_y, body_y, year_y, year_valid = [], [], [], [], []

    idx = 0
    for m_id, cls_name in enumerate(models):
        mk = cls_name.split("__")[0]
        bd_id = m_id % len(bodies)
        for sp, count in [("train", 12), ("test", 5)]:
            for k in range(count):
                yr = 2000 + ((k + m_id) % 15)
                rp = f"{mk}_{cls_name.split('__')[1]}_{yr}/img_{idx:03d}.jpg"
                all_rows.append(
                    {
                        "rel_path": rp,
                        "make": mk,
                        "model": cls_name.split("__")[1],
                        "canonical_class": cls_name,
                        "raw_year": str(yr),
                        "year_valid": "True",
                        "is_clean": "True",
                        "split": sp,
                    }
                )
                rel_paths.append(rp)
                splits.append(sp)
                cls_list.append(cls_name)
                make_y.append(makes.index(mk))
                model_y.append(m_id)
                body_y.append(bd_id)
                year_y.append(yr)
                year_valid.append(True)
                idx += 1

    # Add 2 few_shot classes (3 images each) -> 4 queries across 2 classes
    for fs_id, cls_name in enumerate(["porsche__911", "volvo__v70"]):
        mk = cls_name.split("__")[0]
        for k in range(3):
            rp = f"{mk}_{cls_name.split('__')[1]}_2008/fs_{fs_id}_{k}.jpg"
            all_rows.append(
                {
                    "rel_path": rp,
                    "make": mk,
                    "model": cls_name.split("__")[1],
                    "canonical_class": cls_name,
                    "raw_year": "2008",
                    "year_valid": "True",
                    "is_clean": "True",
                    "split": "few_shot",
                }
            )
            rel_paths.append(rp)
            splits.append("few_shot")
            cls_list.append(cls_name)
            idx += 1

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rel_path", "make", "model", "canonical_class", "raw_year", "year_valid", "is_clean", "split"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    label_maps = {
        "make_classes": makes,
        "model_classes": models,
        "body_classes": bodies,
        "color_classes": list(COLOR_CLASSES),
        "color_groups_11": COLOR_GROUPS_11,
        "year_min": YEAR_MIN,
        "year_max": YEAR_MAX,
    }
    (models_dir / "label_maps.json").write_text(json.dumps(label_maps), encoding="utf-8")

    # Training history CSV
    hist_fields = [
        "phase", "epoch", "train_loss", "model_top1", "make_top1", "body_top1",
        "body_macro_f1", "year_mae", "year_within2", "color_val_top1",
        "color_val_macro_f1", "color_val_group11",
    ]
    with open(models_dir / "training_history.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hist_fields)
        w.writeheader()
        w.writerow(
            {
                "phase": "phase1_warmup", "epoch": 1, "train_loss": 3.2, "model_top1": 0.55,
                "make_top1": 0.80, "body_top1": 0.82, "body_macro_f1": 0.72, "year_mae": 2.4, "year_within2": 0.55,
            }
        )
        for ep in (1, 2):
            w.writerow(
                {
                    "phase": "phase1_finetune", "epoch": ep, "train_loss": 1.8 - 0.5 * ep,
                    "model_top1": 0.85 + 0.05 * ep, "make_top1": 0.94 + 0.02 * ep,
                    "body_top1": 0.93 + 0.02 * ep, "body_macro_f1": 0.90 + 0.02 * ep,
                    "year_mae": 1.6 - 0.2 * ep, "year_within2": 0.72 + 0.04 * ep,
                }
            )
        if include_optional:
            for ep in (1, 2):
                w.writerow(
                    {
                        "phase": "phase2_color", "epoch": ep, "train_loss": 0.9 - 0.2 * ep,
                        "color_val_top1": 0.78 + 0.06 * ep, "color_val_macro_f1": 0.72 + 0.06 * ep,
                        "color_val_group11": 0.85 + 0.05 * ep,
                    }
                )

    (models_dir / "training_summary.json").write_text(
        json.dumps({"vcor_summary": {"frozen_zid_linear_probe": {"top1": 0.65, "macro_f1": 0.58, "group11_top1": 0.72}}}),
        encoding="utf-8",
    )

    # VMMRdb test predictions for 30 test samples (first 6 models * 5 test each)
    te_positions = [i for i, sp in enumerate(splits) if sp == "test"]
    n_te = len(te_positions)
    te_mk = np.array([makes.index(all_rows[i]["make"]) for i in te_positions], dtype=np.int64)
    te_md = np.array([models.index(all_rows[i]["canonical_class"]) for i in te_positions], dtype=np.int64)
    te_bd = np.array([models.index(all_rows[i]["canonical_class"]) % len(bodies) for i in te_positions], dtype=np.int64)
    te_yr = np.array([int(all_rows[i]["raw_year"]) for i in te_positions], dtype=np.int64)
    te_yv = np.ones(n_te, dtype=bool)
    te_yv[0] = False  # Masked corrupted year check

    make_probs = np.full((n_te, len(makes)), 0.05, dtype=np.float16)
    make_probs[np.arange(n_te), te_mk] = 0.90
    model_logits = rng.normal(0.0, 0.2, size=(n_te, len(models))).astype(np.float16)
    model_logits[np.arange(n_te), te_md] = 5.0
    body_probs = np.full((n_te, len(bodies)), 0.02, dtype=np.float16)
    body_probs[np.arange(n_te), te_bd] = 0.86
    year_exp = te_yr.astype(np.float32) + rng.uniform(-0.8, 0.8, size=n_te).astype(np.float32)

    np.savez(
        models_dir / "vmmrdb_test_predictions.npz",
        make_probs=make_probs,
        model_logits=model_logits,
        body_probs=body_probs,
        year_expected=year_exp,
        year_std=np.full(n_te, 1.1, dtype=np.float32),
        make_targets=te_mk,
        model_targets=te_md,
        body_targets=te_bd,
        year_targets=te_yr,
        year_valid=te_yv,
        rel_paths=np.asarray([rel_paths[i] for i in te_positions], dtype=str),
    )

    # Synthetic fine-tuned embeddings for all clean images
    n_all = len(rel_paths)
    ft_embs = rng.normal(size=(n_all, 32)).astype(np.float32)
    cls_centers = rng.normal(size=(len(models) + 2, 32)).astype(np.float32)
    unique_all_cls = sorted(set(cls_list))
    for i, c in enumerate(cls_list):
        ft_embs[i] = cls_centers[unique_all_cls.index(c)] + 0.1 * rng.normal(size=32)
    ft_embs /= np.linalg.norm(ft_embs, axis=1, keepdims=True)
    boxes = np.tile(np.array([0, 0, 100, 100, 0.95, 0.8], dtype=np.float32), (n_all, 1))
    used = np.ones(n_all, dtype=bool)
    save_feature_bundle(models_dir / "features_finetuned_openclip_vitb16.npz", ft_embs, boxes, used, rel_paths, {"dim": 32})

    if include_optional:
        fr_embs = rng.normal(size=(n_all, 32)).astype(np.float32)
        fr_embs /= np.linalg.norm(fr_embs, axis=1, keepdims=True)
        save_feature_bundle(features_dir / "features_openclip_vitb16.npz", fr_embs, boxes, used, rel_paths, {"dim": 32})

        n_c = len(COLOR_CLASSES) * 3
        c_targets = np.repeat(np.arange(len(COLOR_CLASSES)), 3)
        c_probs = np.full((n_c, len(COLOR_CLASSES)), 0.01, dtype=np.float16)
        c_probs[np.arange(n_c), c_targets] = 0.86
        np.savez(
            models_dir / "vcor_test_predictions.npz",
            probs=c_probs,
            preds=c_targets.astype(np.int64),
            targets=c_targets.astype(np.int64),
        )

    stage2_csv = reports_dir / "feature_extraction_metrics.csv"
    stage3a_csv = reports_dir / "body_type_baseline_metrics.csv"
    with open(stage2_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["backbone", "dim", "task", "method", "split", "n_eval", "top1", "top5", "macro_f1", "mae_years", "within_2yr"])
        w.writeheader()
        w.writerow({"backbone": "openclip_vitb16", "dim": 512, "task": "Make", "method": "Linear Probe", "split": "test", "n_eval": n_te, "top1": 0.9191, "top5": 0.994, "macro_f1": 0.8835})
        w.writerow({"backbone": "fusion_openclip_dinov2", "dim": 1280, "task": "Make", "method": "Linear Probe", "split": "test", "n_eval": n_te, "top1": 0.9381, "top5": 0.9963, "macro_f1": 0.9087})
        w.writerow({"backbone": "openclip_vitb16", "dim": 512, "task": "Model", "method": "Nearest Prototype", "split": "test", "n_eval": n_te, "top1": 0.6409, "top5": 0.8924, "macro_f1": 0.5579})
        w.writerow({"backbone": "openclip_vitb16", "dim": 512, "task": "Model", "method": "Cosine kNN (k=5)", "split": "test", "n_eval": n_te, "top1": 0.9078, "top5": 0.9767, "macro_f1": 0.7846})
        w.writerow({"backbone": "openclip_vitb16", "dim": 512, "task": "Model", "method": "Linear Probe", "split": "test", "n_eval": n_te, "top1": 0.9052, "top5": 0.9865, "macro_f1": 0.8058})
        w.writerow({"backbone": "fusion_openclip_dinov2", "dim": 1280, "task": "Model", "method": "Linear Probe", "split": "test", "n_eval": n_te, "top1": 0.9335, "top5": 0.9902, "macro_f1": 0.8484})
        w.writerow({"backbone": "openclip_vitb16", "dim": 512, "task": "Few-Shot Model", "method": "1-Shot Prototype", "split": "few_shot", "n_eval": 4, "top1": 0.3507, "top5": 0.5069})
        w.writerow({"backbone": "openclip_vitb16", "dim": 512, "task": "Year", "method": "Ridge (alpha=0.1)", "split": "test_year_valid", "n_eval": n_te, "mae_years": 2.027, "within_2yr": 0.5872})
        w.writerow({"backbone": "fusion_openclip_dinov2", "dim": 1280, "task": "Year", "method": "Ridge (alpha=0.1)", "split": "test_year_valid", "n_eval": n_te, "mae_years": 1.788, "within_2yr": 0.6485})

    with open(stage3a_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["backbone", "dim", "method", "split", "n_eval", "top1", "top3", "macro_f1"])
        w.writeheader()
        w.writerow({"backbone": "openclip_vitb16", "dim": 512, "method": "Linear Probe", "split": "test", "n_eval": n_te, "top1": 0.9235, "top3": 0.9955, "macro_f1": 0.876})
        w.writerow({"backbone": "openclip_vitb16", "dim": 512, "method": "Cosine kNN (k=5)", "split": "test", "n_eval": n_te, "top1": 0.9699, "top3": 0.9952, "macro_f1": 0.9466})
        w.writerow({"backbone": "fusion_openclip_dinov2", "dim": 1280, "method": "Linear Probe", "split": "test", "n_eval": n_te, "top1": 0.938, "top3": 0.9963, "macro_f1": 0.8989})

    return {
        "models_dir": models_dir,
        "features_dir": features_dir,
        "manifest_path": manifest_path,
        "reports_dir": reports_dir,
        "stage2_csv": stage2_csv,
        "stage3a_csv": stage3a_csv,
        "data_dir": root / "missing_raw_data",
    }


class TestStage5Evaluation(unittest.TestCase):
    def test_load_real_baseline_reference(self) -> None:
        s2 = BASE_DIR / "reports" / "feature_extraction_metrics.csv"
        s3 = BASE_DIR / "reports" / "body_type_baseline_metrics.csv"
        if not (s2.is_file() and s3.is_file()):
            self.skipTest("Stage 2 or Stage 3A baseline CSVs not found")

        ref = load_baseline_reference(s2, s3)
        self.assertAlmostEqual(ref["openclip_model_lp"]["top1"] or 0.0, 0.9052, places=4)
        self.assertAlmostEqual(ref["fusion_model_lp"]["top1"] or 0.0, 0.9335, places=4)
        self.assertAlmostEqual(ref["fusion_year_ridge"]["mae_years"] or 0.0, 1.788, places=3)
        self.assertAlmostEqual(ref["fusion_body_lp"]["macro_f1"] or 0.0, 0.8989, places=4)
        self.assertAlmostEqual(ref["openclip_few_shot"]["top1"] or 0.0, 0.3507, places=4)

    def test_stage5_full_synthetic_pipeline_generates_all_reports_and_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _make_synthetic_stage5_inputs(Path(tmp), include_optional=True)
            res = evaluate_stage5(
                models_dir=paths["models_dir"],
                features_dir=paths["features_dir"],
                manifest_path=paths["manifest_path"],
                data_dir=paths["data_dir"],
                reports_dir=paths["reports_dir"],
                stage2_csv=paths["stage2_csv"],
                stage3a_csv=paths["stage3a_csv"],
                export_onnx_if_missing=False,
                run_benchmark=False,
                seed=42,
            )

            self.assertTrue(res["metrics_csv"].is_file())
            self.assertTrue(res["report_md"].is_file())
            tasks_found = {r["task"] for r in res["comparison_rows"]}
            self.assertEqual(
                tasks_found,
                {"Make", "Model", "Few-Shot Model", "Body Type", "Year", "Color", "Representation"},
            )

            expected_figs = [
                "stage5_training_curves.png",
                "stage5_before_after_bars.png",
                "stage5_tsne_before_after.png",
                "stage5_body_type_confusion_matrix.png",
                "stage5_color_confusion_matrix.png",
                "stage5_year_dldl_evaluation.png",
            ]
            for name in expected_figs:
                fp = paths["reports_dir"] / "figures" / name
                self.assertTrue(fp.is_file(), f"Missing figure {name}")
                with Image.open(fp) as im:
                    self.assertGreaterEqual(im.size[0], 400)
                    self.assertGreaterEqual(im.size[1], 300)

    def test_stage5_graceful_degradation_without_optional_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _make_synthetic_stage5_inputs(Path(tmp), include_optional=False)
            res = evaluate_stage5(
                models_dir=paths["models_dir"],
                features_dir=paths["features_dir"],
                manifest_path=paths["manifest_path"],
                data_dir=paths["data_dir"],
                reports_dir=paths["reports_dir"],
                stage2_csv=paths["stage2_csv"],
                stage3a_csv=paths["stage3a_csv"],
                export_onnx_if_missing=False,
                run_benchmark=False,
                seed=42,
            )

            self.assertTrue(res["metrics_csv"].is_file())
            self.assertTrue(res["report_md"].is_file())
            self.assertIsNone(res["color_metrics"])
            for name in ("stage5_training_curves.png", "stage5_body_type_confusion_matrix.png", "stage5_year_dldl_evaluation.png"):
                self.assertTrue((paths["reports_dir"] / "figures" / name).is_file())


if __name__ == "__main__":
    unittest.main()
