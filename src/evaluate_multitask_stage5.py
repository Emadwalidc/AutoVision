"""Evaluate fine-tuned multi-task AutoVision outputs vs Stage 2/3A baselines and verify ONNX."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, f1_score, silhouette_score

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_attributes_manifest import BODY_TYPE_CLASSES, _plot_confusion_matrix  # noqa: E402
from evaluate_features_baseline import (  # noqa: E402
    align_bundle_with_manifest,
    eval_cosine_knn,
    eval_few_shot_retrieval,
    eval_nearest_prototype,
    load_feature_bundle,
    load_manifest_lookup,
    plot_retrieval_montage,
    plot_tsne_models,
    plot_year_regression,
)
from multitask_model import (  # noqa: E402
    COLOR_CLASSES,
    COLOR_GROUPS_11,
    YEAR_MAX,
    YEAR_MIN,
    AutoVisionMultiTaskNet,
    AutoVisionOnnxWrapper,
)


def _to_float(val: str | None) -> float | None:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    return float(s)


def load_baseline_reference(stage2_csv: Path, stage3a_csv: Path) -> dict[str, dict[str, float | None]]:
    """Loads frozen baseline reference numbers from Stage 2 and Stage 3A CSV reports."""
    ref: dict[str, dict[str, float | None]] = {}

    if stage2_csv.is_file():
        s2_map: dict[tuple[str, str, str], dict[str, float | None]] = {}
        with open(stage2_csv, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                key = (row["backbone"], row["task"], row["method"])
                s2_map[key] = {
                    "top1": _to_float(row.get("top1")),
                    "top5": _to_float(row.get("top5")),
                    "macro_f1": _to_float(row.get("macro_f1")),
                    "mae_years": _to_float(row.get("mae_years")),
                    "within_2yr": _to_float(row.get("within_2yr")),
                }
        aliases = {
            "openclip_make_lp": ("openclip_vitb16", "Make", "Linear Probe"),
            "openclip_make_knn": ("openclip_vitb16", "Make", "Cosine kNN (k=5)"),
            "fusion_make_lp": ("fusion_openclip_dinov2", "Make", "Linear Probe"),
            "openclip_model_proto": ("openclip_vitb16", "Model", "Nearest Prototype"),
            "openclip_model_knn": ("openclip_vitb16", "Model", "Cosine kNN (k=5)"),
            "openclip_model_lp": ("openclip_vitb16", "Model", "Linear Probe"),
            "fusion_model_lp": ("fusion_openclip_dinov2", "Model", "Linear Probe"),
            "openclip_few_shot": ("openclip_vitb16", "Few-Shot Model", "1-Shot Prototype"),
            "openclip_year_ridge": ("openclip_vitb16", "Year", "Ridge (alpha=0.1)"),
            "fusion_year_ridge": ("fusion_openclip_dinov2", "Year", "Ridge (alpha=0.1)"),
        }
        for name, k in aliases.items():
            if k in s2_map:
                ref[name] = s2_map[k]

    if stage3a_csv.is_file():
        s3_map: dict[tuple[str, str], dict[str, float | None]] = {}
        with open(stage3a_csv, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                key = (row["backbone"], row["method"])
                s3_map[key] = {
                    "top1": _to_float(row.get("top1")),
                    "top3": _to_float(row.get("top3")),
                    "macro_f1": _to_float(row.get("macro_f1")),
                }
        for name, k in {
            "openclip_body_lp": ("openclip_vitb16", "Linear Probe"),
            "openclip_body_knn": ("openclip_vitb16", "Cosine kNN (k=5)"),
            "fusion_body_lp": ("fusion_openclip_dinov2", "Linear Probe"),
        }.items():
            if k in s3_map:
                ref[name] = s3_map[k]

    return ref


def _topk_metrics(scores: np.ndarray, y_true: np.ndarray, num_classes: int, k_second: int) -> tuple[dict[str, float], np.ndarray]:
    k_eff = min(k_second, scores.shape[1])
    topk = np.argsort(-scores, axis=1)[:, :k_eff]
    y_pred = topk[:, 0]
    top1 = float(np.mean(y_pred == y_true))
    topk_acc = float(np.mean([yt in row for yt, row in zip(y_true, topk)]))
    macro_f1 = float(f1_score(y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0))
    return {"top1": top1, f"top{k_second}": topk_acc, "macro_f1": macro_f1}, y_pred


def compute_vmmrdb_detailed_metrics(preds_npz: Path, label_maps: dict[str, Any]) -> dict[str, Any]:
    """Computes full Make, Model, Body Type, and Year metrics from saved test predictions."""
    with np.load(preds_npz, allow_pickle=False) as d:
        make_probs = d["make_probs"].astype(np.float32)
        model_logits = d["model_logits"].astype(np.float32)
        body_probs = d["body_probs"].astype(np.float32)
        year_exp = d["year_expected"].astype(np.float32)
        year_std = d["year_std"].astype(np.float32) if "year_std" in d else np.zeros_like(year_exp)
        make_y = d["make_targets"].astype(np.int64)
        model_y = d["model_targets"].astype(np.int64)
        body_y = d["body_targets"].astype(np.int64)
        year_y = d["year_targets"].astype(np.float32)
        year_valid = d["year_valid"].astype(bool)

    make_res, _ = _topk_metrics(make_probs, make_y, make_probs.shape[1], 5)
    model_res, model_pred = _topk_metrics(model_logits, model_y, model_logits.shape[1], 5)
    body_res, body_pred = _topk_metrics(body_probs, body_y, body_probs.shape[1], 3)

    # Per-model F1 breakdown
    model_names = label_maps.get("model_classes") or [f"model_{i}" for i in range(model_logits.shape[1])]
    n_models = model_logits.shape[1]
    per_model_f1 = f1_score(model_y, model_pred, labels=list(range(n_models)), average=None, zero_division=0)
    model_sup = np.bincount(model_y, minlength=n_models)
    min_sup = 10 if int((model_sup >= 10).sum()) >= 5 else 1
    eligible = [
        {"class": model_names[i], "support": int(model_sup[i]), "f1": round(float(per_model_f1[i]), 4)}
        for i in range(n_models)
        if model_sup[i] >= min_sup
    ]
    best_models = sorted(eligible, key=lambda r: (-r["f1"], -r["support"], r["class"]))[:10]
    hardest_models = sorted(eligible, key=lambda r: (r["f1"], -r["support"], r["class"]))[:10]

    # Per-body-type breakdown
    body_names = label_maps.get("body_classes") or list(BODY_TYPE_CLASSES)
    n_bodies = body_probs.shape[1]
    body_f1s = f1_score(body_y, body_pred, labels=list(range(n_bodies)), average=None, zero_division=0)
    per_class_body = []
    for i in range(n_bodies):
        mask_b = body_y == i
        sup = int(mask_b.sum())
        rec = float((body_pred[mask_b] == i).mean()) if sup > 0 else 0.0
        per_class_body.append(
            {
                "body_type": body_names[i] if i < len(body_names) else f"body_{i}",
                "support": sup,
                "recall": round(rec, 4),
                "f1": round(float(body_f1s[i]), 4),
            }
        )

    # Masked year evaluation
    yr_true_v = year_y[year_valid]
    yr_pred_v = year_exp[year_valid]
    yr_abs = np.abs(yr_pred_v - yr_true_v) if len(yr_true_v) > 0 else np.zeros(1, dtype=np.float32)
    yr_sq = (yr_pred_v - yr_true_v) ** 2 if len(yr_true_v) > 0 else np.zeros(1, dtype=np.float32)
    year_res = {
        "n_year_valid": int(year_valid.sum()),
        "mae": round(float(yr_abs.mean()), 4),
        "rmse": round(float(math.sqrt(float(yr_sq.mean()))), 4),
        "within_1yr": round(float((yr_abs <= 1.0).mean()), 4),
        "within_2yr": round(float((yr_abs <= 2.0).mean()), 4),
        "within_3yr": round(float((yr_abs <= 3.0).mean()), 4),
        "mean_pred_std": round(float(year_std[year_valid].mean()), 4) if int(year_valid.sum()) > 0 else 0.0,
    }

    return {
        "n_eval": int(len(model_y)),
        "make": {k: round(v, 4) for k, v in make_res.items()},
        "model": {k: round(v, 4) for k, v in model_res.items()},
        "body": {k: round(v, 4) for k, v in body_res.items()},
        "year": year_res,
        "best_models": best_models,
        "hardest_models": hardest_models,
        "per_class_body": per_class_body,
        "_arrays": {
            "body_y_true": body_y,
            "body_y_pred": body_pred,
            "year_y_true": yr_true_v,
            "year_y_pred": yr_pred_v,
        },
    }


def compute_color_detailed_metrics(vcor_npz: Path, label_maps: dict[str, Any]) -> dict[str, Any]:
    """Computes 15-class and 12-group color metrics + per-color recall table."""
    with np.load(vcor_npz, allow_pickle=False) as d:
        probs = d["probs"].astype(np.float32)
        preds = d["preds"].astype(np.int64) if "preds" in d else probs.argmax(axis=1)
        targets = d["targets"].astype(np.int64)

    color_names = label_maps.get("color_classes") or list(COLOR_CLASSES)
    color_groups = label_maps.get("color_groups_11") or COLOR_GROUPS_11
    n_colors = probs.shape[1]

    k3 = min(3, n_colors)
    top3_idx = np.argsort(-probs, axis=1)[:, :k3]
    top1 = float(np.mean(preds == targets))
    top3 = float(np.mean([yt in row for yt, row in zip(targets, top3_idx)]))
    macro_f1 = float(f1_score(targets, preds, labels=list(range(n_colors)), average="macro", zero_division=0))

    g_true = [color_groups.get(color_names[int(y)], color_names[int(y)]) for y in targets]
    g_pred = [color_groups.get(color_names[int(p)], color_names[int(p)]) for p in preds]
    group11_top1 = float(np.mean([a == b for a, b in zip(g_true, g_pred)]))

    f1_per_c = f1_score(targets, preds, labels=list(range(n_colors)), average=None, zero_division=0)
    per_class_color = []
    for i in range(n_colors):
        cname = color_names[i] if i < len(color_names) else f"color_{i}"
        m = targets == i
        sup = int(m.sum())
        rec = float((preds[m] == i).mean()) if sup > 0 else 0.0
        per_class_color.append(
            {
                "color": cname,
                "group11": color_groups.get(cname, cname),
                "support": sup,
                "recall": round(rec, 4),
                "f1": round(float(f1_per_c[i]), 4),
            }
        )

    return {
        "n_eval": int(len(targets)),
        "top1": round(top1, 4),
        "top3": round(top3, 4),
        "macro_f1": round(macro_f1, 4),
        "group11_top1": round(group11_top1, 4),
        "per_class_color": per_class_color,
        "_arrays": {"y_true": targets, "y_pred": preds},
    }


def plot_training_curves(
    history_csv: Path,
    vcor_frozen_lp_top1: float | None,
    out_path: Path,
) -> bool:
    """Plots Phase 1 (Make/Model/Body/Year) and Phase 2 (VCoR Color) training curves."""
    if not history_csv.is_file():
        return False

    with open(history_csv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return False

    p1 = [r for r in rows if r.get("phase") in ("phase1_warmup", "phase1_finetune")]
    p2 = [r for r in rows if r.get("phase") == "phase2_color"]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), dpi=150)

    # Panel (0, 0): Phase 1 Loss + Model / Make Top-1
    ax00 = axes[0, 0]
    if p1:
        x1 = np.arange(len(p1))
        labels1 = [f"W{r['epoch']}" if r["phase"] == "phase1_warmup" else f"E{r['epoch']}" for r in p1]
        losses = [_to_float(r.get("train_loss")) or 0.0 for r in p1]
        md_acc = [(_to_float(r.get("model_top1")) or 0.0) * 100.0 for r in p1]
        mk_acc = [(_to_float(r.get("make_top1")) or 0.0) * 100.0 for r in p1]

        l_loss = ax00.plot(x1, losses, "o--", color="#d62728", linewidth=1.8, label="Train Loss")
        ax00.set_ylabel("Multi-Task Train Loss", color="#d62728")
        ax00_r = ax00.twinx()
        l_md = ax00_r.plot(x1, md_acc, "s-", color="#1f77b4", linewidth=2.0, label="Model Val Top-1 (%)")
        l_mk = ax00_r.plot(x1, mk_acc, "^-", color="#2ca02c", linewidth=2.0, label="Make Val Top-1 (%)")
        ax00_r.set_ylabel("Validation Top-1 (%)")
        ax00.set_xticks(x1)
        ax00.set_xticklabels(labels1, fontsize=8)
        lines = l_loss + l_md + l_mk
        ax00.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)
        ax00.grid(True, alpha=0.25)
    else:
        ax00.text(0.5, 0.5, "Phase 1 history unavailable", ha="center", va="center")
    ax00.set_title("Phase 1: Loss & Model / Make Accuracy", fontsize=11, fontweight="bold")

    # Panel (0, 1): Phase 1 Body Type
    ax01 = axes[0, 1]
    if p1:
        x1 = np.arange(len(p1))
        labels1 = [f"W{r['epoch']}" if r["phase"] == "phase1_warmup" else f"E{r['epoch']}" for r in p1]
        bd_t1 = [(_to_float(r.get("body_top1")) or 0.0) * 100.0 for r in p1]
        bd_f1 = [(_to_float(r.get("body_macro_f1")) or 0.0) * 100.0 for r in p1]
        ax01.plot(x1, bd_t1, "o-", color="#1f77b4", linewidth=2.0, label="Body Top-1 (%)")
        ax01.plot(x1, bd_f1, "s-", color="#ff7f0e", linewidth=2.0, label="Body Macro-F1 (%)")
        ax01.set_xticks(x1)
        ax01.set_xticklabels(labels1, fontsize=8)
        ax01.set_ylabel("Percentage (%)")
        ax01.legend(loc="lower right", fontsize=8.5)
        ax01.grid(True, alpha=0.25)
    else:
        ax01.text(0.5, 0.5, "Phase 1 history unavailable", ha="center", va="center")
    ax01.set_title("Phase 1: Body Type (8 Classes)", fontsize=11, fontweight="bold")

    # Panel (1, 0): Phase 1 Year MAE & Within ±2yr
    ax10 = axes[1, 0]
    if p1:
        x1 = np.arange(len(p1))
        labels1 = [f"W{r['epoch']}" if r["phase"] == "phase1_warmup" else f"E{r['epoch']}" for r in p1]
        yr_mae = [_to_float(r.get("year_mae")) or 0.0 for r in p1]
        yr_w2 = [(_to_float(r.get("year_within2")) or 0.0) * 100.0 for r in p1]
        l_mae = ax10.plot(x1, yr_mae, "o-", color="#9467bd", linewidth=2.0, label="Year MAE (yr)")
        ax10.set_ylabel("MAE (years)", color="#9467bd")
        ax10_r = ax10.twinx()
        l_w2 = ax10_r.plot(x1, yr_w2, "s--", color="#17becf", linewidth=2.0, label="Within ±2 yr (%)")
        ax10_r.set_ylabel("Within ±2 yr (%)", color="#17becf")
        ax10.set_xticks(x1)
        ax10.set_xticklabels(labels1, fontsize=8)
        lines = l_mae + l_w2
        ax10.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)
        ax10.grid(True, alpha=0.25)
    else:
        ax10.text(0.5, 0.5, "Phase 1 history unavailable", ha="center", va="center")
    ax10.set_title("Phase 1: Year Estimation (DLDL)", fontsize=11, fontweight="bold")

    # Panel (1, 1): Phase 2 VCoR Color
    ax11 = axes[1, 1]
    if p2:
        x2 = [int(float(r["epoch"])) for r in p2]
        c_t1 = [(_to_float(r.get("color_val_top1")) or 0.0) * 100.0 for r in p2]
        c_f1 = [(_to_float(r.get("color_val_macro_f1")) or 0.0) * 100.0 for r in p2]
        c_g11 = [(_to_float(r.get("color_val_group11")) or 0.0) * 100.0 for r in p2]
        ax11.plot(x2, c_t1, "o-", color="#1f77b4", linewidth=2.0, label="15-Color Top-1 (%)")
        ax11.plot(x2, c_f1, "s-", color="#2ca02c", linewidth=2.0, label="15-Color Macro-F1 (%)")
        ax11.plot(x2, c_g11, "^--", color="#ff7f0e", linewidth=1.8, label="12-Group Top-1 (%)")
        if vcor_frozen_lp_top1 is not None:
            ax11.axhline(
                vcor_frozen_lp_top1 * 100.0,
                color="#7f7f7f",
                linestyle=":",
                linewidth=1.8,
                label=f"Frozen z_id LP Test ({vcor_frozen_lp_top1*100:.1f}%)",
            )
        ax11.set_xticks(x2)
        ax11.set_xlabel("Phase 2 Epoch")
        ax11.set_ylabel("Percentage (%)")
        ax11.legend(loc="lower right", fontsize=8)
        ax11.grid(True, alpha=0.25)
    else:
        ax11.text(0.5, 0.5, "Phase 2 Color not present in history", ha="center", va="center", color="#555555")
    ax11.set_title("Phase 2: VCoR Color Head (Frozen ViT)", fontsize=11, fontweight="bold")

    fig.suptitle("AutoVision Stage 4 Training Dynamics", fontsize=13, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_before_after_bars(comparison_rows: list[dict[str, Any]], out_path: Path) -> bool:
    """Plots grouped percentage bar chart comparing Frozen baselines vs Fine-Tuned AutoVision."""
    chart_targets = [
        ("Make", "Top-1 (Head/LP)", "Make\nTop-1"),
        ("Model", "Nearest Prototype Top-1", "Model Proto\nTop-1"),
        ("Model", "Top-1 (Head/LP)", "Model Head\nTop-1"),
        ("Model", "Macro-F1", "Model\nMacro-F1"),
        ("Body Type", "Macro-F1", "Body\nMacro-F1"),
        ("Year", "Within ±2 yr (%)", "Year\n±2 yr"),
        ("Few-Shot Model", "1-Shot Prototype Top-1", "Few-Shot\n1-Shot"),
    ]
    lookup = {(r["task"], r["metric"]): r for r in comparison_rows}
    selected = [(lbl, lookup[(t, m)]) for t, m, lbl in chart_targets if (t, m) in lookup]
    if not selected:
        return False

    labels = [s[0] for s in selected]
    vals_clip = [(s[1].get("frozen_openclip") or 0.0) * 100.0 for s in selected]
    vals_fuse = [(s[1].get("frozen_fusion") or 0.0) * 100.0 for s in selected]
    vals_ft = [(s[1].get("finetuned_autovision") or 0.0) * 100.0 for s in selected]

    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(12.0, 5.8), dpi=150)

    bars1 = ax.bar(x - w, vals_clip, w, label="Frozen OpenCLIP ViT-B/16 (512-D)", color="#7ea6e0", edgecolor="#333333", linewidth=0.5)
    bars2 = ax.bar(x, vals_fuse, w, label="Frozen Fusion CLIP+DINOv2 (1280-D)", color="#b3b3b3", edgecolor="#333333", linewidth=0.5)
    bars3 = ax.bar(x + w, vals_ft, w, label="Fine-Tuned AutoVision (512-D)", color="#2ca02c", edgecolor="#333333", linewidth=0.5)

    for bars in (bars1, bars2, bars3):
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width() / 2.0, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)")
    ax.set_title("Before vs After Fine-Tuning across All Vehicle Recognition Tasks (Test Set)", fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_tsne_before_after(
    X_frozen_test: np.ndarray | None,
    X_ft_test: np.ndarray,
    meta_test: list[dict[str, Any]],
    out_path: Path,
    seed: int = 42,
) -> tuple[float, float]:
    """Plots side-by-side 2D t-SNE on aligned top-10 test classes (Frozen vs Fine-Tuned)."""
    if X_frozen_test is None:
        sil_ft = plot_tsne_models(X_ft_test, meta_test, "finetuned_openclip_vitb16", out_path, seed=seed)
        return float("nan"), sil_ft

    counts = Counter(r["canonical_class"] for r in meta_test)
    top_classes = [cls for cls, _ in counts.most_common(10)]
    cls_set = set(top_classes)
    idxs = [i for i, r in enumerate(meta_test) if r["canonical_class"] in cls_set]

    rng = np.random.default_rng(seed)
    if len(idxs) > 1500:
        idxs = sorted(rng.choice(idxs, size=1500, replace=False).tolist())

    cls_to_id = {c: idx for idx, c in enumerate(top_classes)}
    y_sub = np.array([cls_to_id[meta_test[i]["canonical_class"]] for i in idxs])
    n_pts = len(idxs)
    perp = min(30.0, max(1.0, float(n_pts - 1) / 4.0))

    coords_list = []
    sils = []
    for X_space in (X_frozen_test, X_ft_test):
        X_sub = X_space[idxs]
        pca_dim = min(50, X_sub.shape[1], max(n_pts - 1, 2))
        X_pca = PCA(n_components=pca_dim, random_state=seed).fit_transform(X_sub) if pca_dim >= 2 and X_sub.shape[1] > pca_dim else X_sub
        coords = TSNE(n_components=2, perplexity=perp, random_state=seed, init="pca", learning_rate="auto").fit_transform(X_pca)
        sil = float(silhouette_score(X_sub, y_sub)) if len(set(y_sub)) >= 2 and n_pts > len(set(y_sub)) else 0.0
        coords_list.append(coords)
        sils.append(sil)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14.5, 6.2), dpi=150)
    cmap = plt.get_cmap("tab10")
    handles = []
    for cid, cls_name in enumerate(top_classes):
        m = y_sub == cid
        if np.any(m):
            sc = ax_l.scatter(coords_list[0][m, 0], coords_list[0][m, 1], s=16, alpha=0.78, color=cmap(cid % 10), label=cls_name)
            ax_r.scatter(coords_list[1][m, 0], coords_list[1][m, 1], s=16, alpha=0.78, color=cmap(cid % 10))
            handles.append(sc)

    ax_l.set_title(f"Before: Frozen OpenCLIP (Sil={sils[0]:.3f})", fontsize=11, fontweight="bold")
    ax_r.set_title(f"After: Fine-Tuned AutoVision (Sil={sils[1]:.3f})", fontsize=11, fontweight="bold")
    for ax in (ax_l, ax_r):
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"2D t-SNE Representation Before vs After Multi-Task Fine-Tuning (Top-{len(top_classes)} Test Models, N={n_pts})",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    fig.legend(handles, [h.get_label() for h in handles], bbox_to_anchor=(0.99, 0.88), loc="upper right", fontsize=8.5, frameon=True)
    fig.subplots_adjust(left=0.055, right=0.81, top=0.87, bottom=0.10, wspace=0.22)
    fig.savefig(out_path)
    plt.close(fig)
    return sils[0], sils[1]


def plot_color_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    top1: float,
    macro_f1: float,
    out_path: Path,
) -> None:
    """Plots 15x15 row-normalized confusion matrix for vehicle color classification."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, 100.0 * cm / np.maximum(row_sums, 1), 0.0)

    fig, ax = plt.subplots(figsize=(11.5, 9.6), dpi=150)
    im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=100.0)
    fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03, label="Recall (%)")

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=38, ha="right", fontsize=8.5)
    ax.set_yticklabels(class_names, fontsize=8.5)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_pct[i, j]
            cnt = cm[i, j]
            if val >= 1.5 or i == j:
                txt_color = "white" if val >= 55.0 else "#111111"
                ax.text(j, i, f"{val:.1f}%\n({cnt})", ha="center", va="center", color=txt_color, fontsize=6.5)

    ax.set_title(
        f"VCoR Color Confusion Matrix (SpatialAttn | Top-1={top1*100:.2f}%, Macro-F1={macro_f1*100:.2f}%)",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("True Color")
    ax.set_xlabel("Predicted Color")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def verify_or_export_onnx(
    models_dir: Path,
    label_maps: dict[str, Any],
    export_if_missing: bool = False,
    run_benchmark: bool = True,
    benchmark_runs: int = 5,
) -> dict[str, Any]:
    """Checks or exports ONNX, benchmarks CPU latency, and verifies PyTorch vs ONNX parity."""
    pt_path = models_dir / "autovision_multitask_best.pt"
    onnx_path = models_dir / "autovision_multitask.onnx"
    info: dict[str, Any] = {
        "pt_exists": pt_path.is_file(),
        "onnx_exists": onnx_path.is_file(),
        "onnx_exported_locally": False,
        "pt_cpu_ms": None,
        "onnx_cpu_ms": None,
        "max_abs_diff": None,
        "notes": [],
    }

    pt_core: AutoVisionMultiTaskNet | None = None
    if pt_path.is_file():
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        pt_core = AutoVisionMultiTaskNet(
            num_makes=len(label_maps.get("make_classes", [])) or 22,
            num_models=len(label_maps.get("model_classes", [])) or 225,
            num_bodies=len(label_maps.get("body_classes", [])) or 8,
            num_years=int(label_maps.get("year_max", YEAR_MAX)) - int(label_maps.get("year_min", YEAR_MIN)) + 1,
            num_colors=len(label_maps.get("color_classes", [])) or len(COLOR_CLASSES),
            pretrained=None,
        ).eval()
        pt_core.load_state_dict(state, strict=True)

        if export_if_missing and not onnx_path.is_file():
            try:
                import onnx  # noqa: F401

                wrapper = AutoVisionOnnxWrapper(pt_core).eval()
                dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)
                torch.onnx.export(
                    wrapper,
                    dummy,
                    str(onnx_path),
                    input_names=["image"],
                    output_names=["embedding", "make_logits", "model_logits", "body_logits", "year_logits", "color_logits"],
                    dynamic_axes={
                        k: {0: "batch"}
                        for k in ("image", "embedding", "make_logits", "model_logits", "body_logits", "year_logits", "color_logits")
                    },
                    opset_version=17,
                    do_constant_folding=True,
                )
                info["onnx_exists"] = onnx_path.is_file()
                info["onnx_exported_locally"] = info["onnx_exists"]
            except Exception as exc:
                info["notes"].append(f"ONNX local export skipped: {exc}")

    if not run_benchmark:
        return info

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)
    x_t = torch.from_numpy(x_np)
    pt_outputs: list[np.ndarray] | None = None

    if pt_core is not None:
        wrapper = AutoVisionOnnxWrapper(pt_core).eval()
        with torch.no_grad():
            pt_outputs = [t.numpy() for t in wrapper(x_t)]
            pt_times = []
            for _ in range(max(benchmark_runs, 1)):
                t0 = time.perf_counter()
                _ = wrapper(x_t)
                pt_times.append((time.perf_counter() - t0) * 1000.0)
        info["pt_cpu_ms"] = round(float(statistics.median(pt_times)), 2)

    if onnx_path.is_file():
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            inp_name = sess.get_inputs()[0].name
            onnx_outs = sess.run(None, {inp_name: x_np})
            ort_times = []
            for _ in range(max(benchmark_runs, 1)):
                t0 = time.perf_counter()
                _ = sess.run(None, {inp_name: x_np})
                ort_times.append((time.perf_counter() - t0) * 1000.0)
            info["onnx_cpu_ms"] = round(float(statistics.median(ort_times)), 2)
            if pt_outputs is not None and len(pt_outputs) == len(onnx_outs):
                max_diff = max(float(np.max(np.abs(a - b))) for a, b in zip(pt_outputs, onnx_outs))
                info["max_abs_diff"] = round(max_diff, 6)
        except ImportError:
            info["notes"].append("onnxruntime not installed locally; ONNX CPU benchmark skipped.")
        except Exception as exc:
            info["notes"].append(f"ONNX Runtime verification failed: {exc}")

    return info


def _delta_str(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return ""
    return f"{a - b:+.4f}"


def evaluate_stage5(
    models_dir: Path,
    features_dir: Path,
    manifest_path: Path,
    data_dir: Path,
    reports_dir: Path,
    stage2_csv: Path,
    stage3a_csv: Path,
    export_onnx_if_missing: bool = False,
    run_benchmark: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Runs complete Stage 5 Before-vs-After comparison, plotting, and ONNX checks."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_baseline_reference(stage2_csv, stage3a_csv)
    label_maps: dict[str, Any] = {}
    lmap_path = models_dir / "label_maps.json"
    if lmap_path.is_file():
        label_maps = json.loads(lmap_path.read_text(encoding="utf-8"))

    train_summary: dict[str, Any] = {}
    sum_path = models_dir / "training_summary.json"
    if sum_path.is_file():
        train_summary = json.loads(sum_path.read_text(encoding="utf-8"))

    vmmrdb_res: dict[str, Any] | None = None
    vmmrdb_npz = models_dir / "vmmrdb_test_predictions.npz"
    if vmmrdb_npz.is_file():
        vmmrdb_res = compute_vmmrdb_detailed_metrics(vmmrdb_npz, label_maps)

    color_res: dict[str, Any] | None = None
    vcor_npz = models_dir / "vcor_test_predictions.npz"
    if vcor_npz.is_file():
        color_res = compute_color_detailed_metrics(vcor_npz, label_maps)

    # Evaluate fine-tuned 512-D embedding space (Prototype, Cosine kNN, Few-Shot, t-SNE)
    ft_npz = models_dir / "features_finetuned_openclip_vitb16.npz"
    ft_proto_m: dict[str, float] | None = None
    ft_knn_m: dict[str, float] | None = None
    ft_few_m: dict[str, Any] | None = train_summary.get("few_shot_metrics")
    sil_frozen = float("nan")
    sil_ft = float("nan")
    montage_created = False

    if ft_npz.is_file() and manifest_path.is_file():
        manifest_lookup = load_manifest_lookup(manifest_path)
        X_ft_all, meta_ft_all = align_bundle_with_manifest(load_feature_bundle(ft_npz), manifest_lookup)

        tr_idx = [i for i, r in enumerate(meta_ft_all) if r["split"] == "train"]
        te_idx = [i for i, r in enumerate(meta_ft_all) if r["split"] == "test"]
        if tr_idx and te_idx:
            classes = sorted({meta_ft_all[i]["canonical_class"] for i in tr_idx})
            cls_to_id = {c: idx for idx, c in enumerate(classes)}
            X_tr_ft = X_ft_all[tr_idx]
            X_te_ft = X_ft_all[te_idx]
            y_tr_ft = np.array([cls_to_id[meta_ft_all[i]["canonical_class"]] for i in tr_idx], dtype=np.int64)
            y_te_ft = np.array([cls_to_id.get(meta_ft_all[i]["canonical_class"], -1) for i in te_idx], dtype=np.int64)

            valid_te = y_te_ft >= 0
            if np.any(valid_te):
                ft_proto_m, _ = eval_nearest_prototype(X_tr_ft, y_tr_ft, X_te_ft[valid_te], y_te_ft[valid_te], len(classes))
                ft_knn_m = eval_cosine_knn(X_tr_ft, y_tr_ft, X_te_ft[valid_te], y_te_ft[valid_te], len(classes), k=5)

            ft_few_m = eval_few_shot_retrieval(X_ft_all, meta_ft_all)

            # Side-by-side t-SNE using strictly aligned test rel_paths
            frozen_clip_npz = features_dir / "features_openclip_vitb16.npz"
            X_frozen_te = None
            X_ft_tsne = X_te_ft
            meta_ft_tsne = [meta_ft_all[i] for i in te_idx]
            if frozen_clip_npz.is_file():
                X_fr_all, meta_fr_all = align_bundle_with_manifest(load_feature_bundle(frozen_clip_npz), manifest_lookup)
                fr_te_map = {
                    meta_fr_all[i]["rel_path"]: i
                    for i in range(len(meta_fr_all))
                    if meta_fr_all[i]["split"] == "test"
                }
                aligned_fr, aligned_ft, aligned_meta = [], [], []
                for pos, gi in enumerate(te_idx):
                    rp = meta_ft_all[gi]["rel_path"]
                    if rp in fr_te_map:
                        aligned_fr.append(X_fr_all[fr_te_map[rp]])
                        aligned_ft.append(X_te_ft[pos])
                        aligned_meta.append(meta_ft_all[gi])
                if len(aligned_meta) >= 2:
                    X_frozen_te = np.vstack(aligned_fr)
                    X_ft_tsne = np.vstack(aligned_ft)
                    meta_ft_tsne = aligned_meta

            if len(meta_ft_tsne) >= 4:
                sil_frozen, sil_ft = plot_tsne_before_after(
                    X_frozen_te,
                    X_ft_tsne,
                    meta_ft_tsne,
                    figures_dir / "stage5_tsne_before_after.png",
                    seed=seed,
                )

            montage_created = plot_retrieval_montage(
                X_tr_ft,
                [meta_ft_all[i] for i in tr_idx],
                X_te_ft,
                [meta_ft_all[i] for i in te_idx],
                data_dir,
                "finetuned_openclip_vitb16",
                figures_dir / "stage5_retrieval_montage_finetuned.png",
                seed=seed,
            )

    # Plots from prediction archives & history
    vcor_lp_ref = train_summary.get("vcor_summary", {}).get("frozen_zid_linear_probe", {})
    curves_created = plot_training_curves(
        models_dir / "training_history.csv",
        vcor_lp_ref.get("top1"),
        figures_dir / "stage5_training_curves.png",
    )

    if vmmrdb_res is not None:
        arr = vmmrdb_res["_arrays"]
        body_names = label_maps.get("body_classes") or list(BODY_TYPE_CLASSES)
        _plot_confusion_matrix(
            y_true=arr["body_y_true"],
            y_pred=arr["body_y_pred"],
            class_names=body_names,
            backbone_name="Fine-Tuned AutoVision",
            top1=vmmrdb_res["body"]["top1"],
            macro_f1=vmmrdb_res["body"]["macro_f1"],
            out_path=figures_dir / "stage5_body_type_confusion_matrix.png",
        )
        if len(arr["year_y_true"]) > 0:
            plot_year_regression(
                arr["year_y_true"],
                arr["year_y_pred"],
                vmmrdb_res["year"]["mae"],
                "Fine-Tuned AutoVision DLDL",
                figures_dir / "stage5_year_dldl_evaluation.png",
            )

    if color_res is not None:
        carr = color_res["_arrays"]
        color_names = label_maps.get("color_classes") or list(COLOR_CLASSES)
        plot_color_confusion_matrix(
            y_true=carr["y_true"],
            y_pred=carr["y_pred"],
            class_names=color_names,
            top1=color_res["top1"],
            macro_f1=color_res["macro_f1"],
            out_path=figures_dir / "stage5_color_confusion_matrix.png",
        )

    onnx_info = verify_or_export_onnx(
        models_dir,
        label_maps,
        export_if_missing=export_onnx_if_missing,
        run_benchmark=run_benchmark,
    )

    # Build unified Before-vs-After comparison table
    comparison_rows: list[dict[str, Any]] = []

    def _add_row(task: str, metric: str, f_clip: float | None, f_fuse: float | None, ft_val: float | None, notes: str = "") -> None:
        comparison_rows.append(
            {
                "task": task,
                "metric": metric,
                "frozen_openclip": round(f_clip, 4) if f_clip is not None else "",
                "frozen_fusion": round(f_fuse, 4) if f_fuse is not None else "",
                "finetuned_autovision": round(ft_val, 4) if ft_val is not None else "",
                "delta_vs_openclip": _delta_str(ft_val, f_clip),
                "delta_vs_fusion": _delta_str(ft_val, f_fuse),
                "notes": notes,
            }
        )

    if vmmrdb_res is not None:
        _add_row(
            "Make",
            "Top-1 (Head/LP)",
            baseline.get("openclip_make_lp", {}).get("top1"),
            baseline.get("fusion_make_lp", {}).get("top1"),
            vmmrdb_res["make"]["top1"],
        )
        _add_row(
            "Make",
            "Macro-F1",
            baseline.get("openclip_make_lp", {}).get("macro_f1"),
            baseline.get("fusion_make_lp", {}).get("macro_f1"),
            vmmrdb_res["make"]["macro_f1"],
        )
        _add_row(
            "Model",
            "Top-1 (Head/LP)",
            baseline.get("openclip_model_lp", {}).get("top1"),
            baseline.get("fusion_model_lp", {}).get("top1"),
            vmmrdb_res["model"]["top1"],
            "CosFace Head",
        )
        _add_row(
            "Model",
            "Top-5 (Head/LP)",
            baseline.get("openclip_model_lp", {}).get("top5"),
            baseline.get("fusion_model_lp", {}).get("top5"),
            vmmrdb_res["model"]["top5"],
        )
        _add_row(
            "Model",
            "Macro-F1",
            baseline.get("openclip_model_lp", {}).get("macro_f1"),
            baseline.get("fusion_model_lp", {}).get("macro_f1"),
            vmmrdb_res["model"]["macro_f1"],
        )

    if ft_proto_m is not None:
        _add_row(
            "Model",
            "Nearest Prototype Top-1",
            baseline.get("openclip_model_proto", {}).get("top1"),
            None,
            ft_proto_m["top1"],
        )
    if ft_knn_m is not None:
        _add_row(
            "Model",
            "Cosine kNN (k=5) Top-1",
            baseline.get("openclip_model_knn", {}).get("top1"),
            None,
            ft_knn_m["top1"],
        )

    if ft_few_m is not None and ft_few_m.get("few_shot_top1") is not None:
        _add_row(
            "Few-Shot Model",
            "1-Shot Prototype Top-1",
            baseline.get("openclip_few_shot", {}).get("top1"),
            None,
            ft_few_m.get("few_shot_top1"),
            f"queries={ft_few_m.get('few_shot_queries', '')}",
        )
        _add_row(
            "Few-Shot Model",
            "1-Shot Prototype Top-5",
            baseline.get("openclip_few_shot", {}).get("top5"),
            None,
            ft_few_m.get("few_shot_top5"),
            f"queries={ft_few_m.get('few_shot_queries', '')}",
        )

    if vmmrdb_res is not None:
        _add_row(
            "Body Type",
            "Top-1 (Head/LP)",
            baseline.get("openclip_body_lp", {}).get("top1"),
            baseline.get("fusion_body_lp", {}).get("top1"),
            vmmrdb_res["body"]["top1"],
        )
        _add_row(
            "Body Type",
            "Macro-F1",
            baseline.get("openclip_body_lp", {}).get("macro_f1"),
            baseline.get("fusion_body_lp", {}).get("macro_f1"),
            vmmrdb_res["body"]["macro_f1"],
            f"Cosine kNN baseline={baseline.get('openclip_body_knn', {}).get('macro_f1', '')}",
        )
        _add_row(
            "Year",
            "MAE (years, valid only)",
            baseline.get("openclip_year_ridge", {}).get("mae_years"),
            baseline.get("fusion_year_ridge", {}).get("mae_years"),
            vmmrdb_res["year"]["mae"],
            "Lower is better",
        )
        _add_row(
            "Year",
            "Within ±2 yr (%)",
            baseline.get("openclip_year_ridge", {}).get("within_2yr"),
            baseline.get("fusion_year_ridge", {}).get("within_2yr"),
            vmmrdb_res["year"]["within_2yr"],
        )

    if color_res is not None:
        _add_row("Color", "15-Class Top-1", vcor_lp_ref.get("top1"), None, color_res["top1"], "VCoR test")
        _add_row("Color", "15-Class Macro-F1", vcor_lp_ref.get("macro_f1"), None, color_res["macro_f1"], "VCoR test")
        _add_row("Color", "12-Group Automotive Top-1", vcor_lp_ref.get("group11_top1"), None, color_res["group11_top1"], "VCoR test")

    if not math.isnan(sil_ft):
        _add_row(
            "Representation",
            "2D t-SNE Silhouette",
            sil_frozen if not math.isnan(sil_frozen) else None,
            None,
            sil_ft,
            "Top-10 test classes",
        )

    csv_out = reports_dir / "stage5_before_after_metrics.csv"
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "metric",
                "frozen_openclip",
                "frozen_fusion",
                "finetuned_autovision",
                "delta_vs_openclip",
                "delta_vs_fusion",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    bars_created = plot_before_after_bars(comparison_rows, figures_dir / "stage5_before_after_bars.png")

    md_out = reports_dir / "stage5_evaluation_report.md"
    _write_stage5_markdown_report(
        md_out=md_out,
        comparison_rows=comparison_rows,
        vmmrdb_res=vmmrdb_res,
        color_res=color_res,
        onnx_info=onnx_info,
        curves_created=curves_created,
        bars_created=bars_created,
        sil_ft=sil_ft,
        montage_created=montage_created,
    )

    print(f"[Stage 5] Saved {csv_out}")
    print(f"[Stage 5] Saved {md_out}")
    return {
        "metrics_csv": csv_out,
        "report_md": md_out,
        "comparison_rows": comparison_rows,
        "vmmrdb_metrics": {k: v for k, v in vmmrdb_res.items() if k != "_arrays"} if vmmrdb_res else None,
        "color_metrics": {k: v for k, v in color_res.items() if k != "_arrays"} if color_res else None,
        "onnx_info": onnx_info,
    }


def _fmt_pct_or_num(val: Any, task: str, metric: str) -> str:
    if val == "" or val is None:
        return "—"
    v = float(val)
    if task in ("Year", "Representation") and "MAE" in metric or task == "Representation":
        return f"{v:.4f}" if task == "Representation" else f"{v:.2f} yr"
    return f"{v * 100.0:.2f}%"


def _write_stage5_markdown_report(
    md_out: Path,
    comparison_rows: list[dict[str, Any]],
    vmmrdb_res: dict[str, Any] | None,
    color_res: dict[str, Any] | None,
    onnx_info: dict[str, Any],
    curves_created: bool,
    bars_created: bool,
    sil_ft: float,
    montage_created: bool,
) -> None:
    lines = [
        "# AutoVision — Этап 5: Сравнение «До / После Fine-Tuning» и верификация ONNX\n",
        "## 1. Сводная таблица «До (Frozen Baseline) vs После (Multi-Task Fine-Tuning)»\n",
        "| Задача | Метрика | Frozen OpenCLIP (512-D) | Frozen Fusion (1280-D) | Fine-Tuned AutoVision (512-D) | Прирост vs OpenCLIP | Примечание |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | :--- |",
    ]
    for r in comparison_rows:
        t, m = r["task"], r["metric"]
        c_s = _fmt_pct_or_num(r["frozen_openclip"], t, m)
        f_s = _fmt_pct_or_num(r["frozen_fusion"], t, m)
        a_s = _fmt_pct_or_num(r["finetuned_autovision"], t, m)
        d_s = r["delta_vs_openclip"] or "—"
        if d_s != "—" and not ("MAE" in m or t == "Representation"):
            d_s = f"{float(d_s) * 100.0:+.2f}%"
        lines.append(f"| {t} | {m} | {c_s} | {f_s} | **{a_s}** | {d_s} | {r['notes']} |")

    if vmmrdb_res is not None:
        lines.extend(
            [
                "\n## 2. Детализация по типам кузова (`Body Type`, Test)\n",
                "| Тип кузова | Support | Recall | F1 |",
                "| :--- | ---: | ---: | ---: |",
            ]
        )
        for b in vmmrdb_res["per_class_body"]:
            lines.append(f"| `{b['body_type']}` | {b['support']} | {b['recall']*100:.2f}% | {b['f1']*100:.2f}% |")

        lines.extend(
            [
                "\n## 3. Лучшие и наиболее сложные модели по F1\n",
                "**Топ-10 лучших моделей:**\n",
                "| Модель | Support | F1 |",
                "| :--- | ---: | ---: |",
            ]
        )
        for m in vmmrdb_res["best_models"]:
            lines.append(f"| `{m['class']}` | {m['support']} | {m['f1']*100:.2f}% |")

        lines.extend(["\n**10 наиболее сложных моделей:**\n", "| Модель | Support | F1 |", "| :--- | ---: | ---: |"])
        for m in vmmrdb_res["hardest_models"]:
            lines.append(f"| `{m['class']}` | {m['support']} | {m['f1']*100:.2f}% |")

    if color_res is not None:
        lines.extend(
            [
                "\n## 4. Детализация по цветам (`VCoR Test`, 15 классов)\n",
                "| Цвет | Группа | Support | Recall | F1 |",
                "| :--- | :--- | ---: | ---: | ---: |",
            ]
        )
        for c in color_res["per_class_color"]:
            lines.append(
                f"| `{c['color']}` | `{c['group11']}` | {c['support']} | {c['recall']*100:.2f}% | {c['f1']*100:.2f}% |"
            )

    lines.extend(
        [
            "\n## 5. Проверка инференса CPU и паритета ONNX\n",
            f"- Чекпоинт `.pt` найден: `{onnx_info['pt_exists']}` (CPU медиана latency batch=1: `{onnx_info['pt_cpu_ms']} мс`)",
            f"- Модель `.onnx` найдена: `{onnx_info['onnx_exists']}` (CPU медиана latency batch=1: `{onnx_info['onnx_cpu_ms']} мс`)",
            f"- Максимальное расхождение выходов PyTorch vs ONNX (`max_abs_diff`): `{onnx_info['max_abs_diff']}`",
        ]
    )
    for note in onnx_info.get("notes", []):
        lines.append(f"- Примечание: {note}")

    figs = []
    if curves_created:
        figs.append("figures/stage5_training_curves.png")
    if bars_created:
        figs.append("figures/stage5_before_after_bars.png")
    if not math.isnan(sil_ft):
        figs.append("figures/stage5_tsne_before_after.png")
    if vmmrdb_res is not None:
        figs.extend(["figures/stage5_body_type_confusion_matrix.png", "figures/stage5_year_dldl_evaluation.png"])
    if color_res is not None:
        figs.append("figures/stage5_color_confusion_matrix.png")
    if montage_created:
        figs.append("figures/stage5_retrieval_montage_finetuned.png")

    lines.append("\n## 6. Сформированные графики\n")
    for fg in figs:
        lines.append(f"- `{fg}`")

    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 5 Before-vs-After evaluation and ONNX check.")
    parser.add_argument("--models-dir", type=Path, default=BASE_DIR / "models")
    parser.add_argument("--features-dir", type=Path, default=BASE_DIR / "features")
    parser.add_argument("--manifest", type=Path, default=BASE_DIR / "data_manifests" / "local_vmmrdb_multitask_manifest.csv")
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\Projects\bitirme\data"))
    parser.add_argument("--reports-dir", type=Path, default=BASE_DIR / "reports")
    parser.add_argument("--stage2-csv", type=Path, default=BASE_DIR / "reports" / "feature_extraction_metrics.csv")
    parser.add_argument("--stage3a-csv", type=Path, default=BASE_DIR / "reports" / "body_type_baseline_metrics.csv")
    parser.add_argument("--export-onnx-if-missing", action="store_true")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_stage5(
        models_dir=args.models_dir,
        features_dir=args.features_dir,
        manifest_path=args.manifest,
        data_dir=args.data_dir,
        reports_dir=args.reports_dir,
        stage2_csv=args.stage2_csv,
        stage3a_csv=args.stage3a_csv,
        export_onnx_if_missing=args.export_onnx_if_missing,
        run_benchmark=not args.no_benchmark,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
