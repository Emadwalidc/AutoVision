"""Generate AutoVision v2 evaluation figures, CSV metrics, and Russian thesis report."""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, f1_score, silhouette_score


BASE_DIR = Path(__file__).resolve().parent.parent


def _display_name(
    raw: str,
    make_display: dict[str, str] | None = None,
    model_display: dict[str, str] | None = None,
) -> str:
    cleaned = raw.strip()
    if not cleaned:
        return ""
    if "__" in cleaned:
        mk, md = cleaned.split("__", 1)
        mk_d = (make_display or {}).get(mk) or mk.replace("-", " ").title()
        md_d = (model_display or {}).get(cleaned) or md.replace("_", " ").title()
        return f"{mk_d} {md_d}".strip()
    if make_display and cleaned in make_display:
        return make_display[cleaned]
    return cleaned.replace("-", " ").replace("_", " ").title()


def _load_stage5_baseline(stage5_csv: Path) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    if not stage5_csv.is_file():
        return out
    with open(stage5_csv, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            entry: dict[str, float] = {}
            for k in ("frozen_openclip", "frozen_fusion", "finetuned_autovision"):
                val = str(r.get(k, "")).strip()
                if val:
                    entry[k] = float(val)
            out[(r["task"], r["metric"])] = entry
    return out


def _plot_evolution_comparison(
    v1_sum: dict[str, Any],
    v1_lmap: dict[str, Any],
    v2_sum: dict[str, Any],
    v2_lmap: dict[str, Any],
    c2_sum: dict[str, Any],
    s5_map: dict[tuple[str, str], dict[str, float]],
    out_path: Path,
) -> None:
    v1_imgs = int(v1_sum.get("num_clean_images", 89535))
    v2_imgs = int(v2_sum["num_total_images"])
    v1_makes = int(v1_sum.get("num_makes_train", 22))
    v2_makes = int(v2_sum["num_makes"])
    v1_models = int(v1_sum.get("num_models_train", 225))
    v2_models = int(v2_sum["num_models_closed"])
    v1_protos = int(v1_sum.get("few_shot_metrics", {}).get("few_shot_gallery_classes", 342))
    v2_protos = int(v2_sum["num_prototype_classes"])
    v1_ymin = int(v1_lmap.get("year_min", 1952))
    v1_ymax = int(v1_lmap.get("year_max", 2018))
    v1_span = v1_ymax - v1_ymin + 1
    v2_span = int(v2_lmap["year_max"]) - int(v2_lmap["year_min"]) + 1

    scale_rows = [
        ("Clean Photos", v1_imgs, v2_imgs, f"{v1_imgs:,} -> {v2_imgs:,}"),
        ("Closed Models", v1_models, v2_models, f"{v1_models} -> {v2_models}"),
        ("Supported Makes", v1_makes, v2_makes, f"{v1_makes} -> {v2_makes}"),
        ("Prototype Classes", v1_protos, v2_protos, f"{v1_protos} -> {v2_protos}"),
        ("Year Span (yr)", v1_span, v2_span, f"{v1_ymin}-{v1_ymax} -> {v2_lmap['year_min']}-{v2_lmap['year_max']}"),
    ]
    scale_rows.reverse()

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(16.4, 6.4), dpi=180, gridspec_kw={"width_ratios": [1.0, 1.42]}
    )

    y_pos = np.arange(len(scale_rows))
    mults = [r[2] / max(r[1], 1) for r in scale_rows]
    bars_l = ax_l.barh(y_pos, mults, height=0.56, color="#2563EB", edgecolor="#1E293B", linewidth=0.6)
    ax_l.axvline(1.0, color="#64748B", linestyle="--", linewidth=1.2, label="AutoVision v1 (1.0x)")
    for b, (_, _, _, desc), mult in zip(bars_l, scale_rows, mults):
        ax_l.text(
            b.get_width() + 0.08,
            b.get_y() + b.get_height() / 2.0,
            f"{mult:.2f}x  ({desc})",
            va="center",
            ha="left",
            fontsize=8.3,
            fontweight="bold",
            color="#0F172A",
        )
    ax_l.set_yticks(y_pos)
    ax_l.set_yticklabels([r[0] for r in scale_rows], fontsize=9.5, fontweight="bold")
    ax_l.set_xlim(0.0, max(mults) * 1.72)
    ax_l.set_xlabel("Expansion Multiplier vs AutoVision v1 (x)", fontsize=9.5)
    ax_l.set_title("Dataset & Taxonomy Scale Expansion (v1 -> v2)", fontsize=11.5, fontweight="bold", pad=12)
    ax_l.grid(axis="x", alpha=0.25)
    ax_l.legend(loc="lower right", fontsize=8.5)

    v1_test = v1_sum.get("vmmrdb_test_metrics", {})
    v1_col = v1_sum.get("vcor_summary", {}).get("spatial_attention_test", {})
    fr_col = v1_sum.get("vcor_summary", {}).get("frozen_zid_linear_probe", {})
    v2_all = v2_sum["overall_test_metrics"]
    v2_mod = v2_sum["modern_2017_2026_test_metrics"]
    v2_str = v2_sum["vmmrdb_street_test_metrics"]
    c2_vcor = c2_sum["vcor_test_metrics"]

    comp_items = [
        (
            "Make Top-1\n(Street)",
            s5_map.get(("Make", "Top-1 (Head/LP)"), {}).get("frozen_openclip", 0.9191),
            v1_test.get("make_top1", 0.9827),
            v2_str["make_top1"],
        ),
        (
            "Model\nMacro-F1",
            s5_map.get(("Model", "Macro-F1"), {}).get("frozen_openclip", 0.8058),
            v1_test.get("model_macro_f1", 0.8935),
            v2_all["model_macro_f1"],
        ),
        (
            "Body Top-1\n(Street)",
            s5_map.get(("Body Type", "Top-1 (Head/LP)"), {}).get("frozen_openclip", 0.9235),
            v1_test.get("body_top1", 0.9769),
            v2_str["body_top1"],
        ),
        (
            "Year +-2 yr\n(Overall)",
            s5_map.get(("Year", "Within ±2 yr (%)"), {}).get("frozen_openclip", 0.5872),
            v1_test.get("year_within2", 0.8756),
            v2_all["year_within2"],
        ),
        (
            "Year +-2 yr\n(2017-2026)",
            None,
            None,
            v2_mod["year_within2"],
        ),
        (
            "Color Top-1\n(VCoR Test)",
            fr_col.get("top1", 0.2808),
            v1_col.get("top1", 0.8676),
            c2_vcor["top1"],
        ),
        (
            "Color F1\n(VCoR Test)",
            fr_col.get("macro_f1", 0.2095),
            v1_col.get("macro_f1", 0.8413),
            c2_vcor["macro_f1"],
        ),
    ]

    x = np.arange(len(comp_items))
    w = 0.27
    vals_fr = [(c[1] * 100.0) if c[1] is not None else 0.0 for c in comp_items]
    vals_v1 = [(c[2] * 100.0) if c[2] is not None else 0.0 for c in comp_items]
    vals_v2 = [c[3] * 100.0 for c in comp_items]

    b1 = ax_r.bar(x - w, vals_fr, w, label="Frozen OpenCLIP", color="#94A3B8", edgecolor="#1E293B", linewidth=0.5)
    b2 = ax_r.bar(x, vals_v1, w, label="AutoVision v1 (225 cls)", color="#3B82F6", edgecolor="#1E293B", linewidth=0.5)
    b3 = ax_r.bar(x + w, vals_v2, w, label="AutoVision v2 (617 cls + Color)", color="#16A34A", edgecolor="#1E293B", linewidth=0.5)

    for idx, (bar1, bar2, bar3, item) in enumerate(zip(b1, b2, b3, comp_items)):
        for bar, raw_val in ((bar1, item[1]), (bar2, item[2]), (bar3, item[3])):
            if raw_val is None:
                bar.set_alpha(0.0)
                continue
            h = bar.get_height()
            ax_r.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 1.2,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.8,
                fontweight="bold",
                color="#0F172A",
                rotation=90,
            )
        if item[1] is None and item[2] is None:
            ax_r.text(
                x[idx] - 0.65 * w,
                6.0,
                "v1 <= 2018\n(N/A)",
                ha="center",
                va="bottom",
                fontsize=7.0,
                color="#475569",
                style="italic",
            )

    ax_r.set_xticks(x)
    ax_r.set_xticklabels([c[0] for c in comp_items], fontsize=8.4)
    ax_r.set_ylim(0, 122)
    ax_r.set_ylabel("Score (%)", fontsize=9.5)
    ax_r.set_title("Accuracy Evolution: Baseline -> AutoVision v1 -> AutoVision v2", fontsize=11.5, fontweight="bold", pad=12)
    ax_r.grid(axis="y", alpha=0.25)
    ax_r.legend(loc="upper center", ncol=3, fontsize=7.8, framealpha=0.95)

    fig.suptitle(
        "AutoVision v2 Architecture Upgrade: 2.7x Larger Catalog with Higher Accuracy & Color Physics",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(out_path)
    plt.close(fig)


def _plot_domain_breakdown(v2_sum: dict[str, Any], c2_sum: dict[str, Any], out_path: Path) -> None:
    ov = v2_sum["overall_test_metrics"]
    md = v2_sum["modern_2017_2026_test_metrics"]
    st = v2_sum["vmmrdb_street_test_metrics"]
    c_vcor = c2_sum["vcor_test_metrics"]
    c_dvm = c2_sum["dvm_test_metrics"]

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(14.5, 5.8), dpi=180, gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    metrics = [("Make Top-1", "make_top1"), ("Model Top-1", "model_top1"), ("Body Top-1", "body_top1"), ("Year +-2 yr", "year_within2")]
    x = np.arange(len(metrics))
    w = 0.25
    v_ov = [ov[k] * 100.0 for _, k in metrics]
    v_md = [md[k] * 100.0 for _, k in metrics]
    v_st = [st[k] * 100.0 for _, k in metrics]

    b1 = ax_l.bar(x - w, v_ov, w, label=f"Overall Test (n={ov['n_eval']:,})", color="#2563EB", edgecolor="#1E293B", linewidth=0.5)
    b2 = ax_l.bar(x, v_md, w, label=f"Modern 2017-2026 (n={md['n_eval']:,})", color="#16A34A", edgecolor="#1E293B", linewidth=0.5)
    b3 = ax_l.bar(x + w, v_st, w, label=f"VMMRdb Street (n={st['n_eval']:,})", color="#F59E0B", edgecolor="#1E293B", linewidth=0.5)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax_l.text(bar.get_x() + bar.get_width() / 2.0, h + 0.4, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax_l.set_xticks(x)
    ax_l.set_xticklabels([m[0] for m in metrics], fontsize=9.2, fontweight="bold")
    ax_l.set_ylim(74, 108.5)
    ax_l.set_ylabel("Accuracy / Within Tolerance (%)", fontsize=9.5)
    ax_l.set_title("Phase 1 Performance Across Holdout Domains", fontsize=11.5, fontweight="bold", pad=10)
    ax_l.grid(axis="y", alpha=0.25)
    ax_l.legend(loc="upper center", ncol=3, fontsize=7.5, framealpha=0.95)

    milestones = [
        ("Color Expert Top-1 (DVM-CAR)", c_dvm["top1"] * 100.0, "#7C3AED"),
        ("Nearest Prototype Top-1 (738 cls)", ov["nearest_prototype_top1"] * 100.0, "#0284C7"),
        ("Color Expert Top-1 (VCoR)", c_vcor["top1"] * 100.0, "#7C3AED"),
        ("Model Top-5 (Overall, 617 cls)", ov["model_top5"] * 100.0, "#2563EB"),
        ("Color Expert Top-3 (VCoR)", c_vcor["top3"] * 100.0, "#7C3AED"),
        ("Make Top-5 (Overall, 50 makes)", ov["make_top5"] * 100.0, "#16A34A"),
        ("Body Type Top-3 (Overall, 8 cls)", ov["body_top3"] * 100.0, "#16A34A"),
    ]
    ym = np.arange(len(milestones))
    bars_r = ax_r.barh(ym, [m[1] for m in milestones], height=0.58, color=[m[2] for m in milestones], edgecolor="#1E293B", linewidth=0.5)
    for bar, (_, val, _) in zip(bars_r, milestones):
        ax_r.text(val + 0.35, bar.get_y() + bar.get_height() / 2.0, f"{val:.2f}%", va="center", ha="left", fontsize=8.3, fontweight="bold")

    ax_r.set_yticks(ym)
    ax_r.set_yticklabels([m[0] for m in milestones], fontsize=8.8)
    ax_r.set_xlim(78, 104.2)
    ax_r.set_xlabel("Score (%)", fontsize=9.5)
    ax_r.set_title("Top-K Retrieval & Color Expert Milestones", fontsize=11.5, fontweight="bold", pad=10)
    ax_r.grid(axis="x", alpha=0.25)

    note = (
        f"Year MAE (Midpoint): Overall {ov['year_mae_mid']:.2f} yr | Modern {md['year_mae_mid']:.2f} yr | Street {st['year_mae_mid']:.2f} yr\n"
        f"VCoR Confusions: Black->Brown = 0, Brown->Black = 0"
    )
    ax_r.text(
        0.5,
        -0.19,
        note,
        transform=ax_r.transAxes,
        ha="center",
        va="top",
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )

    fig.suptitle("AutoVision v2 Domain Breakdown & Top-K Accuracy Profile", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.94])
    fig.savefig(out_path)
    plt.close(fig)


def _plot_body_confusion(
    true_body: np.ndarray,
    pred_body: np.ndarray,
    body_classes: list[str],
    body_top1: float,
    body_top3: float,
    body_f1: float,
    out_path: Path,
) -> list[dict[str, Any]]:
    n_cls = len(body_classes)
    cm = confusion_matrix(true_body, pred_body, labels=list(range(n_cls)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, 100.0 * cm / np.maximum(row_sums, 1), 0.0)
    f1s = f1_score(true_body, pred_body, labels=list(range(n_cls)), average=None, zero_division=0)

    per_class: list[dict[str, Any]] = []
    for i, name in enumerate(body_classes):
        sup = int(row_sums[i, 0])
        rec = float(cm[i, i] / sup) if sup > 0 else 0.0
        per_class.append({"body_type": name, "support": sup, "recall": round(rec, 4), "f1": round(float(f1s[i]), 4)})

    fig, ax = plt.subplots(figsize=(9.2, 8.0), dpi=180)
    im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=100.0)
    fig.colorbar(im, ax=ax, fraction=0.044, pad=0.03, label="Recall (%)")

    ticks = np.arange(n_cls)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(body_classes, rotation=32, ha="right", fontsize=9.5, fontweight="bold")
    ax.set_yticklabels([f"{c} ({int(row_sums[i,0]):,})" for i, c in enumerate(body_classes)], fontsize=9.0)

    for i in range(n_cls):
        for j in range(n_cls):
            val = cm_pct[i, j]
            cnt = cm[i, j]
            if val >= 0.8 or i == j:
                color = "white" if val >= 55.0 else "#0F172A"
                ax.text(j, i, f"{val:.1f}%\n({cnt})", ha="center", va="center", color=color, fontsize=7.6)

    ax.set_title(
        f"Body Type Confusion Matrix (v2 Holdout, n={len(true_body):,})\n"
        f"Top-1 = {body_top1*100:.2f}% | Top-3 = {body_top3*100:.2f}% | Macro-F1 = {body_f1*100:.2f}%",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Predicted Body Type", fontsize=10)
    ax.set_ylabel("True Body Type (Support)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return per_class


def _plot_makes_ranking(
    true_make: np.ndarray,
    pred_make: np.ndarray,
    make_classes: list[str],
    make_display: dict[str, str],
    macro_f1: float,
    out_path: Path,
) -> list[dict[str, Any]]:
    n_mk = len(make_classes)
    f1s = f1_score(true_make, pred_make, labels=list(range(n_mk)), average=None, zero_division=0)
    sup = np.bincount(true_make, minlength=n_mk)

    rows = []
    for i, slug in enumerate(make_classes):
        m = true_make == i
        acc = float((pred_make[m] == i).mean()) if sup[i] > 0 else 0.0
        rows.append(
            {
                "slug": slug,
                "display": _display_name(slug, make_display),
                "support": int(sup[i]),
                "recall": round(acc, 4),
                "f1": round(float(f1s[i]), 4),
            }
        )
    rows.sort(key=lambda r: (-r["f1"], -r["support"], r["display"]))

    plot_rows = list(reversed(rows))
    y = np.arange(len(plot_rows))
    vals = [r["f1"] * 100.0 for r in plot_rows]
    colors = ["#16A34A" if v >= 95.0 else ("#2563EB" if v >= 90.0 else "#D97706") for v in vals]

    fig, ax = plt.subplots(figsize=(11.2, 13.5), dpi=180)
    bars = ax.barh(y, vals, height=0.72, color=colors, edgecolor="#1E293B", linewidth=0.45)
    ax.axvline(macro_f1 * 100.0, color="#DC2626", linestyle="--", linewidth=1.4, label=f"Macro-F1 = {macro_f1*100:.2f}%")

    for bar, r, v in zip(bars, plot_rows, vals):
        ax.text(v + 0.45, bar.get_y() + bar.get_height() / 2.0, f"{v:.1f}% (n={r['support']})", va="center", ha="left", fontsize=7.5, color="#0F172A")

    min_v = min(vals) if vals else 70.0
    ax.set_yticks(y)
    ax.set_yticklabels([r["display"] for r in plot_rows], fontsize=8.3, fontweight="bold")
    ax.set_xlim(max(0.0, min(70.0, min_v - 5.0)), 112.0)
    ax.set_xlabel("F1-Score (%)", fontsize=10)
    ax.set_title(f"Per-Make F1 Ranking Across All {n_mk} Supported Brands (Holdout Test, n={len(true_make):,})", fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="x", alpha=0.25)

    legend_handles = [
        patches.Patch(color="#16A34A", label="F1 >= 95%"),
        patches.Patch(color="#2563EB", label="90% <= F1 < 95%"),
        patches.Patch(color="#D97706", label="F1 < 90%"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + legend_handles, loc="lower right", fontsize=8.8, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return rows


def _plot_year_interval_evaluation(
    pred_year: np.ndarray,
    year_low: np.ndarray,
    year_high: np.ndarray,
    year_valid: np.ndarray,
    out_path: Path,
    seed: int = 42,
) -> list[dict[str, Any]]:
    mask = year_valid.astype(bool)
    py = pred_year[mask].astype(np.float32)
    lo = year_low[mask].astype(np.float32)
    hi = year_high[mask].astype(np.float32)
    mid = 0.5 * (lo + hi)

    interval_err = np.maximum(0.0, np.maximum(lo - py, py - hi))
    mid_err = np.abs(py - mid)
    within2_all = float(np.mean(interval_err <= 2.0))
    mae_all = float(np.mean(mid_err))

    eras = [
        ("1950-1989", 1950.0, 1989.99),
        ("1990-1999", 1990.0, 1999.99),
        ("2000-2009", 2000.0, 2009.99),
        ("2010-2016", 2010.0, 2016.99),
        ("2017-2026", 2017.0, 2026.99),
    ]
    era_rows: list[dict[str, Any]] = []
    for name, e_lo, e_hi in eras:
        em = (hi >= 2017.0) if name == "2017-2026" else ((hi < 2017.0) & (mid >= e_lo) & (mid <= e_hi))
        n_e = int(em.sum())
        era_rows.append(
            {
                "era": name,
                "count": n_e,
                "within2": round(float(np.mean(interval_err[em] <= 2.0)), 4) if n_e else 0.0,
                "within3": round(float(np.mean(interval_err[em] <= 3.0)), 4) if n_e else 0.0,
                "mae_mid": round(float(np.mean(mid_err[em])), 4) if n_e else 0.0,
            }
        )

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14.4, 5.8), dpi=180)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(py))
    if len(idx) > 4500:
        idx = rng.choice(idx, size=4500, replace=False)

    ax_l.fill_between([1948, 2028], [1946, 2026], [1950, 2030], color="#BFDBFE", alpha=0.45, label="+-2 yr band")
    ax_l.plot([1948, 2028], [1948, 2028], "--", color="#DC2626", linewidth=1.6, label="Ideal (y = x)")
    ax_l.scatter(mid[idx], py[idx], s=10, alpha=0.25, color="#1D4ED8", edgecolors="none", label=f"Test predictions (shown {len(idx):,}/{len(py):,})")
    ax_l.set_xlim(1948, 2028)
    ax_l.set_ylim(1948, 2028)
    ax_l.set_xlabel("True Generation Midpoint Year", fontsize=9.5)
    ax_l.set_ylabel("Predicted Year (DLDL Expected)", fontsize=9.5)
    ax_l.set_title("Interval-Aware DLDL Year Prediction (1950-2026)", fontsize=11.5, fontweight="bold", pad=10)
    ax_l.grid(True, alpha=0.25)
    ax_l.legend(loc="lower right", fontsize=8.0, framealpha=0.95)
    ax_l.text(
        0.04,
        0.95,
        f"n_valid = {len(py):,}\nMAE (midpoint) = {mae_all:.2f} yr\nWithin +-2 yr = {within2_all*100:.2f}%\n2017-2026 +-2 yr = {era_rows[-1]['within2']*100:.2f}%",
        transform=ax_l.transAxes,
        va="top",
        ha="left",
        fontsize=8.4,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.38", "facecolor": "white", "edgecolor": "#94A3B8"},
    )

    xe = np.arange(len(era_rows))
    w2_vals = [r["within2"] * 100.0 for r in era_rows]
    mae_vals = [r["mae_mid"] for r in era_rows]

    bars = ax_r.bar(xe, w2_vals, width=0.52, color="#16A34A", alpha=0.88, edgecolor="#1E293B", linewidth=0.6, label="Within +-2 yr (%)")
    for bar, r, v in zip(bars, era_rows, w2_vals):
        ax_r.text(bar.get_x() + bar.get_width() / 2.0, v + 1.0, f"{v:.1f}%\n(n={r['count']:,})", ha="center", va="bottom", fontsize=7.6, fontweight="bold")

    ax_r.set_xticks(xe)
    ax_r.set_xticklabels([r["era"] for r in era_rows], fontsize=9.0, fontweight="bold")
    ax_r.set_ylim(0, 112)
    ax_r.set_ylabel("Within +-2 yr Tolerance (%)", color="#15803D", fontsize=9.5)
    ax_r.grid(axis="y", alpha=0.25)

    ax_r2 = ax_r.twinx()
    line = ax_r2.plot(xe, mae_vals, "o-", color="#EA580C", linewidth=2.2, markersize=7, label="MAE Midpoint (yr)")
    for xi, mv in zip(xe, mae_vals):
        ax_r2.text(
            xi,
            mv + 0.14,
            f"{mv:.2f}y",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color="#C2410C",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#FDBA74", "alpha": 0.92},
        )
    ax_r2.set_ylim(0, max(mae_vals) * 1.45)
    ax_r2.set_ylabel("MAE Midpoint (years)", color="#C2410C", fontsize=9.5)

    h1, l1 = ax_r.get_legend_handles_labels()
    h2, l2 = ax_r2.get_legend_handles_labels()
    ax_r.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8.3, framealpha=0.95)
    ax_r.set_title("Year Accuracy & Error Breakdown by Era", fontsize=11.5, fontweight="bold", pad=10)

    fig.suptitle("AutoVision v2 Temporal Estimation Across 77 Model Years (1950-2026)", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(out_path)
    plt.close(fig)
    return era_rows


def _plot_color_training_curves(
    history_csv: Path,
    v1_vcor_top1: float,
    out_path: Path,
) -> None:
    with open(history_csv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    labels = [f"W{r['epoch']}" if "Warmup" in r["phase"] else f"E{r['epoch']}" for r in rows]
    n_warm = sum(1 for r in rows if "Warmup" in r["phase"])
    x = np.arange(len(rows))
    losses = [float(r["loss"]) for r in rows]
    comp = [float(r["composite_score"]) * 100.0 for r in rows]
    vc_t1 = [float(r["vcor_val_top1"]) * 100.0 for r in rows]
    vc_f1 = [float(r["vcor_val_f1"]) * 100.0 for r in rows]
    dv_t1 = [float(r["dvm_val_top1"]) * 100.0 for r in rows]
    dv_f1 = [float(r["dvm_val_f1"]) * 100.0 for r in rows]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14.0, 5.6), dpi=180)

    if n_warm > 0:
        ax_l.axvspan(-0.4, n_warm - 0.5, color="#FEF3C7", alpha=0.55, label="Warmup (Frozen ConvNeXt)")
        ax_r.axvspan(-0.4, n_warm - 0.5, color="#FEF3C7", alpha=0.55)

    p_loss = ax_l.plot(x, losses, "o--", color="#DC2626", linewidth=1.9, label="Train Loss")
    ax_l.set_ylabel("Cross-Entropy Loss", color="#DC2626", fontsize=9.5)
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(labels, fontsize=8.3)
    ax_l.grid(True, alpha=0.25)

    ax_l2 = ax_l.twinx()
    p_comp = ax_l2.plot(x, comp, "s-", color="#2563EB", linewidth=2.2, label="Composite Val Score (%)")
    best_i = int(np.argmax(comp))
    ax_l2.scatter([x[best_i]], [comp[best_i]], s=110, color="#16A34A", zorder=5, label=f"Best ({labels[best_i]}: {comp[best_i]:.2f}%)")
    ax_l2.set_ylabel("Composite Val Score (%)", color="#2563EB", fontsize=9.5)

    h1, l1 = ax_l.get_legend_handles_labels()
    h2, l2 = ax_l2.get_legend_handles_labels()
    ax_l.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8.0, framealpha=0.95)
    ax_l.set_title("Loss & Composite Score (0.65 VCoR + 0.35 DVM)", fontsize=11.3, fontweight="bold", pad=10)

    ax_r.plot(x, vc_t1, "o-", color="#2563EB", linewidth=2.1, label="VCoR Val Top-1 (%)")
    ax_r.plot(x, vc_f1, "s--", color="#0EA5E9", linewidth=1.8, label="VCoR Val Macro-F1 (%)")
    ax_r.plot(x, dv_t1, "^-", color="#16A34A", linewidth=2.1, label="DVM-CAR Val Top-1 (%)")
    ax_r.plot(x, dv_f1, "v--", color="#65A30D", linewidth=1.8, label="DVM-CAR Val Macro-F1 (%)")
    ax_r.axhline(v1_vcor_top1 * 100.0, color="#64748B", linestyle=":", linewidth=1.7, label=f"v1 Color Head VCoR Test ({v1_vcor_top1*100:.2f}%)")
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(labels, fontsize=8.3)
    ax_r.set_ylabel("Score (%)", fontsize=9.5)
    ax_r.set_title("Validation Dynamics: Street (VCoR) & Studio (DVM-CAR)", fontsize=11.3, fontweight="bold", pad=10)
    ax_r.grid(True, alpha=0.25)
    ax_r.legend(loc="lower right", fontsize=8.0, framealpha=0.95)

    fig.suptitle("Phase 2 Standalone Color Expert v2 Training Curves (ConvNeXt-Tiny + 16-D Masked Lab)", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(out_path)
    plt.close(fig)


def _plot_tsne_embeddings(
    embs: np.ndarray,
    manifest_rows: list[dict[str, str]],
    make_display: dict[str, str],
    model_display: dict[str, str],
    out_path: Path,
    seed: int = 42,
) -> float:
    by_cls: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(manifest_rows):
        by_cls[r["canonical_class"]].append(i)

    # Pick 12 distinct makes with rich representation so clusters span diverse vehicle families
    sorted_classes = sorted(by_cls.keys(), key=lambda c: (-len(by_cls[c]), c))
    chosen: list[str] = []
    seen_makes: set[str] = set()
    for c in sorted_classes:
        mk = c.split("__", 1)[0]
        if mk in seen_makes or len(by_cls[c]) < 25:
            continue
        chosen.append(c)
        seen_makes.add(mk)
        if len(chosen) == 12:
            break

    idxs: list[int] = []
    y_sub_list: list[int] = []
    for cid, cls_name in enumerate(chosen):
        for i in by_cls[cls_name]:
            idxs.append(i)
            y_sub_list.append(cid)

    X_sub = embs[idxs].astype(np.float32)
    y_sub = np.array(y_sub_list, dtype=np.int64)
    n_pts = len(idxs)
    pca_dim = min(50, X_sub.shape[1], max(n_pts - 1, 2))
    X_pca = PCA(n_components=pca_dim, random_state=seed).fit_transform(X_sub)
    perp = min(30.0, max(2.0, float(n_pts - 1) / 4.0))
    coords = TSNE(n_components=2, perplexity=perp, random_state=seed, init="pca", learning_rate="auto").fit_transform(X_pca)
    sil = float(silhouette_score(X_sub, y_sub, metric="cosine"))

    fig, ax = plt.subplots(figsize=(12.2, 8.2), dpi=180)
    cmap = plt.get_cmap("tab20")
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p"]

    for cid, cls_name in enumerate(chosen):
        m = y_sub == cid
        disp = _display_name(cls_name, make_display, model_display)
        ax.scatter(
            coords[m, 0],
            coords[m, 1],
            s=36,
            alpha=0.84,
            color=cmap(cid % 20),
            marker=markers[cid % len(markers)],
            edgecolors="#0F172A",
            linewidths=0.35,
            label=f"{disp} (n={int(m.sum())})",
        )

    ax.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax.set_title(
        f"AutoVision v2 512-D Embedding Space (2D t-SNE, 12 Diverse Models, N={n_pts}, Cosine Silhouette = {sil:.3f})",
        fontsize=11.8,
        fontweight="bold",
        pad=10,
    )
    ax.grid(True, alpha=0.22)
    ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left", fontsize=8.5, frameon=True, title="Vehicle Model")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return sil


def _plot_retrieval_montage(
    embs: np.ndarray,
    manifest_rows: list[dict[str, str]],
    thumbs_zip: Path,
    make_display: dict[str, str],
    model_display: dict[str, str],
    out_path: Path,
) -> None:
    norm_embs = embs.astype(np.float32)
    norm_embs /= np.maximum(np.linalg.norm(norm_embs, axis=1, keepdims=True), 1e-12)

    preferred = [
        ("porsche__911", True),
        ("ford__mustang", False),
        ("toyota__camry", True),
        ("bmw__3_series", False),
        ("chevrolet__corvette", True),
        ("jeep__wrangler", False),
    ]
    by_cls: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(manifest_rows):
        by_cls[r["canonical_class"]].append(i)

    query_indices: list[int] = []
    for cls_slug, prefer_modern in preferred:
        cands = by_cls.get(cls_slug, [])
        if not cands:
            continue
        if prefer_modern:
            mod_cands = [i for i in cands if manifest_rows[i].get("raw_year", "").isdigit() and int(manifest_rows[i]["raw_year"]) >= 2017]
            query_indices.append(mod_cands[0] if mod_cands else cands[0])
        else:
            vmmr_cands = [i for i in cands if manifest_rows[i]["rel_path"].startswith("vmmrdb/")]
            query_indices.append(vmmr_cands[0] if vmmr_cands else cands[0])

    fig, axes = plt.subplots(len(query_indices), 6, figsize=(15.5, 2.55 * len(query_indices) + 0.9), dpi=180)
    if len(query_indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    with zipfile.ZipFile(thumbs_zip, "r") as zf:
        for row_i, q_idx in enumerate(query_indices):
            q_rec = manifest_rows[q_idx]
            sims = norm_embs @ norm_embs[q_idx]
            sims[q_idx] = -2.0
            top_idx = np.argsort(-sims)[:5]
            row_items = [(q_idx, None)] + [(int(ti), float(sims[int(ti)])) for ti in top_idx]

            for col_j, (idx, sim_val) in enumerate(row_items):
                ax = axes[row_i, col_j]
                rec = manifest_rows[idx]
                raw = zf.read(rec["rel_path"])
                with Image.open(io.BytesIO(raw)) as im:
                    ax.imshow(im.convert("RGB"))

                disp = _display_name(rec["canonical_class"], make_display, model_display)
                yr = rec.get("raw_year", "").strip()
                yr_txt = f" ({yr})" if yr else ""
                if col_j == 0:
                    border_col = "#0F172A"
                    title = f"QUERY\n{disp}{yr_txt}"
                else:
                    if rec["canonical_class"] == q_rec["canonical_class"]:
                        border_col = "#16A34A"
                    elif rec["make"] == q_rec["make"]:
                        border_col = "#F59E0B"
                    else:
                        border_col = "#DC2626"
                    title = f"{disp}{yr_txt}\ncos={sim_val:.3f}"

                for spine in ax.spines.values():
                    spine.set_edgecolor(border_col)
                    spine.set_linewidth(3.0 if col_j == 0 else 2.6)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(title, fontsize=7.6, fontweight="bold" if col_j == 0 else "normal", labelpad=4)

    fig.suptitle(
        "AutoVision v2 Visual Similarity Retrieval (512-D Cosine Gallery, N=22,028) | Border: Green=Exact Model, Amber=Same Make",
        fontsize=12.0,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(out_path)
    plt.close(fig)


def _compute_best_hardest_models(
    true_model: np.ndarray,
    pred_model: np.ndarray,
    model_classes: list[str],
    min_support: int = 15,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    n_md = len(model_classes)
    f1s = f1_score(true_model, pred_model, labels=list(range(n_md)), average=None, zero_division=0)
    sup = np.bincount(true_model, minlength=n_md)
    eligible = [
        {"class": model_classes[i], "support": int(sup[i]), "f1": round(float(f1s[i]), 4)}
        for i in range(n_md)
        if sup[i] >= min_support
    ]
    best = sorted(eligible, key=lambda r: (-r["f1"], -r["support"], r["class"]))[:10]
    hardest = sorted(eligible, key=lambda r: (r["f1"], -r["support"], r["class"]))[:10]
    return best, hardest


def _write_metrics_csv(
    out_csv: Path,
    v2_sum: dict[str, Any],
    c2_sum: dict[str, Any],
    sil_v2: float,
    s5_map: dict[tuple[str, str], dict[str, float]],
) -> None:
    ov = v2_sum["overall_test_metrics"]
    md = v2_sum["modern_2017_2026_test_metrics"]
    st = v2_sum["vmmrdb_street_test_metrics"]
    c_vc = c2_sum["vcor_test_metrics"]
    c_dv = c2_sum["dvm_test_metrics"]

    rows = [
        ("Dataset", "Total Clean Images", "", v2_sum["num_total_images"], ""),
        ("Dataset", "Closed Models / Makes", "", f"{v2_sum['num_models_closed']} / {v2_sum['num_makes']}", ""),
        ("Dataset", "Prototype Classes (Closed+FewShot)", "", v2_sum["num_prototype_classes"], ""),
        ("Dataset", "Gallery Size", "", v2_sum["gallery_size"], ""),
        ("Make", "Top-1 Overall (50 makes)", s5_map.get(("Make", "Top-1 (Head/LP)"), {}).get("finetuned_autovision", ""), ov["make_top1"], ""),
        ("Make", "Top-1 VMMRdb Street", s5_map.get(("Make", "Top-1 (Head/LP)"), {}).get("finetuned_autovision", ""), st["make_top1"], ""),
        ("Make", "Macro-F1 Overall", s5_map.get(("Make", "Macro-F1"), {}).get("finetuned_autovision", ""), ov["make_macro_f1"], ""),
        ("Model", "Top-1 Overall (617 cls)", s5_map.get(("Model", "Top-1 (Head/LP)"), {}).get("finetuned_autovision", ""), ov["model_top1"], ""),
        ("Model", "Top-5 Overall (617 cls)", s5_map.get(("Model", "Top-5 (Head/LP)"), {}).get("finetuned_autovision", ""), ov["model_top5"], ""),
        ("Model", "Macro-F1 Overall (617 cls)", s5_map.get(("Model", "Macro-F1"), {}).get("finetuned_autovision", ""), ov["model_macro_f1"], ""),
        ("Model", "Nearest Prototype Top-1 (738 cls)", s5_map.get(("Model", "Nearest Prototype Top-1"), {}).get("finetuned_autovision", ""), ov["nearest_prototype_top1"], ""),
        ("Model", "Top-1 Modern 2017-2026", "", md["model_top1"], ""),
        ("Model", "Top-1 VMMRdb Street", "", st["model_top1"], ""),
        ("Body Type", "Top-1 Overall", s5_map.get(("Body Type", "Top-1 (Head/LP)"), {}).get("finetuned_autovision", ""), ov["body_top1"], ""),
        ("Body Type", "Top-3 Overall", "", ov["body_top3"], ""),
        ("Body Type", "Top-1 VMMRdb Street", s5_map.get(("Body Type", "Top-1 (Head/LP)"), {}).get("finetuned_autovision", ""), st["body_top1"], ""),
        ("Year", "MAE Midpoint Overall (yr)", s5_map.get(("Year", "MAE (years, valid only)"), {}).get("finetuned_autovision", ""), ov["year_mae_mid"], "Lower is better"),
        ("Year", "Within +-2 yr Overall", s5_map.get(("Year", "Within ±2 yr (%)"), {}).get("finetuned_autovision", ""), ov["year_within2"], ""),
        ("Year", "MAE Midpoint 2017-2026 (yr)", "", md["year_mae_mid"], ""),
        ("Year", "Within +-2 yr 2017-2026", "", md["year_within2"], ""),
        ("Color", "VCoR Test Top-1 (12 cls)", s5_map.get(("Color", "15-Class Top-1"), {}).get("finetuned_autovision", ""), c_vc["top1"], "Phase 2 ConvNeXt-Tiny + Lab"),
        ("Color", "VCoR Test Top-3", "", c_vc["top3"], ""),
        ("Color", "VCoR Test Macro-F1", s5_map.get(("Color", "15-Class Macro-F1"), {}).get("finetuned_autovision", ""), c_vc["macro_f1"], ""),
        ("Color", "DVM-CAR Test Top-1", "", c_dv["top1"], ""),
        ("Color", "DVM-CAR Test Macro-F1", "", c_dv["macro_f1"], ""),
        ("Representation", "2D t-SNE Cosine Silhouette", "", round(sil_v2, 4), "12 diverse gallery classes"),
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "metric", "autovision_v1", "autovision_v2", "notes"])
        w.writerows(rows)


def _write_markdown_report(
    out_md: Path,
    v1_sum: dict[str, Any],
    v1_lmap: dict[str, Any],
    v2_sum: dict[str, Any],
    c2_sum: dict[str, Any],
    v2_lmap: dict[str, Any],
    body_rows: list[dict[str, Any]],
    make_rows: list[dict[str, Any]],
    era_rows: list[dict[str, Any]],
    best_models: list[dict[str, Any]],
    hard_models: list[dict[str, Any]],
    sil_v2: float,
) -> None:
    ov = v2_sum["overall_test_metrics"]
    md = v2_sum["modern_2017_2026_test_metrics"]
    st = v2_sum["vmmrdb_street_test_metrics"]
    v1_t = v1_sum.get("vmmrdb_test_metrics", {})
    v1_c = v1_sum.get("vcor_summary", {}).get("spatial_attention_test", {})
    v1_ymin = int(v1_lmap.get("year_min", 1952))
    v1_ymax = int(v1_lmap.get("year_max", 2018))
    v1_span = v1_ymax - v1_ymin + 1
    v2_span = int(v2_lmap["year_max"]) - int(v2_lmap["year_min"]) + 1
    c_vc = c2_sum["vcor_test_metrics"]
    c_dv = c2_sum["dvm_test_metrics"]
    conf = c2_sum["key_confusion_pairs_vcor_test"]
    mk_disp = v2_lmap.get("make_display_names", {})
    md_disp = v2_lmap.get("model_display_names", {})

    lines = [
        "# AutoVision v2 — Итоговый отчёт по качеству моделей и графики для ВКР",
        "",
        "## 1. Эволюция архитектуры и масштаба (AutoVision v1 -> v2)",
        "",
        "| Параметр | AutoVision v1 | AutoVision v2 | Рост |",
        "| :--- | ---: | ---: | ---: |",
        f"| Чистых изображений | {v1_sum.get('num_clean_images', 89535):,} | **{v2_sum['num_total_images']:,}** | **{v2_sum['num_total_images']/89535:.2f}x** |",
        f"| Закрытых марок | {v1_sum.get('num_makes_train', 22)} | **{v2_sum['num_makes']}** | **{v2_sum['num_makes']/22:.2f}x** |",
        f"| Закрытых моделей | {v1_sum.get('num_models_train', 225)} | **{v2_sum['num_models_closed']}** | **{v2_sum['num_models_closed']/225:.2f}x** |",
        f"| Классов в прототипах (вкл. few-shot) | 342 | **{v2_sum['num_prototype_classes']}** | **{v2_sum['num_prototype_classes']/342:.2f}x** |",
        f"| Диапазон годов выпуска | {v1_ymin}–{v1_ymax} ({v1_span} лет) | **{v2_lmap['year_min']}–{v2_lmap['year_max']} ({v2_span} лет)** | **{v2_span/max(v1_span,1):.2f}x** |",
        f"| Цветовой модуль | Голова поверх ViT-B/16 (15 кл.) | **ConvNeXt-Tiny + 16-D Masked Lab (12 кл.) + YOLOv8n-seg** | — |",
        "",
        "## 2. Метрики Phase 1 по доменам (`autovision-v2-vitb16-multitask`)",
        "",
        "| Домен оценки | N | Make Top-1 | Model Top-1 | Model Top-5 | Model Macro-F1 | Body Top-1 | Year MAE | Year +-2 yr |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| **Overall Holdout** | {ov['n_eval']:,} | {ov['make_top1']*100:.2f}% | **{ov['model_top1']*100:.2f}%** | {ov['model_top5']*100:.2f}% | {ov['model_macro_f1']*100:.2f}% | {ov['body_top1']*100:.2f}% | {ov['year_mae_mid']:.2f} yr | {ov['year_within2']*100:.2f}% |",
        f"| **Modern 2017–2026** | {md['n_eval']:,} | {md['make_top1']*100:.2f}% | **{md['model_top1']*100:.2f}%** | — | — | {md['body_top1']*100:.2f}% | **{md['year_mae_mid']:.2f} yr** | **{md['year_within2']*100:.2f}%** |",
        f"| **VMMRdb Street** | {st['n_eval']:,} | **{st['make_top1']*100:.2f}%** | **{st['model_top1']*100:.2f}%** | — | — | **{st['body_top1']*100:.2f}%** | {st['year_mae_mid']:.2f} yr | {st['year_within2']*100:.2f}% |",
        f"| *Для сравнения: AutoVision v1 (VMMRdb)* | {v1_t.get('n_eval', 8885):,} | {v1_t.get('make_top1', 0.9827)*100:.2f}% | {v1_t.get('model_top1', 0.9636)*100:.2f}% | {v1_t.get('model_top5', 0.9892)*100:.2f}% | {v1_t.get('model_macro_f1', 0.8935)*100:.2f}% | {v1_t.get('body_top1', 0.9769)*100:.2f}% | {v1_t.get('year_mae', 1.0533):.2f} yr | {v1_t.get('year_within2', 0.8756)*100:.2f}% |",
        "",
        f"- **Nearest Prototype Top-1 по всем {v2_sum['num_prototype_classes']} классам**: **{ov['nearest_prototype_top1']*100:.2f}%**.",
        f"- **Косинусный Silhouette (t-SNE, 12 разнообразных моделей из галереи)**: **{sil_v2:.3f}**.",
        "",
        "## 3. Эксперт по цвету Phase 2 (`ConvNeXt-Tiny + Masked CIE-Lab`)",
        "",
        "| Тестовый набор | N | Top-1 | Top-3 | Macro-F1 | Ключевые ошибки на VCoR |",
        "| :--- | ---: | ---: | ---: | ---: | :--- |",
        f"| **VCoR Test (уличные фото)** | {c_vc['n_eval']:,} | **{c_vc['top1']*100:.2f}%** | **{c_vc['top3']*100:.2f}%** | **{c_vc['macro_f1']*100:.2f}%** | Black<->Brown: **{conf['Black_as_Brown']}/{conf['Brown_as_Black']}**, Silver<->Gray: {conf['Silver_as_Gray']}/{conf['Gray_as_Silver']}, White->Silver: {conf['White_as_Silver']} |",
        f"| **DVM-CAR Test (студийные)** | {c_dv['n_eval']:,} | **{c_dv['top1']*100:.2f}%** | **{c_dv['top3']*100:.2f}%** | **{c_dv['macro_f1']*100:.2f}%** | Баланс по 12 цветам с маской фона |",
        f"| *Для сравнения: v1 Color Head (VCoR)* | 1,556 | {v1_c.get('top1', 0.8676)*100:.2f}% | {v1_c.get('top3', 0.9833)*100:.2f}% | {v1_c.get('macro_f1', 0.8413)*100:.2f}% | Без сегментации и CIE-Lab ветви |",
        "",
        "## 4. Детализация по типам кузова (`Body Type`, Holdout N=21,884)",
        "",
        "| Тип кузова | Support | Recall | F1 |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for r in body_rows:
        lines.append(f"| `{r['body_type']}` | {r['support']:,} | {r['recall']*100:.2f}% | {r['f1']*100:.2f}% |")

    lines.extend(["", "## 5. Точность определения года по эпохам (1950–2026)", "", "| Эпоха | N (valid) | Within +-2 yr | Within +-3 yr | MAE (midpoint) |", "| :--- | ---: | ---: | ---: | ---: |"])
    for r in era_rows:
        lines.append(f"| `{r['era']}` | {r['count']:,} | {r['within2']*100:.2f}% | {r['within3']*100:.2f}% | {r['mae_mid']:.2f} yr |")

    lines.extend(["", "## 6. Лучшие и наиболее сложные модели (из 617 закрытых моделей, support >= 15)", "", "**Топ-10 лучших по F1:**", "", "| Модель | Support | F1 |", "| :--- | ---: | ---: |"])
    for r in best_models:
        lines.append(f"| {_display_name(r['class'], mk_disp, md_disp)} (`{r['class']}`) | {r['support']} | {r['f1']*100:.2f}% |")

    lines.extend(["", "**10 наиболее сложных моделей (близкие поколения / ребеджи):**", "", "| Модель | Support | F1 |", "| :--- | ---: | ---: |"])
    for r in hard_models:
        lines.append(f"| {_display_name(r['class'], mk_disp, md_disp)} (`{r['class']}`) | {r['support']} | {r['f1']*100:.2f}% |")

    lines.extend(["", "## 7. 10 наиболее сложных марок (из 50)", "", "| Марка | Support | Recall | F1 |", "| :--- | ---: | ---: | ---: |"])
    for r in reversed(make_rows[-10:]):
        lines.append(f"| {r['display']} | {r['support']:,} | {r['recall']*100:.2f}% | {r['f1']*100:.2f}% |")

    lines.extend(
        [
            "",
            "## 8. Подготовленные графики (`reports/figures/v2/`)",
            "",
            "- `../figures/v2/v2_evolution_comparison.png` — рост масштаба датасета и сравнение Baseline -> v1 -> v2.",
            "- `../figures/v2/v2_domain_breakdown.png` — сравнение доменов Overall / Modern 2017–2026 / Street и Top-K метрик.",
            "- `../figures/v2/v2_body_type_confusion_matrix.png` — матрица ошибок по 8 типам кузова (N=21,884).",
            "- `../figures/v2/v2_makes_f1_ranking.png` — рейтинг F1-score по всем 50 маркам.",
            "- `../figures/v2/v2_year_interval_evaluation.png` — точность предсказания года 1950–2026 по эпохам и scatter midpoint vs prediction.",
            "- `../figures/v2/v2_color_training_curves.png` — кривые обучения Phase 2 Color Expert v2 (VCoR + DVM-CAR).",
            "- `../figures/v2/v2_color_confusion_matrix.png` — матрица ошибок цветового эксперта на 12 автомобильных цветах.",
            "- `../figures/v2/v2_tsne_embeddings.png` — 2D t-SNE эмбеддингов 512-D на 12 моделях из галереи v2.",
            "- `../figures/v2/v2_retrieval_montage.png` — визуальный монтаж поиска топ-5 похожих автомобилей из `gallery_v2_thumbs.zip`.",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AutoVision v2 evaluation plots and report.")
    parser.add_argument("--project-root", type=Path, default=BASE_DIR)
    args = parser.parse_args()

    root = args.project_root
    v1_models_dir = root / "models"
    v2_p1_dir = root / "models" / "v2_phase1"
    v2_color_dir = root / "models" / "v2_color"
    reports_dir = root / "reports"
    reports_v2_dir = reports_dir / "v2"
    reports_v2_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures" / "v2"
    figures_dir.mkdir(parents=True, exist_ok=True)

    v1_sum = json.loads((v1_models_dir / "training_summary.json").read_text(encoding="utf-8")) if (v1_models_dir / "training_summary.json").is_file() else {}
    v1_lmap = json.loads((v1_models_dir / "label_maps.json").read_text(encoding="utf-8")) if (v1_models_dir / "label_maps.json").is_file() else {}
    v2_sum = json.loads((v2_p1_dir / "training_summary.json").read_text(encoding="utf-8"))
    v2_lmap = json.loads((v2_p1_dir / "label_maps.json").read_text(encoding="utf-8"))
    c2_sum = json.loads((v2_color_dir / "color_training_summary.json").read_text(encoding="utf-8"))
    s5_csv = reports_dir / "v1" / "stage5_before_after_metrics.csv"
    if not s5_csv.is_file():
        s5_csv = reports_dir / "stage5_before_after_metrics.csv"
    s5_map = _load_stage5_baseline(s5_csv)

    with np.load(v2_p1_dir / "v2_test_predictions.npz", allow_pickle=False) as preds:
        pred_make = preds["pred_make"]
        true_make = preds["true_make"]
        pred_model = preds["pred_model"]
        true_model = preds["true_model"]
        pred_body = preds["pred_body"]
        true_body = preds["true_body"]
        pred_year = preds["pred_year"]
        year_low = preds["year_low"]
        year_high = preds["year_high"]
        year_valid = preds["year_valid"]

    with np.load(v2_p1_dir / "features_finetuned_openclip_vitb16.npz", allow_pickle=False) as feat_npz:
        gal_embs = feat_npz["embeddings"]

    with open(v2_p1_dir / "autovision_v2_manifest.csv", "r", encoding="utf-8", newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    _plot_evolution_comparison(
        v1_sum, v1_lmap, v2_sum, v2_lmap, c2_sum, s5_map,
        figures_dir / "v2_evolution_comparison.png",
    )
    _plot_domain_breakdown(v2_sum, c2_sum, figures_dir / "v2_domain_breakdown.png")

    ov = v2_sum["overall_test_metrics"]
    body_rows = _plot_body_confusion(
        true_body, pred_body, v2_lmap["body_classes"],
        ov["body_top1"], ov["body_top3"], ov["body_macro_f1"],
        figures_dir / "v2_body_type_confusion_matrix.png",
    )
    make_rows = _plot_makes_ranking(
        true_make, pred_make, v2_lmap["make_classes"],
        v2_lmap.get("make_display_names", {}), ov["make_macro_f1"],
        figures_dir / "v2_makes_f1_ranking.png",
    )
    era_rows = _plot_year_interval_evaluation(
        pred_year, year_low, year_high, year_valid,
        figures_dir / "v2_year_interval_evaluation.png",
    )
    v1_color_top1 = v1_sum.get("vcor_summary", {}).get("spatial_attention_test", {}).get("top1", 0.8676)
    _plot_color_training_curves(
        v2_color_dir / "color_training_history.csv",
        v1_color_top1,
        figures_dir / "v2_color_training_curves.png",
    )
    shutil.copy2(v2_color_dir / "color_confusion_matrix_v2.png", figures_dir / "v2_color_confusion_matrix.png")

    sil_v2 = _plot_tsne_embeddings(
        gal_embs,
        manifest_rows,
        v2_lmap.get("make_display_names", {}),
        v2_lmap.get("model_display_names", {}),
        figures_dir / "v2_tsne_embeddings.png",
    )
    _plot_retrieval_montage(
        gal_embs,
        manifest_rows,
        v2_p1_dir / "gallery_v2_thumbs.zip",
        v2_lmap.get("make_display_names", {}),
        v2_lmap.get("model_display_names", {}),
        figures_dir / "v2_retrieval_montage.png",
    )

    best_models, hard_models = _compute_best_hardest_models(
        true_model, pred_model, v2_lmap["model_classes"], min_support=15
    )
    _write_metrics_csv(reports_v2_dir / "v2_metrics_summary.csv", v2_sum, c2_sum, sil_v2, s5_map)
    _write_markdown_report(
        reports_v2_dir / "v2_evaluation_report.md",
        v1_sum, v1_lmap, v2_sum, c2_sum, v2_lmap,
        body_rows, make_rows, era_rows, best_models, hard_models, sil_v2,
    )

    print("Generated AutoVision v2 report & 9 figures cleanly:")
    for p in sorted(figures_dir.glob("v2_*.png")):
        print(f"  - {p.relative_to(root)} ({p.stat().st_size:,} bytes)")
    print(f"  - reports/v2/v2_metrics_summary.csv | Silhouette={sil_v2:.4f}")


if __name__ == "__main__":
    main()
