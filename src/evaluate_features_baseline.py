"""Evaluate extracted vehicle embeddings (Linear Probe, kNN, Prototypes, Few-Shot, Year Ridge, Ablation) and generate thesis figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, silhouette_score
from torch.utils.data import DataLoader, TensorDataset

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extract_features import load_feature_bundle  # noqa: E402


def load_manifest_lookup(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Loads clean manifest rows indexed by normalized rel_path."""
    lookup: dict[str, dict[str, Any]] = {}
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("is_clean", "")).strip().lower() != "true":
                continue
            rel = row["rel_path"].replace("\\", "/")
            lookup[rel] = {
                "rel_path": rel,
                "make": row["make"],
                "model": row["model"],
                "canonical_class": row["canonical_class"],
                "raw_year": int(row["raw_year"]),
                "year_valid": str(row.get("year_valid", "")).strip().lower() == "true",
                "split": row["split"],
            }
    return lookup


def align_bundle_with_manifest(
    bundle: dict[str, Any],
    manifest_lookup: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Aligns feature bundle rows with manifest metadata by rel_path."""
    emb_raw = bundle["embeddings"]
    rel_paths = [str(p).replace("\\", "/") for p in bundle["rel_paths"]]
    keep_idx: list[int] = []
    meta_rows: list[dict[str, Any]] = []
    for i, rp in enumerate(rel_paths):
        rec = manifest_lookup.get(rp)
        if rec is not None:
            keep_idx.append(i)
            meta_rows.append(rec)

    if not keep_idx:
        raise ValueError("Zero overlapping rel_paths between feature bundle and manifest.")

    X = emb_raw[keep_idx].astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-12)
    return X, meta_rows


def _topk_macro_f1(
    scores: np.ndarray,
    y_true: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    k = min(5, scores.shape[1])
    top5_idx = np.argsort(-scores, axis=1)[:, :k]
    y_pred = top5_idx[:, 0]
    top1 = float(np.mean(y_pred == y_true))
    top5 = float(np.mean([yt in row for yt, row in zip(y_true, top5_idx)]))
    macro_f1 = float(f1_score(y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0))
    return {"top1": top1, "top5": top5, "macro_f1": macro_f1}


def eval_nearest_prototype(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    num_classes: int,
) -> tuple[dict[str, float], np.ndarray]:
    """Builds L2-normalized centroids on train and evaluates cosine similarity on eval."""
    dim = X_train.shape[1]
    protos = np.zeros((num_classes, dim), dtype=np.float32)
    for c in range(num_classes):
        mask = y_train == c
        if np.any(mask):
            m = X_train[mask].mean(axis=0)
            protos[c] = m / max(float(np.linalg.norm(m)), 1e-12)
    scores = X_eval @ protos.T
    return _topk_macro_f1(scores, y_eval, num_classes), protos


def eval_cosine_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    num_classes: int,
    k: int = 5,
    chunk_size: int = 512,
) -> dict[str, float]:
    """Evaluates distance-weighted cosine kNN in bounded-memory chunks."""
    n_eval = X_eval.shape[0]
    k_eff = min(k, X_train.shape[0])
    scores = np.zeros((n_eval, num_classes), dtype=np.float32)

    X_train_T = X_train.T
    for start in range(0, n_eval, chunk_size):
        end = min(start + chunk_size, n_eval)
        sim_chunk = X_eval[start:end] @ X_train_T
        top_idx = np.argpartition(-sim_chunk, kth=k_eff - 1, axis=1)[:, :k_eff]
        row_arange = np.arange(end - start)[:, None]
        top_sims = np.clip(sim_chunk[row_arange, top_idx], 0.0, None)
        top_labels = y_train[top_idx]

        chunk_scores = np.zeros((end - start, num_classes), dtype=np.float32)
        for j in range(k_eff):
            np.add.at(chunk_scores, (np.arange(end - start), top_labels[:, j]), top_sims[:, j])
        scores[start:end] = chunk_scores

    return _topk_macro_f1(scores, y_eval, num_classes)


def eval_few_shot_retrieval(
    X_all: np.ndarray,
    meta_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluates 1-shot prototype retrieval for few_shot classes (>=2 images) across all classes."""
    by_split_cls: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    all_classes: set[str] = set()
    for i, r in enumerate(meta_rows):
        all_classes.add(r["canonical_class"])
        by_split_cls[r["split"]][r["canonical_class"]].append(i)

    cls_list = sorted(all_classes)
    cls_to_id = {c: idx for idx, c in enumerate(cls_list)}
    dim = X_all.shape[1]

    gallery_vecs: list[np.ndarray] = []
    gallery_labels: list[int] = []

    for cls_name, idxs in by_split_cls["train"].items():
        m = X_all[idxs].mean(axis=0)
        gallery_vecs.append(m / max(float(np.linalg.norm(m)), 1e-12))
        gallery_labels.append(cls_to_id[cls_name])

    query_idxs: list[int] = []
    query_labels: list[int] = []
    eval_few_classes = 0

    for cls_name in sorted(by_split_cls["few_shot"]):
        idxs = sorted(by_split_cls["few_shot"][cls_name], key=lambda i: meta_rows[i]["rel_path"])
        if not idxs:
            continue
        support_idx = idxs[0]
        gallery_vecs.append(X_all[support_idx])
        gallery_labels.append(cls_to_id[cls_name])
        if len(idxs) >= 2:
            eval_few_classes += 1
            for q_idx in idxs[1:]:
                query_idxs.append(q_idx)
                query_labels.append(cls_to_id[cls_name])

    if not query_idxs:
        return {
            "few_shot_classes_eval": 0,
            "few_shot_queries": 0,
            "few_shot_gallery_classes": len(gallery_labels),
            "few_shot_top1": 0.0,
            "few_shot_top5": 0.0,
        }

    G = np.vstack(gallery_vecs).astype(np.float32)
    g_lbls = np.array(gallery_labels, dtype=np.int64)
    Q = X_all[query_idxs].astype(np.float32)
    y_q = np.array(query_labels, dtype=np.int64)

    sims = Q @ G.T
    k = min(5, G.shape[0])
    top_idx = np.argsort(-sims, axis=1)[:, :k]
    pred_labels = g_lbls[top_idx]

    top1 = float(np.mean(pred_labels[:, 0] == y_q))
    top5 = float(np.mean([yt in row for yt, row in zip(y_q, pred_labels)]))
    return {
        "few_shot_classes_eval": eval_few_classes,
        "few_shot_queries": len(query_idxs),
        "few_shot_gallery_classes": len(gallery_labels),
        "few_shot_top1": top1,
        "few_shot_top5": top5,
    }


def train_and_eval_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    epochs: int = 20,
    seed: int = 42,
) -> tuple[dict[str, float], nn.Module]:
    """Trains a PyTorch Linear Probe with AdamW + CosineAnnealingLR and evaluates on test."""
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = nn.Linear(X_train.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.int64)))
    train_loader = DataLoader(train_ds, batch_size=min(512, max(len(train_ds), 1)), shuffle=True)

    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    best_val_acc = -1.0
    X_val_t = torch.from_numpy(X_val).to(device)

    for _ in range(max(epochs, 1)):
        head.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(head(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if len(X_val) > 0:
            head.eval()
            with torch.no_grad():
                preds = head(X_val_t).argmax(dim=1).cpu().numpy()
            val_acc = float(np.mean(preds == y_val))
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        test_scores = head(torch.from_numpy(X_test).to(device)).cpu().numpy()
    return _topk_macro_f1(test_scores, y_test, num_classes), head.cpu()


def train_and_eval_year_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, float], Ridge, np.ndarray]:
    """Tunes Ridge alpha on val and evaluates year regression metrics on test."""
    best_alpha = 1.0
    best_mae = float("inf")
    if len(X_val) > 0:
        for alpha in (0.1, 1.0, 10.0, 50.0):
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            preds_val = model.predict(X_val)
            mae = float(np.mean(np.abs(preds_val - y_val)))
            if mae < best_mae:
                best_mae = mae
                best_alpha = alpha

    reg = Ridge(alpha=best_alpha)
    reg.fit(X_train, y_train)
    preds_test = reg.predict(X_test)
    errors = np.abs(preds_test - y_test)

    metrics = {
        "alpha": float(best_alpha),
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean((preds_test - y_test) ** 2))),
        "within_1yr": float(np.mean(errors <= 1.0)),
        "within_2yr": float(np.mean(errors <= 2.0)),
        "within_3yr": float(np.mean(errors <= 3.0)),
    }
    return metrics, reg, preds_test


def _match_ablation_bundles(
    ablation_paths: list[Path] | None,
    bundles: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Loads and matches each --ablation-raw file to the corresponding feature bundle index."""
    if not ablation_paths:
        return {}

    abl_bundles = [(p, load_feature_bundle(p)) for p in ablation_paths]
    mapping: dict[int, dict[str, Any]] = {}
    used_abl: set[int] = set()

    for idx, b in enumerate(bundles):
        bb = b["metadata"].get("backbone", "")
        for j, (ap, ab) in enumerate(abl_bundles):
            if j in used_abl:
                continue
            ab_bb = ab["metadata"].get("backbone", "")
            if (bb and ab_bb == bb) or (bb and bb in ap.stem):
                mapping[idx] = ab
                used_abl.add(j)
                break

    for idx in range(len(bundles)):
        if idx in mapping:
            continue
        for j, (_, ab) in enumerate(abl_bundles):
            if j not in used_abl:
                mapping[idx] = ab
                used_abl.add(j)
                break

    return mapping


def _fuse_bundles(bundles: list[dict[str, Any]], fused_name: str) -> dict[str, Any]:
    """Aligns bundles by rel_path, L2-normalizes each block, concatenates, and L2-normalizes."""
    lookups: list[dict[str, int]] = []
    for b in bundles:
        lookups.append({str(rp).replace("\\", "/"): i for i, rp in enumerate(b["rel_paths"])})

    common_rels: list[str] = []
    idx_lists: list[list[int]] = [[] for _ in bundles]
    for rp_raw in bundles[0]["rel_paths"]:
        rp = str(rp_raw).replace("\\", "/")
        if all(rp in lk for lk in lookups):
            common_rels.append(rp)
            for b_i, lk in enumerate(lookups):
                idx_lists[b_i].append(lk[rp])

    if not common_rels:
        raise ValueError("Cannot fuse bundles: zero overlapping rel_paths.")

    blocks: list[np.ndarray] = []
    for b, idxs in zip(bundles, idx_lists):
        block = b["embeddings"][idxs].astype(np.float32)
        block = block / np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-12)
        blocks.append(block)

    fused = np.hstack(blocks)
    fused = fused / np.maximum(np.linalg.norm(fused, axis=1, keepdims=True), 1e-12)

    return {
        "embeddings": fused.astype(np.float16),
        "bboxes": bundles[0]["bboxes"][idx_lists[0]],
        "used_yolo": bundles[0]["used_yolo"][idx_lists[0]],
        "rel_paths": np.asarray(common_rels, dtype=str),
        "metadata": {
            "backbone": fused_name,
            "embed_dim": int(fused.shape[1]),
            "yolo_hit_rate": bundles[0]["metadata"].get("yolo_hit_rate"),
        },
    }


def plot_pca_variance(
    pca_curves: list[tuple[str, np.ndarray]],
    out_path: Path,
) -> None:
    """Plots cumulative PCA explained variance across evaluated backbones."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for label, cum_var in pca_curves:
        xs = np.arange(1, len(cum_var) + 1)
        ax.plot(xs, cum_var * 100.0, linewidth=2.2, label=label)
    ax.axhline(90.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, label="90% variance")
    ax.set_title("Cumulative PCA Explained Variance (Clean Train Features)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_tsne_models(
    X_test: np.ndarray,
    meta_test: list[dict[str, Any]],
    backbone_name: str,
    out_path: Path,
    seed: int = 42,
) -> float:
    """Plots 2D t-SNE of top-10 test classes (up to 1,500 points) with Silhouette annotation."""
    counts = Counter(r["canonical_class"] for r in meta_test)
    top_classes = [cls for cls, _ in counts.most_common(10)]
    cls_set = set(top_classes)
    idxs = [i for i, r in enumerate(meta_test) if r["canonical_class"] in cls_set]

    rng = np.random.default_rng(seed)
    if len(idxs) > 1500:
        idxs = sorted(rng.choice(idxs, size=1500, replace=False).tolist())

    X_sub = X_test[idxs]
    y_sub_str = [meta_test[i]["canonical_class"] for i in idxs]
    cls_to_id = {c: idx for idx, c in enumerate(top_classes)}
    y_sub = np.array([cls_to_id[c] for c in y_sub_str])

    n_pts = len(idxs)
    pca_dim = min(50, X_sub.shape[1], max(n_pts - 1, 2))
    X_pca = PCA(n_components=pca_dim, random_state=seed).fit_transform(X_sub) if pca_dim >= 2 else X_sub

    perp = min(30.0, max(2.0, float(n_pts - 1) / 4.0))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=seed, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(X_pca)

    sil = float(silhouette_score(X_sub, y_sub)) if len(set(y_sub)) >= 2 and n_pts > len(set(y_sub)) else 0.0

    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=150)
    cmap = plt.get_cmap("tab10")
    for cid, cls_name in enumerate(top_classes):
        m = y_sub == cid
        if np.any(m):
            ax.scatter(coords[m, 0], coords[m, 1], s=18, alpha=0.78, color=cmap(cid % 10), label=cls_name)
    ax.set_title(
        f"2D t-SNE of Top-{len(top_classes)} Test Models ({backbone_name} | Cosine Sil={sil:.3f})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8, frameon=True)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return sil


def plot_year_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mae: float,
    backbone_name: str,
    out_path: Path,
) -> None:
    """Plots True vs Predicted Year scatter and absolute error histogram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)

    ymin = float(min(y_true.min(), y_pred.min()) - 1)
    ymax = float(max(y_true.max(), y_pred.max()) + 1)
    ax1.scatter(y_true, y_pred, s=10, alpha=0.35, color="#1f77b4")
    ax1.plot([ymin, ymax], [ymin, ymax], "r--", linewidth=1.5, label="Ideal (y = x)")
    ax1.fill_between([ymin, ymax], [ymin - 2, ymax - 2], [ymin + 2, ymax + 2], color="red", alpha=0.1, label="±2 yr band")
    ax1.set_title(f"Year Prediction ({backbone_name} | MAE={mae:.2f} yr)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("True Model Year")
    ax1.set_ylabel("Predicted Year")
    ax1.set_xlim(ymin, ymax)
    ax1.set_ylim(ymin, ymax)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", fontsize=8)

    residuals = y_pred - y_true
    ax2.hist(residuals, bins=30, color="#2ca02c", alpha=0.75, edgecolor="white")
    ax2.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax2.axvline(-2.0, color="red", linestyle=":", linewidth=1.2)
    ax2.axvline(2.0, color="red", linestyle=":", linewidth=1.2, label="±2 yr")
    ax2.set_title("Year Residual Distribution (Pred - True)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Error (years)")
    ax2.set_ylabel("Images")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_retrieval_montage(
    X_train: np.ndarray,
    meta_train: list[dict[str, Any]],
    X_test: np.ndarray,
    meta_test: list[dict[str, Any]],
    data_dir: Path,
    backbone_name: str,
    out_path: Path,
    seed: int = 42,
) -> bool:
    """Creates a 6-row Visual Top-5 Retrieval Montage if raw images exist under data_dir."""
    if not data_dir.is_dir():
        return False

    rng = random.Random(seed)
    make_counts = Counter(r["make"] for r in meta_test)
    by_make: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(meta_test):
        if (data_dir / r["rel_path"]).is_file():
            by_make[r["make"]].append(i)

    seen_classes: set[str] = set()
    query_idxs: list[int] = []

    for mk, _ in make_counts.most_common():
        candidates = sorted(by_make.get(mk, []), key=lambda idx: meta_test[idx]["rel_path"])
        if not candidates:
            continue
        pick = rng.choice(candidates)
        query_idxs.append(pick)
        seen_classes.add(meta_test[pick]["canonical_class"])
        if len(query_idxs) == 6:
            break

    if len(query_idxs) < 6:
        remaining = [
            i for i, r in enumerate(meta_test)
            if i not in query_idxs and r["canonical_class"] not in seen_classes and (data_dir / r["rel_path"]).is_file()
        ]
        remaining.sort(key=lambda idx: meta_test[idx]["rel_path"])
        rng.shuffle(remaining)
        for i in remaining:
            cls_name = meta_test[i]["canonical_class"]
            if cls_name in seen_classes:
                continue
            seen_classes.add(cls_name)
            query_idxs.append(i)
            if len(query_idxs) == 6:
                break

    if not query_idxs:
        return False

    Q = X_test[query_idxs]
    sims = Q @ X_train.T
    k_neighbors = min(5, X_train.shape[0])
    top_k_idx = np.argsort(-sims, axis=1)[:, :k_neighbors]

    n_rows = len(query_idxs)
    n_cols = 1 + k_neighbors
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.75 * n_cols, 2.75 * n_rows + 0.6), dpi=140)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_i, q_idx in enumerate(query_idxs):
        q_meta = meta_test[q_idx]
        with Image.open(data_dir / q_meta["rel_path"]) as im:
            q_img = ImageOps.fit(im.convert("RGB"), (220, 165))
        ax_q = axes[row_i, 0]
        ax_q.imshow(q_img)
        ax_q.set_title(
            f"QUERY\n{q_meta['canonical_class']}\n({q_meta['raw_year']})",
            fontsize=7.5,
            fontweight="bold",
            pad=4,
        )
        for spine in ax_q.spines.values():
            spine.set_edgecolor("#1f77b4")
            spine.set_linewidth(2.5)
        ax_q.set_xticks([])
        ax_q.set_yticks([])

        for col_j, tr_idx in enumerate(top_k_idx[row_i], start=1):
            tr_meta = meta_train[int(tr_idx)]
            sim_val = float(sims[row_i, int(tr_idx)])
            if tr_meta["canonical_class"] == q_meta["canonical_class"]:
                border = "#2ca02c"
            elif tr_meta["make"] == q_meta["make"]:
                border = "#ff7f0e"
            else:
                border = "#d62728"

            tr_path = data_dir / tr_meta["rel_path"]
            ax_n = axes[row_i, col_j]
            if tr_path.is_file():
                with Image.open(tr_path) as im:
                    n_img = ImageOps.fit(im.convert("RGB"), (220, 165))
                ax_n.imshow(n_img)
            ax_n.set_title(
                f"#{col_j} sim={sim_val:.3f}\n{tr_meta['canonical_class']}\n({tr_meta['raw_year']})",
                fontsize=7.5,
                pad=4,
            )
            for spine in ax_n.spines.values():
                spine.set_edgecolor(border)
                spine.set_linewidth(2.5)
            ax_n.set_xticks([])
            ax_n.set_yticks([])

    fig.suptitle(
        f"Visual Top-{k_neighbors} Cosine Retrieval ({backbone_name} | Green=Model, Orange=Make, Red=Miss)",
        fontsize=12,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.945], h_pad=1.6, w_pad=0.6)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def evaluate_all(
    manifest_path: Path,
    feature_paths: list[Path],
    ablation_paths: list[Path] | None,
    data_dir: Path,
    out_dir: Path,
    epochs: int = 20,
    seed: int = 42,
    fuse: bool = False,
) -> dict[str, Any]:
    """Runs full baseline evaluation across feature bundles and writes metrics & thesis plots."""
    manifest_lookup = load_manifest_lookup(manifest_path)
    bundles = [load_feature_bundle(p) for p in feature_paths]
    abl_map = _match_ablation_bundles(ablation_paths, bundles)
    eval_paths = list(feature_paths)
    eval_bundles = list(bundles)

    if fuse and len(bundles) >= 2:
        bb_names = [
            b["metadata"].get("backbone") or p.stem.replace("features_", "")
            for p, b in zip(feature_paths, bundles)
        ]
        prefixes = [name.split("_")[0] for name in bb_names]
        fused_name = (
            "fusion_" + "_".join(prefixes)
            if len(set(prefixes)) == len(prefixes)
            else "fusion_" + "_".join(bb_names)
        )
        fused_idx = len(eval_bundles)
        eval_paths.append(Path(f"{fused_name}.npz"))
        eval_bundles.append(_fuse_bundles(bundles, fused_name))
        if all(i in abl_map for i in range(len(bundles))):
            abl_map[fused_idx] = _fuse_bundles(
                [abl_map[i] for i in range(len(bundles))],
                fused_name,
            )

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    pca_curves: list[tuple[str, np.ndarray]] = []
    ablation_summaries: list[dict[str, Any]] = []

    best_backbone_info: dict[str, Any] | None = None
    best_model_lp_top1 = -1.0
    best_retrieval_info: dict[str, Any] | None = None
    best_model_knn_top1 = -1.0

    for b_idx, (feat_path, bundle) in enumerate(zip(eval_paths, eval_bundles)):
        bb_name = bundle["metadata"].get("backbone") or feat_path.stem.replace("features_", "")
        X, meta_rows = align_bundle_with_manifest(bundle, manifest_lookup)

        train_idx = [i for i, r in enumerate(meta_rows) if r["split"] == "train"]
        val_idx = [i for i, r in enumerate(meta_rows) if r["split"] == "val"]
        test_idx = [i for i, r in enumerate(meta_rows) if r["split"] == "test"]
        if not train_idx or not test_idx:
            raise ValueError(f"{bb_name}: train ({len(train_idx)}) and test ({len(test_idx)}) must be non-empty.")

        makes = sorted({meta_rows[i]["make"] for i in train_idx})
        make_to_id = {m: i for i, m in enumerate(makes)}
        models = sorted({meta_rows[i]["canonical_class"] for i in train_idx})
        model_to_id = {c: i for i, c in enumerate(models)}

        # Filter val/test to labels present in train (guaranteed on full manifest, safe on subsets)
        val_valid = [i for i in val_idx if meta_rows[i]["make"] in make_to_id and meta_rows[i]["canonical_class"] in model_to_id]
        test_valid = [i for i in test_idx if meta_rows[i]["make"] in make_to_id and meta_rows[i]["canonical_class"] in model_to_id]

        X_tr = X[train_idx]
        X_va = X[val_valid] if val_valid else X[:0]
        X_te = X[test_valid]

        y_make_tr = np.array([make_to_id[meta_rows[i]["make"]] for i in train_idx], dtype=np.int64)
        y_make_va = np.array([make_to_id[meta_rows[i]["make"]] for i in val_valid], dtype=np.int64)
        y_make_te = np.array([make_to_id[meta_rows[i]["make"]] for i in test_valid], dtype=np.int64)

        y_mod_tr = np.array([model_to_id[meta_rows[i]["canonical_class"]] for i in train_idx], dtype=np.int64)
        y_mod_va = np.array([model_to_id[meta_rows[i]["canonical_class"]] for i in val_valid], dtype=np.int64)
        y_mod_te = np.array([model_to_id[meta_rows[i]["canonical_class"]] for i in test_valid], dtype=np.int64)

        # 1. Prototypes
        proto_make, _ = eval_nearest_prototype(X_tr, y_make_tr, X_te, y_make_te, len(makes))
        proto_mod, _ = eval_nearest_prototype(X_tr, y_mod_tr, X_te, y_mod_te, len(models))

        # 2. Cosine kNN (k=5)
        knn_make = eval_cosine_knn(X_tr, y_make_tr, X_te, y_make_te, len(makes), k=5)
        knn_mod = eval_cosine_knn(X_tr, y_mod_tr, X_te, y_mod_te, len(models), k=5)

        # 3. Linear Probes
        lp_make, head_make = train_and_eval_linear_probe(
            X_tr, y_make_tr, X_va, y_make_va, X_te, y_make_te, len(makes), epochs=epochs, seed=seed
        )
        lp_mod, head_mod = train_and_eval_linear_probe(
            X_tr, y_mod_tr, X_va, y_mod_va, X_te, y_mod_te, len(models), epochs=epochs, seed=seed
        )

        # 4. Few-shot retrieval
        few_metrics = eval_few_shot_retrieval(X, meta_rows)

        # 5. Year Regression (Ridge)
        yr_tr = [i for i in train_idx if meta_rows[i]["year_valid"]]
        yr_va = [i for i in val_valid if meta_rows[i]["year_valid"]]
        yr_te = [i for i in test_valid if meta_rows[i]["year_valid"]]
        year_metrics, reg_year, yr_preds_te = train_and_eval_year_ridge(
            X[yr_tr],
            np.array([meta_rows[i]["raw_year"] for i in yr_tr], dtype=np.float32),
            X[yr_va] if yr_va else X[:0],
            np.array([meta_rows[i]["raw_year"] for i in yr_va], dtype=np.float32),
            X[yr_te],
            np.array([meta_rows[i]["raw_year"] for i in yr_te], dtype=np.float32),
        )
        y_yr_true_te = np.array([meta_rows[i]["raw_year"] for i in yr_te], dtype=np.float32)

        # PCA cumulative curve
        n_comp = min(256, X_tr.shape[0], X_tr.shape[1])
        pca = PCA(n_components=n_comp, random_state=seed).fit(X_tr)
        pca_curves.append((f"{bb_name} ({X_tr.shape[1]}D)", np.cumsum(pca.explained_variance_ratio_)))

        for task, method, res in [
            ("Make", "Nearest Prototype", proto_make),
            ("Make", "Cosine kNN (k=5)", knn_make),
            ("Make", "Linear Probe", lp_make),
            ("Model", "Nearest Prototype", proto_mod),
            ("Model", "Cosine kNN (k=5)", knn_mod),
            ("Model", "Linear Probe", lp_mod),
        ]:
            metric_rows.append(
                {
                    "backbone": bb_name,
                    "dim": X.shape[1],
                    "task": task,
                    "method": method,
                    "split": "test",
                    "n_eval": len(test_valid),
                    "top1": round(res["top1"], 4),
                    "top5": round(res["top5"], 4),
                    "macro_f1": round(res["macro_f1"], 4),
                    "mae_years": "",
                    "within_2yr": "",
                }
            )

        metric_rows.append(
            {
                "backbone": bb_name,
                "dim": X.shape[1],
                "task": "Few-Shot Model",
                "method": "1-Shot Prototype",
                "split": "few_shot",
                "n_eval": few_metrics["few_shot_queries"],
                "top1": round(few_metrics["few_shot_top1"], 4),
                "top5": round(few_metrics["few_shot_top5"], 4),
                "macro_f1": "",
                "mae_years": "",
                "within_2yr": "",
            }
        )
        metric_rows.append(
            {
                "backbone": bb_name,
                "dim": X.shape[1],
                "task": "Year",
                "method": f"Ridge (alpha={year_metrics['alpha']})",
                "split": "test_year_valid",
                "n_eval": len(yr_te),
                "top1": "",
                "top5": "",
                "macro_f1": "",
                "mae_years": round(year_metrics["mae"], 3),
                "within_2yr": round(year_metrics["within_2yr"], 4),
            }
        )

        # 6. Optional YOLO Crop vs Raw Frame Ablation
        if b_idx in abl_map:
            abl_bundle = abl_map[b_idx]
            X_raw_all, meta_raw = align_bundle_with_manifest(abl_bundle, manifest_lookup)
            crop_idx_by_rel = {meta_rows[i]["rel_path"]: i for i in test_valid}
            matched_crop_idx: list[int] = []
            matched_raw_idx: list[int] = []
            for j, r in enumerate(meta_raw):
                ci = crop_idx_by_rel.get(r["rel_path"])
                if ci is not None:
                    matched_crop_idx.append(ci)
                    matched_raw_idx.append(j)

            if matched_crop_idx:
                X_c_sub = X[matched_crop_idx]
                X_r_sub = X_raw_all[matched_raw_idx]
                y_mk_sub = np.array([make_to_id[meta_rows[i]["make"]] for i in matched_crop_idx], dtype=np.int64)
                y_md_sub = np.array([model_to_id[meta_rows[i]["canonical_class"]] for i in matched_crop_idx], dtype=np.int64)

                with torch.no_grad():
                    c_mk = _topk_macro_f1(head_make(torch.from_numpy(X_c_sub)).numpy(), y_mk_sub, len(makes))
                    r_mk = _topk_macro_f1(head_make(torch.from_numpy(X_r_sub)).numpy(), y_mk_sub, len(makes))
                    c_md = _topk_macro_f1(head_mod(torch.from_numpy(X_c_sub)).numpy(), y_md_sub, len(models))
                    r_md = _topk_macro_f1(head_mod(torch.from_numpy(X_r_sub)).numpy(), y_md_sub, len(models))

                c_proto_md, _ = eval_nearest_prototype(X_tr, y_mod_tr, X_c_sub, y_md_sub, len(models))
                r_proto_md, _ = eval_nearest_prototype(X_tr, y_mod_tr, X_r_sub, y_md_sub, len(models))

                yr_sub_pos = [k for k, ci in enumerate(matched_crop_idx) if meta_rows[ci]["year_valid"]]
                if yr_sub_pos:
                    y_yr_sub = np.array([meta_rows[matched_crop_idx[k]]["raw_year"] for k in yr_sub_pos], dtype=np.float32)
                    c_yr_mae = float(np.mean(np.abs(reg_year.predict(X_c_sub[yr_sub_pos]) - y_yr_sub)))
                    r_yr_mae = float(np.mean(np.abs(reg_year.predict(X_r_sub[yr_sub_pos]) - y_yr_sub)))
                else:
                    c_yr_mae, r_yr_mae = math.nan, math.nan

                ablation_summaries.append(
                    {
                        "backbone": bb_name,
                        "n_matched": len(matched_crop_idx),
                        "crop_make_lp_top1": c_mk["top1"],
                        "raw_make_lp_top1": r_mk["top1"],
                        "crop_model_lp_top1": c_md["top1"],
                        "raw_model_lp_top1": r_md["top1"],
                        "crop_model_proto_top1": c_proto_md["top1"],
                        "raw_model_proto_top1": r_proto_md["top1"],
                        "crop_year_mae": c_yr_mae,
                        "raw_year_mae": r_yr_mae,
                    }
                )
                for cond, mk_res, md_res, mae_v in [
                    ("YOLO Crop", c_mk, c_md, c_yr_mae),
                    ("Raw Frame (No YOLO)", r_mk, r_md, r_yr_mae),
                ]:
                    metric_rows.append(
                        {
                            "backbone": bb_name,
                            "dim": X.shape[1],
                            "task": "Ablation Make/Model",
                            "method": f"Linear Probe ({cond})",
                            "split": "ablation_test",
                            "n_eval": len(matched_crop_idx),
                            "top1": round(md_res["top1"], 4),
                            "top5": round(md_res["top5"], 4),
                            "macro_f1": round(mk_res["top1"], 4),
                            "mae_years": "" if math.isnan(mae_v) else round(mae_v, 3),
                            "within_2yr": "",
                        }
                    )

        if knn_mod["top1"] > best_model_knn_top1:
            best_model_knn_top1 = knn_mod["top1"]
            best_retrieval_info = {
                "backbone": bb_name,
                "X_tr": X_tr,
                "meta_tr": [meta_rows[i] for i in train_idx],
                "X_te": X_te,
                "meta_te": [meta_rows[i] for i in test_valid],
                "model_knn_top1": knn_mod["top1"],
                "make_knn_top1": knn_make["top1"],
            }

        if lp_mod["top1"] > best_model_lp_top1:
            best_model_lp_top1 = lp_mod["top1"]
            best_backbone_info = {
                "backbone": bb_name,
                "X_tr": X_tr,
                "meta_tr": [meta_rows[i] for i in train_idx],
                "X_te": X_te,
                "meta_te": [meta_rows[i] for i in test_valid],
                "y_yr_true_te": y_yr_true_te,
                "y_yr_pred_te": yr_preds_te,
                "yr_mae": year_metrics["mae"],
                "yolo_hit_rate": bundle["metadata"].get("yolo_hit_rate"),
            }

    # Generate thesis figures
    plot_pca_variance(pca_curves, figures_dir / "feature_pca_variance.png")
    assert best_backbone_info is not None
    assert best_retrieval_info is not None
    tsne_sil = plot_tsne_models(
        best_backbone_info["X_te"],
        best_backbone_info["meta_te"],
        best_backbone_info["backbone"],
        figures_dir / "feature_tsne_models.png",
        seed=seed,
    )
    plot_year_regression(
        best_backbone_info["y_yr_true_te"],
        best_backbone_info["y_yr_pred_te"],
        best_backbone_info["yr_mae"],
        best_backbone_info["backbone"],
        figures_dir / "feature_year_regression.png",
    )
    montage_ok = plot_retrieval_montage(
        best_retrieval_info["X_tr"],
        best_retrieval_info["meta_tr"],
        best_retrieval_info["X_te"],
        best_retrieval_info["meta_te"],
        data_dir,
        best_retrieval_info["backbone"],
        figures_dir / "feature_retrieval_montage.png",
        seed=seed,
    )

    metrics_csv_path = out_dir / "feature_extraction_metrics.csv"
    fieldnames = [
        "backbone",
        "dim",
        "task",
        "method",
        "split",
        "n_eval",
        "top1",
        "top5",
        "macro_f1",
        "mae_years",
        "within_2yr",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    report_path = out_dir / "feature_extraction_report.md"
    _write_markdown_report(
        report_path=report_path,
        metric_rows=metric_rows,
        ablation_summaries=ablation_summaries,
        best_backbone=best_backbone_info["backbone"],
        retrieval_info=best_retrieval_info,
        tsne_sil=tsne_sil,
        montage_ok=montage_ok,
    )

    print(f"[DONE] Metrics -> {metrics_csv_path}")
    print(f"[DONE] Report  -> {report_path}")
    print(f"[DONE] Figures -> {figures_dir}")
    return {
        "metrics_csv": str(metrics_csv_path),
        "report_md": str(report_path),
        "best_backbone": best_backbone_info["backbone"],
        "tsne_silhouette": tsne_sil,
        "montage_created": montage_ok,
    }


def _write_markdown_report(
    report_path: Path,
    metric_rows: list[dict[str, Any]],
    ablation_summaries: list[dict[str, Any]],
    best_backbone: str,
    retrieval_info: dict[str, Any],
    tsne_sil: float,
    montage_ok: bool,
) -> None:
    lines = [
        "# Отчёт по извлечению и оценке признаков (AutoVision Feature Baseline)",
        "",
        "Оценка выполнена строго по схеме без утечек (`train` → `val` для выбора модели → `test` + `few_shot`).",
        "",
        "## 1. Сводная таблица базовых метрик (Test / Few-Shot)",
        "",
        "| Backbone | Dim | Задача | Метод | N eval | Top-1 | Top-5 | Macro-F1 | MAE (лет) | Within ±2 г. |",
        "| :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in metric_rows:
        if r["split"] == "ablation_test":
            continue
        t1 = f"{r['top1']*100:.2f}%" if r["top1"] != "" else "—"
        t5 = f"{r['top5']*100:.2f}%" if r["top5"] != "" else "—"
        mf1 = f"{r['macro_f1']*100:.2f}%" if r["macro_f1"] != "" else "—"
        mae = f"{r['mae_years']:.2f}" if r["mae_years"] != "" else "—"
        w2 = f"{r['within_2yr']*100:.1f}%" if r["within_2yr"] != "" else "—"
        lines.append(
            f"| `{r['backbone']}` | {r['dim']} | {r['task']} | {r['method']} | {r['n_eval']:,} | "
            f"{t1} | {t5} | {mf1} | {mae} | {w2} |"
        )

    if ablation_summaries:
        lines.extend(
            [
                "",
                "## 2. Абляция: влияние кропа YOLOv8n vs Сырой кадр (Test subset)",
                "",
                "| Backbone | N | Марка Top-1 (YOLO → Raw) | Модель LP Top-1 (YOLO → Raw) | Модель Proto Top-1 (YOLO → Raw) | Год MAE (YOLO → Raw) |",
                "| :--- | ---: | :--- | :--- | :--- | :--- |",
            ]
        )
        for ab in ablation_summaries:
            mk_str = f"{ab['crop_make_lp_top1']*100:.2f}% vs {ab['raw_make_lp_top1']*100:.2f}%"
            md_str = f"{ab['crop_model_lp_top1']*100:.2f}% vs {ab['raw_model_lp_top1']*100:.2f}%"
            pr_str = f"{ab['crop_model_proto_top1']*100:.2f}% vs {ab['raw_model_proto_top1']*100:.2f}%"
            yr_str = (
                f"{ab['crop_year_mae']:.2f} vs {ab['raw_year_mae']:.2f}"
                if not math.isnan(ab["crop_year_mae"])
                else "—"
            )
            lines.append(f"| `{ab['backbone']}` | {ab['n_matched']:,} | {mk_str} | {md_str} | {pr_str} | {yr_str} |")

    lines.extend(
        [
            "",
            f"## 3. Визуализации для диплома (лучший backbone: `{best_backbone}`, t-SNE Silhouette={tsne_sil:.3f})",
            "",
            "- [Кривая объяснённой дисперсии PCA](figures/feature_pca_variance.png)",
            "- [2D t-SNE проекция Топ-10 моделей](figures/feature_tsne_models.png)",
            "- [Регрессия года выпуска (Predicted vs True)](figures/feature_year_regression.png)",
        ]
    )
    if montage_ok:
        ret_bb = retrieval_info["backbone"]
        ret_mod = retrieval_info["model_knn_top1"] * 100
        ret_mk = retrieval_info["make_knn_top1"] * 100
        lines.append(
            f"- [Визуальный поиск Top-5 похожих авто](figures/feature_retrieval_montage.png) "
            f"(`{ret_bb}` — лучший Cosine kNN: {ret_mod:.2f}% Model / {ret_mk:.2f}% Make)"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate extracted car feature bundles for thesis baseline.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=BASE_DIR / "data_manifests" / "local_vmmrdb_manifest.csv",
    )
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument("--ablation-raw", type=Path, nargs="*", default=None)
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\Projects\bitirme\data"))
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "reports")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fuse",
        action="store_true",
        help="Also evaluate L2-concatenated fusion across all --features bundles.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_all(
        manifest_path=args.manifest,
        feature_paths=args.features,
        ablation_paths=args.ablation_raw,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        seed=args.seed,
        fuse=args.fuse,
    )


if __name__ == "__main__":
    main()
