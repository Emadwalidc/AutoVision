"""Build complete 8-class Body Type taxonomy, enrich local manifest, and run Body Type baseline."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate_features_baseline import (  # noqa: E402
    _fuse_bundles,
    eval_cosine_knn,
    eval_nearest_prototype,
    load_feature_bundle,
    train_and_eval_linear_probe,
)

BODY_TYPE_CLASSES: tuple[str, ...] = (
    "Sedan",
    "SUV",
    "Pickup",
    "Minivan",
    "Coupe",
    "Hatchback",
    "Wagon",
    "Convertible",
)

_BODY_GROUPS: dict[str, tuple[str, ...]] = {
    "SUV": (
        "mazda__tribute", "mercedes-benz__gl-class", "mercedes-benz__gla-class",
        "mercedes-benz__glk-class", "mercedes-benz__m-class", "mercury__mariner",
        "mercury__mountaineer", "mini__cooper_countryman", "mitsubishi__endeavor",
        "mitsubishi__montero", "mitsubishi__outlander", "nissan__armada",
        "nissan__juke", "nissan__murano", "nissan__pathfinder", "nissan__rogue",
        "nissan__x-trail", "nissan__xterra", "oldsmobile__bravada", "peugeot__3008",
        "plymouth__trailduster", "pontiac__aztek", "pontiac__torrent",
        "porsche__cayenne", "renault__captur", "saab__9-7", "saturn__outlook",
        "saturn__vue", "subaru__crosstrek", "subaru__forester", "subaru__tribeca",
        "suzuki__grand_vitara", "suzuki__samurai", "suzuki__sidekick",
        "suzuki__vitara", "suzuki__x90", "suzuki__xl7", "toyota__4runner",
        "toyota__fj40", "toyota__fjcruiser", "toyota__fortuner",
        "toyota__highlander", "toyota__landcruiser", "toyota__rav4",
        "toyota__sequoia", "toyota__venza", "volkswagen__tiguan",
        "volkswagen__touareg", "volvo__xc60", "volvo__xc90",
        "willys__cj2a", "willys__cj3b",
    ),
    "Pickup": (
        "mitsubishi__mighty max", "mitsubishi__raider", "nissan__720",
        "nissan__d21_hardbody", "nissan__frontier", "nissan__titan",
        "ram__1500", "ram__2500", "ram__3500", "subaru__baja", "subaru__brat",
        "suzuki__equator", "toyota__1-ton truck", "toyota__hilux",
        "toyota__pickup", "toyota__t100", "toyota__tacoma", "toyota__tundra",
    ),
    "Minivan": (
        "mercedes-benz__r-class", "mercedes-benz__sprinter", "mercury__villager",
        "mitsubishi__chariot", "mitsubishi__delica", "nissan__nv200",
        "nissan__presage", "nissan__quest", "oldsmobile__silhouette",
        "opel__vivaro", "plymouth__voyager", "pontiac__montana", "saturn__relay",
        "toyota__estima", "toyota__hiace", "toyota__previa", "toyota__sienna",
        "volkswagen__bus", "volkswagen__camper", "volkswagen__eurovan",
        "volkswagen__routan", "volkswagen__vanagon",
    ),
    "Wagon": (
        "peugeot__407_wagon", "saab__9-2", "subaru__outback",
        "volkswagen__squareback", "volvo__145_wagon", "volvo__245",
        "volvo__v40", "volvo__v50", "volvo__v70", "volvo__v90", "volvo__xc70",
    ),
    "Convertible": (
        "mercedes-benz__sl-class", "mercedes-benz__slk-class", "mg__mgb",
        "mg__midget", "plymouth__prowler", "pontiac__solstice", "porsche__914",
        "porsche__boxster", "rolls-royce__corniche", "saturn__sky",
        "triumph__spitfire", "triumph__tr", "volkswagen__cabrio",
        "volkswagen__dunebuggy", "volkswagen__eos",
    ),
    "Coupe": (
        "mazda__rx7", "mazda__rx8", "mercedes-benz__cl-class",
        "mercedes-benz__clk-class", "mercedes-benz__slr", "mercury__capri",
        "mercury__cougar", "mercury__cyclone", "mitsubishi__3000gt",
        "mitsubishi__eclipse", "mitsubishi__starion", "nash__metropolitan",
        "nissan__200sx", "nissan__240sx", "nissan__300zx", "nissan__350z",
        "nissan__370z", "nissan__gtr", "nissan__skyline", "oldsmobile__442",
        "oldsmobile__starfire", "oldsmobile__toronado", "opel__gt",
        "plymouth__barracuda", "plymouth__duster", "plymouth__gtx",
        "plymouth__laser", "plymouth__roadrunner", "plymouth__scamp",
        "pontiac__fiero", "pontiac__firebird", "pontiac__gto",
        "pontiac__transam", "porsche__911", "porsche__912", "porsche__917",
        "porsche__924", "porsche__928", "porsche__944", "porsche__cayman",
        "saturn__sc", "scion__frs", "scion__tc", "subaru__brz", "subaru__svx",
        "sunbeam__rapier", "toyota__celica", "toyota__mr2", "toyota__paseo",
        "toyota__solara", "toyota__supra", "triumph__gt6",
        "volkswagen__corrado", "volkswagen__karmannghia",
        "volkswagen__scirocco", "volvo__142", "volvo__242", "volvo__780",
        "volvo__c70",
    ),
    "Hatchback": (
        "mazda__speed3", "mercedes-benz__b-class", "mini__austin",
        "mini__cooper", "mitsubishi__colt", "nissan__cube", "nissan__leaf",
        "nissan__pulsar", "opel__astra", "plymouth__colt", "plymouth__horizon",
        "plymouth__sundance", "pontiac__g39", "pontiac__vibe", "pontiac__wave",
        "saab__9-6", "saturn__astra", "scion__iq", "scion__xa", "scion__xb",
        "scion__xd", "smart__fortwo", "subaru__360", "suzuki__reno",
        "suzuki__swift", "suzuki__sx4", "toyota__matrix", "toyota__prius",
        "toyota__starlet", "toyota__yaris", "volkswagen__beetle",
        "volkswagen__golf", "volkswagen__gti", "volkswagen__r32", "volvo__c30",
    ),
    "Sedan": (
        "mazda__speed6", "mercedes-benz__190-class", "mercedes-benz__220d",
        "mercedes-benz__240d", "mercedes-benz__300d", "mercedes-benz__c-class",
        "mercedes-benz__cla-class", "mercedes-benz__cls-class",
        "mercedes-benz__e-class", "mercedes-benz__s-class", "mercury__comet",
        "mercury__grandmarquis", "mercury__marauder", "mercury__milan",
        "mercury__montclair", "mercury__montego", "mercury__monterey",
        "mercury__mystique", "mercury__parklane", "mercury__sable",
        "mercury__topaz", "mercury__tracer", "mitsubishi__diamante",
        "mitsubishi__galant", "mitsubishi__lancer", "mitsubishi__mirage",
        "morris__minor", "nissan__altima", "nissan__cima", "nissan__maxima",
        "nissan__sentra", "nissan__stanza", "nissan__versa",
        "oldsmobile__achieva", "oldsmobile__alero", "oldsmobile__aurora",
        "oldsmobile__cutlass", "oldsmobile__eightyeight", "oldsmobile__intrigue",
        "oldsmobile__ninetyeight", "oldsmobile__omega", "opel__kadett",
        "packard__super8", "plymouth__acclaim", "plymouth__belvedere",
        "plymouth__breeze", "plymouth__cranbrook", "plymouth__deluxe",
        "plymouth__fury", "plymouth__neon", "plymouth__reliant",
        "plymouth__satellite", "plymouth__savoy", "plymouth__valiant",
        "plymouth__volare", "pontiac__bonneville", "pontiac__catalina",
        "pontiac__chieftain", "pontiac__g5", "pontiac__g6", "pontiac__g8",
        "pontiac__grandam", "pontiac__grandprix", "pontiac__grandville",
        "pontiac__lemans", "pontiac__parisienne", "pontiac__pursuit",
        "pontiac__starchief", "pontiac__sunbird", "pontiac__sunfire",
        "pontiac__tempest", "pontiac__ventura", "porsche__panamera",
        "rambler__american", "rambler__rogue", "rolls-royce__silvershadow",
        "rolls-royce__silverspur", "saab__9-3", "saab__9-5", "saab__900",
        "saturn__aura", "saturn__ion", "saturn__l-series", "saturn__s-series",
        "studebaker__champion", "studebaker__commander", "studebaker__lark",
        "subaru__gl10", "subaru__impreza", "subaru__legacy", "subaru__wrx",
        "suzuki__aerio", "suzuki__esteem", "suzuki__forenza", "suzuki__kizashi",
        "suzuki__verona", "tesla__s", "toyota__aristo", "toyota__avalon",
        "toyota__camry", "toyota__corolla", "toyota__cressida", "toyota__echo",
        "toyota__tercel", "toyota__tiara", "triumph__vitesse",
        "volkswagen__atlantic", "volkswagen__cc", "volkswagen__fox",
        "volkswagen__jetta", "volkswagen__passat", "volkswagen__phaeton",
        "volvo__164", "volvo__240", "volvo__244", "volvo__540", "volvo__670",
        "volvo__740", "volvo__760", "volvo__840", "volvo__850", "volvo__940",
        "volvo__960", "volvo__c40", "volvo__pv444", "volvo__s40", "volvo__s60", "volvo__s70",
        "volvo__s80", "volvo__s90",
    ),
}

MODEL_BODY_TYPE: dict[str, str] = {}
for _btype, _models in _BODY_GROUPS.items():
    for _cls in _models:
        if _cls in MODEL_BODY_TYPE:
            raise ValueError(f"Duplicate class in _BODY_GROUPS: {_cls}")
        MODEL_BODY_TYPE[_cls] = _btype


def resolve_body_type(
    canonical_class: str,
    raw_model: str,
    body_hint: str,
    raw_year: int | str,
    exclude_reason: str,
) -> str:
    """Resolves exact Body Type from explicit overrides and complete 342-class mapping."""
    if exclude_reason != "":
        return ""
    if body_hint in ("Wagon", "Convertible"):
        return body_hint
    if raw_model.strip().lower() in ("sw2", "lw200", "lw300"):
        return "Wagon"
    if canonical_class == "mercury__monterey" and int(raw_year) >= 2004:
        return "Minivan"
    return MODEL_BODY_TYPE[canonical_class]


def verify_classes_coverage(classes_csv: Path) -> None:
    """Fails loudly if any class in local_vmmrdb_classes.csv is unmapped."""
    with open(classes_csv, "r", encoding="utf-8", newline="") as f:
        cls_names = [row["canonical_class"] for row in csv.DictReader(f)]
    missing = sorted(c for c in cls_names if c not in MODEL_BODY_TYPE)
    if missing:
        raise KeyError(f"Unmapped canonical classes ({len(missing)}): {missing[:10]}")


def build_multitask_manifest(
    manifest_path: Path,
    classes_csv: Path,
    out_manifest: Path,
    out_dir: Path,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Enriches local_vmmrdb_manifest.csv with body_type and outputs distribution table & chart."""
    verify_classes_coverage(classes_csv)

    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        in_fields = list(reader.fieldnames or [])
        out_fields = in_fields if "body_type" in in_fields else [*in_fields, "body_type"]
        rows: list[dict[str, str]] = []

        for row in reader:
            row_out = dict(row)
            row_out["body_type"] = resolve_body_type(
                canonical_class=row["canonical_class"],
                raw_model=row["raw_model"],
                body_hint=row.get("body_hint", ""),
                raw_year=row["raw_year"],
                exclude_reason=row.get("exclude_reason", ""),
            )
            if str(row.get("is_clean", "")).strip().lower() == "true" and not row_out["body_type"]:
                raise ValueError(f"Empty body_type on clean row: {row['rel_path']}")
            rows.append(row_out)

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    clean_rows = [r for r in rows if str(r.get("is_clean", "")).strip().lower() == "true"]
    n_clean = len(clean_rows)
    by_bt: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in clean_rows:
        by_bt[r["body_type"]].append(r)

    dist_rows: list[dict[str, Any]] = []
    for bt in BODY_TYPE_CLASSES:
        items = by_bt.get(bt, [])
        sp_cnt = Counter(r["split"] for r in items)
        cls_cnt = Counter(r["canonical_class"] for r in items)
        examples = ", ".join(c for c, _ in cls_cnt.most_common(4))
        dist_rows.append(
            {
                "body_type": bt,
                "clean_imgs": len(items),
                "clean_pct": round(100.0 * len(items) / max(n_clean, 1), 2),
                "train_imgs": sp_cnt["train"],
                "val_imgs": sp_cnt["val"],
                "test_imgs": sp_cnt["test"],
                "few_shot_imgs": sp_cnt["few_shot"],
                "canonical_classes": len(cls_cnt),
                "example_models": examples,
            }
        )
    dist_rows.sort(key=lambda r: (-r["clean_imgs"], r["body_type"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    dist_csv = out_dir / "body_type_distribution.csv"
    with open(dist_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "body_type",
                "clean_imgs",
                "clean_pct",
                "train_imgs",
                "val_imgs",
                "test_imgs",
                "few_shot_imgs",
                "canonical_classes",
                "example_models",
            ],
        )
        writer.writeheader()
        writer.writerows(dist_rows)

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_body_type_distribution(dist_rows, n_clean, figures_dir / "body_type_distribution.png")

    print(f"[1/2] Saved {out_manifest} ({len(rows):,} rows, {n_clean:,} clean)")
    print(f"[1/2] Saved {dist_csv} & body_type_distribution.png")
    return rows, dist_rows


def _plot_body_type_distribution(
    dist_rows: list[dict[str, Any]],
    n_clean: int,
    out_path: Path,
) -> None:
    labels = [r["body_type"] for r in dist_rows]
    counts = [r["clean_imgs"] for r in dist_rows]
    pcts = [r["clean_pct"] for r in dist_rows]

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=150)
    colors = plt.get_cmap("tab10")(np.arange(len(labels)) % 10)
    bars = ax.bar(labels, counts, color=colors, edgecolor="#333333", linewidth=0.6)

    max_cnt = max(counts) if counts else 1
    ax.set_ylim(0, max_cnt * 1.16)
    for bar, cnt, pct in zip(bars, counts, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max_cnt * 0.015,
            f"{cnt:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_title(
        f"Body Type Distribution across {n_clean:,} Clean VMMRdb Images (8 Categories)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Number of Clean Images")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _scores_top1_top3_f1(
    scores: np.ndarray,
    y_true: np.ndarray,
    num_classes: int,
) -> tuple[dict[str, float], np.ndarray]:
    k3 = min(3, scores.shape[1])
    top3_idx = np.argsort(-scores, axis=1)[:, :k3]
    y_pred = top3_idx[:, 0]
    top1 = float(np.mean(y_pred == y_true))
    top3 = float(np.mean([yt in row for yt, row in zip(y_true, top3_idx)]))
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0)
    )
    return {"top1": top1, "top3": top3, "macro_f1": macro_f1}, y_pred


def _cosine_knn_scores(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    num_classes: int,
    k: int = 5,
    chunk_size: int = 512,
) -> np.ndarray:
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
    return scores


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    backbone_name: str,
    top1: float,
    macro_f1: float,
    out_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, 100.0 * cm / np.maximum(row_sums, 1), 0.0)

    fig, ax = plt.subplots(figsize=(8.6, 7.2), dpi=150)
    im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=100.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall (%)")

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_pct[i, j]
            cnt = cm[i, j]
            txt_color = "white" if val >= 55.0 else "#111111"
            label_str = f"{val:.1f}%\n({cnt})" if val >= 1.0 or i == j else ""
            if label_str:
                ax.text(j, i, label_str, ha="center", va="center", color=txt_color, fontsize=7.5)

    ax.set_title(
        f"Body Type Confusion Matrix ({backbone_name} LP | Top-1={top1*100:.2f}%, F1={macro_f1*100:.2f}%)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("True Body Type")
    ax.set_xlabel("Predicted Body Type")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def evaluate_body_type_baseline(
    multitask_rows: list[dict[str, str]],
    features_dir: Path,
    out_dir: Path,
    epochs: int = 20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Evaluates OpenCLIP, DINOv2, and Fusion embeddings on 8-class Body Type."""
    clip_path = features_dir / "features_openclip_vitb16.npz"
    dino_path = features_dir / "features_dinov2_vitb14.npz"
    if not (clip_path.is_file() and dino_path.is_file()):
        print("[2/2] Skipping Body Type feature baseline (feature .npz bundles not found).")
        return []

    lookup = {
        r["rel_path"].replace("\\", "/"): r
        for r in multitask_rows
        if str(r.get("is_clean", "")).strip().lower() == "true"
    }

    b_clip = load_feature_bundle(clip_path)
    b_dino = load_feature_bundle(dino_path)
    b_fuse = _fuse_bundles([b_clip, b_dino], "fusion_openclip_dinov2")

    bundles = [
        ("openclip_vitb16", b_clip),
        ("dinov2_vitb14", b_dino),
        ("fusion_openclip_dinov2", b_fuse),
    ]

    bt_list = list(BODY_TYPE_CLASSES)
    bt_to_id = {b: i for i, b in enumerate(bt_list)}
    n_cls = len(bt_list)

    metric_rows: list[dict[str, Any]] = []
    best_lp_top1 = -1.0
    best_cm_payload: dict[str, Any] | None = None

    for bb_name, bundle in bundles:
        rel_paths = [str(p).replace("\\", "/") for p in bundle["rel_paths"]]
        keep_idx: list[int] = []
        meta: list[dict[str, str]] = []
        for i, rp in enumerate(rel_paths):
            rec = lookup.get(rp)
            if rec is not None:
                keep_idx.append(i)
                meta.append(rec)

        X = bundle["embeddings"][keep_idx].astype(np.float32)
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

        tr_idx = [i for i, r in enumerate(meta) if r["split"] == "train"]
        va_idx = [i for i, r in enumerate(meta) if r["split"] == "val"]
        te_idx = [i for i, r in enumerate(meta) if r["split"] == "test"]

        X_tr, X_va, X_te = X[tr_idx], X[va_idx], X[te_idx]
        y_tr = np.array([bt_to_id[meta[i]["body_type"]] for i in tr_idx], dtype=np.int64)
        y_va = np.array([bt_to_id[meta[i]["body_type"]] for i in va_idx], dtype=np.int64)
        y_te = np.array([bt_to_id[meta[i]["body_type"]] for i in te_idx], dtype=np.int64)

        proto_m, protos = eval_nearest_prototype(X_tr, y_tr, X_te, y_te, n_cls)
        proto_res, _ = _scores_top1_top3_f1(X_te @ protos.T, y_te, n_cls)
        assert abs(proto_res["top1"] - proto_m["top1"]) < 1e-6

        knn_m = eval_cosine_knn(X_tr, y_tr, X_te, y_te, n_cls, k=5)
        knn_scores = _cosine_knn_scores(X_tr, y_tr, X_te, n_cls, k=5)
        knn_res, _ = _scores_top1_top3_f1(knn_scores, y_te, n_cls)
        assert abs(knn_res["top1"] - knn_m["top1"]) < 1e-6

        lp_m, lp_head = train_and_eval_linear_probe(
            X_tr, y_tr, X_va, y_va, X_te, y_te, n_cls, epochs=epochs, seed=seed
        )
        with torch.no_grad():
            lp_scores = lp_head(torch.from_numpy(X_te)).numpy()
        lp_res, lp_preds = _scores_top1_top3_f1(lp_scores, y_te, n_cls)
        assert abs(lp_res["top1"] - lp_m["top1"]) < 1e-6

        for method, res in [
            ("Nearest Prototype", proto_res),
            ("Cosine kNN (k=5)", knn_res),
            ("Linear Probe", lp_res),
        ]:
            metric_rows.append(
                {
                    "backbone": bb_name,
                    "dim": X.shape[1],
                    "method": method,
                    "split": "test",
                    "n_eval": len(te_idx),
                    "top1": round(res["top1"], 4),
                    "top3": round(res["top3"], 4),
                    "macro_f1": round(res["macro_f1"], 4),
                }
            )

        if lp_res["top1"] > best_lp_top1:
            best_lp_top1 = lp_res["top1"]
            best_cm_payload = {
                "backbone": bb_name,
                "y_true": y_te,
                "y_pred": lp_preds,
                "top1": lp_res["top1"],
                "macro_f1": lp_res["macro_f1"],
            }

    metrics_csv = out_dir / "body_type_baseline_metrics.csv"
    with open(metrics_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["backbone", "dim", "method", "split", "n_eval", "top1", "top3", "macro_f1"],
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    assert best_cm_payload is not None
    cm_path = out_dir / "figures" / "body_type_confusion_matrix.png"
    _plot_confusion_matrix(
        y_true=best_cm_payload["y_true"],
        y_pred=best_cm_payload["y_pred"],
        class_names=bt_list,
        backbone_name=best_cm_payload["backbone"],
        top1=best_cm_payload["top1"],
        macro_f1=best_cm_payload["macro_f1"],
        out_path=cm_path,
    )
    print(f"[2/2] Saved {metrics_csv} & {cm_path}")
    return metric_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Body Type taxonomy and evaluate baseline.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=BASE_DIR / "data_manifests" / "local_vmmrdb_manifest.csv",
    )
    parser.add_argument(
        "--classes-csv",
        type=Path,
        default=BASE_DIR / "data_manifests" / "local_vmmrdb_classes.csv",
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=BASE_DIR / "data_manifests" / "local_vmmrdb_multitask_manifest.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "reports")
    parser.add_argument("--features-dir", type=Path, default=BASE_DIR / "features")
    parser.add_argument(
        "--eval-features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, _ = build_multitask_manifest(
        manifest_path=args.manifest,
        classes_csv=args.classes_csv,
        out_manifest=args.out_manifest,
        out_dir=args.out_dir,
    )
    if args.eval_features:
        evaluate_body_type_baseline(
            multitask_rows=rows,
            features_dir=args.features_dir,
            out_dir=args.out_dir,
            epochs=args.epochs,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
