"""Audit, normalize, deduplicate, and index the local VMMRdb dataset.

Reads raw vehicle folders (<make>_<model>_<year>) in read-only mode, normalizes
taxonomies, detects corrupt/tiny/duplicate images via 64-bit dHash, creates
leakage-free train/val/test/few_shot splits, optionally runs a YOLOv8n exterior
quality probe, and writes manifests and a Markdown audit report.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

MAKE_ALIASES: dict[str, str] = {
    "mercedes benz": "mercedes-benz",
    "rollsroyce": "rolls-royce",
}

EXCLUDE_NON_CAR: set[tuple[str, str]] = {
    ("suzuki", "boulevard"),
    ("polaris", "slingshot"),
    ("volvo", "vnl"),
    ("mitsubishi", "fuso"),
    ("toyota", "flatbed"),
    ("studebaker", "flatbed"),
}

EXCLUDE_AMBIGUOUS: set[tuple[str, str]] = {
    ("toyota", "scion"),
    ("nissan", "truck"),
    ("toyota", "van"),
    ("mercedes-benz", "ht350"),
}

MERCEDES_MODEL_MAP: dict[str, str] = {
    **{m: "190-class" for m in ("190", "190d", "e190")},
    **{m: "b-class" for m in ("b200", "b200t")},
    **{
        m: "c-class"
        for m in ("c220", "c230", "c240", "c250", "c280", "c300", "c32", "c320", "c350", "c55")
    },
    **{m: "cl-class" for m in ("cl500", "cl550", "cl600")},
    "cla250": "cla-class",
    **{m: "clk-class" for m in ("clk240", "clk320", "clk350", "clk430", "clk500", "clk55", "clk550")},
    **{m: "cls-class" for m in ("cls500", "cls550")},
    **{
        m: "e-class"
        for m in (
            "e300", "e320", "e350", "e420", "e430", "e500", "e55", "e550", "e63",
            "cd300", "ce300", "td300",
        )
    },
    **{m: "gl-class" for m in ("gl320", "gl350", "gl450", "gl550")},
    **{m: "gla-class" for m in ("gla250", "gla45")},
    "glk350": "glk-class",
    **{m: "m-class" for m in ("ml320", "ml350", "ml430", "ml500", "ml55", "ml550")},
    **{m: "r-class" for m in ("r350", "r500")},
    **{
        m: "s-class"
        for m in (
            "s280", "s320", "s350", "s420", "s430", "s500", "s55", "s550", "s600", "s65",
            "sd300", "sdl300", "se280", "se380", "se400", "se450",
            "sel300", "sel400", "sel420", "sel450", "sel600",
        )
    },
    **{
        m: "sl-class"
        for m in ("sl300", "sl320", "sl380", "sl450", "sl500", "sl55", "sl550", "sl560", "sl600", "slc450")
    },
    **{m: "slk-class" for m in ("slk230", "slk280", "slk300", "slk320", "slk350", "slk55")},
}

MODEL_ALIASES: dict[tuple[str, str], str] = {
    # Toyota
    ("toyota", "camry_le"): "camry",
    ("toyota", "camry_se"): "camry",
    ("toyota", "camry_xle"): "camry",
    ("toyota", "camry_solara"): "solara",
    ("toyota", "camry_solara_convertible"): "solara",
    ("toyota", "solara_convertable"): "solara",
    ("toyota", "solara_convertible"): "solara",
    ("toyota", "corolla_ce"): "corolla",
    ("toyota", "corolla_dx"): "corolla",
    ("toyota", "corolla_le"): "corolla",
    ("toyota", "corolla_s"): "corolla",
    ("toyota", "corolla_wagon"): "corolla",
    ("toyota", "celica_convertible"): "celica",
    # Volkswagen
    ("volkswagen", "bug"): "beetle",
    ("volkswagen", "newbeetle"): "beetle",
    ("volkswagen", "superbeetle"): "beetle",
    ("volkswagen", "cabriolet"): "cabrio",
    ("volkswagen", "jetta_wagon"): "jetta",
    ("volkswagen", "passat_wagon"): "passat",
    ("volkswagen", "rabbit"): "golf",
    # Saturn
    ("saturn", "view"): "vue",
    ("saturn", "sl"): "s-series",
    ("saturn", "sl1"): "s-series",
    ("saturn", "sl2"): "s-series",
    ("saturn", "sl2_wagon"): "s-series",
    ("saturn", "sw2"): "s-series",
    ("saturn", "sc1"): "sc",
    ("saturn", "sc2"): "sc",
    ("saturn", "l100"): "l-series",
    ("saturn", "l200"): "l-series",
    ("saturn", "l300"): "l-series",
    ("saturn", "ls"): "l-series",
    ("saturn", "lw200"): "l-series",
    ("saturn", "lw300"): "l-series",
    # Volvo
    ("volvo", "cx90"): "xc90",
    ("volvo", "c70_convertible"): "c70",
    ("volvo", "240_wagon"): "240",
    ("volvo", "740_wagon"): "740",
    ("volvo", "760_wagon"): "760",
    ("volvo", "850_wagon"): "850",
    ("volvo", "940_wagon"): "940",
    ("volvo", "960_wagon"): "960",
    # Other makes
    ("subaru", "impreza_wagon"): "impreza",
    ("mitsubishi", "eclipse_convertible"): "eclipse",
    ("saab", "9-3_convertible"): "9-3",
    ("suzuki", "esteem_wagon"): "esteem",
    ("porsche", "997"): "911",
    ("porsche", "930"): "911",
    ("mg", "mgb_mk2"): "mgb",
    ("mg", "mgb_mk5"): "mgb",
    ("nissan", "d21"): "d21_hardbody",
    ("nissan", "hardbody"): "d21_hardbody",
}

MODEL_MIN_YEAR: dict[tuple[str, str], int] = {
    ("toyota", "camry"): 1982,
    ("toyota", "corolla"): 1966,
    ("toyota", "prius"): 1997,
    ("toyota", "mr2"): 1984,
    ("toyota", "supra"): 1978,
    ("toyota", "4runner"): 1984,
    ("toyota", "avalon"): 1994,
    ("toyota", "sienna"): 1997,
    ("toyota", "tacoma"): 1995,
    ("toyota", "rav4"): 1994,
    ("mercedes-benz", "c-class"): 1993,
    ("mercedes-benz", "slk-class"): 1996,
    ("mercedes-benz", "m-class"): 1997,
    ("mercedes-benz", "clk-class"): 1997,
    ("mercedes-benz", "cls-class"): 2004,
    ("mercedes-benz", "gl-class"): 2006,
    ("mercedes-benz", "glk-class"): 2008,
    ("mercedes-benz", "gla-class"): 2013,
    ("volkswagen", "jetta"): 1979,
    ("volkswagen", "golf"): 1974,
    ("volkswagen", "passat"): 1973,
    ("volkswagen", "tiguan"): 2007,
    ("volkswagen", "touareg"): 2002,
    ("nissan", "altima"): 1992,
    ("nissan", "maxima"): 1981,
    ("nissan", "sentra"): 1982,
    ("nissan", "murano"): 2002,
    ("nissan", "rogue"): 2007,
    ("nissan", "versa"): 2006,
    ("volvo", "s60"): 2000,
    ("volvo", "xc60"): 2008,
    ("volvo", "xc90"): 2002,
    ("volvo", "s40"): 1995,
    ("porsche", "911"): 1963,
    ("porsche", "boxster"): 1996,
    ("porsche", "cayenne"): 2002,
    ("mazda", "rx8"): 2003,
    ("mazda", "speed3"): 2006,
    ("mini", "cooper"): 1959,
}

MANIFEST_FIELDS = [
    "rel_path",
    "raw_folder",
    "raw_make",
    "raw_model",
    "raw_year",
    "make",
    "model",
    "canonical_class",
    "body_hint",
    "year_valid",
    "width",
    "height",
    "file_size",
    "dhash",
    "is_corrupt",
    "is_tiny",
    "is_duplicate",
    "is_cross_class_dup",
    "exclude_reason",
    "is_clean",
    "split",
]

CLASS_FIELDS = [
    "canonical_class",
    "make",
    "model",
    "raw_models_merged",
    "total_imgs",
    "clean_imgs",
    "train_imgs",
    "val_imgs",
    "test_imgs",
    "few_shot_imgs",
    "valid_year_min",
    "valid_year_max",
]


def normalize_folder_label(folder_name: str) -> dict[str, Any]:
    """Parses `<make>_<model>_<year>` and returns normalized taxonomy metadata."""
    parts = folder_name.split("_")
    if len(parts) < 3 or not (parts[-1].isdigit() and len(parts[-1]) == 4):
        raise ValueError(f"Unexpected folder naming format: {folder_name!r}")

    raw_make = parts[0]
    raw_model = "_".join(parts[1:-1])
    raw_year = int(parts[-1])

    make_lower = raw_make.strip().lower()
    norm_make = MAKE_ALIASES.get(make_lower, make_lower)
    raw_model_lower = raw_model.strip().lower()

    if raw_model_lower.endswith("_wagon"):
        body_hint = "Wagon"
    elif raw_model_lower.endswith("_convertible") or raw_model_lower.endswith("_convertable"):
        body_hint = "Convertible"
    elif raw_model_lower in ("cabrio", "cabriolet"):
        body_hint = "Convertible"
    else:
        body_hint = ""

    if (norm_make, raw_model_lower) in EXCLUDE_NON_CAR:
        exclude_reason = "non_car"
    elif (norm_make, raw_model_lower) in EXCLUDE_AMBIGUOUS:
        exclude_reason = "ambiguous_label"
    else:
        exclude_reason = ""

    if norm_make == "mercedes-benz":
        mb_base = raw_model_lower[:-6] if raw_model_lower.endswith("_wagon") else raw_model_lower
        canonical_model = MERCEDES_MODEL_MAP.get(mb_base, mb_base)
    else:
        canonical_model = MODEL_ALIASES.get((norm_make, raw_model_lower), raw_model_lower)

    min_year = MODEL_MIN_YEAR.get((norm_make, canonical_model), 1950)
    year_valid = (1950 <= raw_year <= 2016) and (raw_year >= min_year)

    return {
        "raw_folder": folder_name,
        "raw_make": raw_make,
        "raw_model": raw_model,
        "raw_year": raw_year,
        "make": norm_make,
        "model": canonical_model,
        "canonical_class": f"{norm_make}__{canonical_model}",
        "body_hint": body_hint,
        "year_valid": year_valid,
        "exclude_reason": exclude_reason,
    }


def compute_dhash(img: Image.Image) -> str:
    """Computes a 64-bit difference hash (dHash) as a 16-character hex string."""
    gray = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).reshape(-1)
    return np.packbits(bits.astype(np.uint8)).tobytes().hex()


def audit_image(path: Path) -> dict[str, Any]:
    """Reads image dimensions, file size, and 64-bit dHash, catching corrupt files."""
    try:
        file_size = path.stat().st_size
    except OSError:
        return {
            "width": 0,
            "height": 0,
            "file_size": 0,
            "dhash": "",
            "is_corrupt": True,
            "is_tiny": False,
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as img:
                width, height = img.size
                if width <= 0 or height <= 0:
                    raise ValueError("Invalid non-positive image dimensions")
                dhash = compute_dhash(img)
        return {
            "width": width,
            "height": height,
            "file_size": file_size,
            "dhash": dhash,
            "is_corrupt": False,
            "is_tiny": min(width, height) < 80,
        }
    except Exception:
        return {
            "width": 0,
            "height": 0,
            "file_size": file_size,
            "dhash": "",
            "is_corrupt": True,
            "is_tiny": False,
        }


def deduplicate_records(records: list[dict[str, Any]]) -> None:
    """Marks exact/near duplicates within classes and cross-class dHash collisions."""
    for rec in records:
        rec["is_duplicate"] = False
        rec["is_cross_class_dup"] = False

    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec["exclude_reason"] or rec["is_corrupt"] or rec["is_tiny"]:
            continue
        if rec["dhash"]:
            by_hash[rec["dhash"]].append(rec)

    for group in by_hash.values():
        if len(group) <= 1:
            continue
        classes = {r["canonical_class"] for r in group}
        if len(classes) > 1:
            for r in group:
                r["is_cross_class_dup"] = True
        else:
            group_sorted = sorted(
                group,
                key=lambda r: (-r["width"] * r["height"], -r["file_size"], r["rel_path"]),
            )
            for r in group_sorted[1:]:
                r["is_duplicate"] = True


def assign_splits(
    records: list[dict[str, Any]],
    min_class_images: int = 15,
    seed: int = 42,
) -> None:
    """Assigns leakage-free stratified splits (`train`/`val`/`test`/`few_shot`/`excluded`)."""
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        is_clean = (
            rec["exclude_reason"] == ""
            and not rec["is_corrupt"]
            and not rec["is_tiny"]
            and not rec["is_duplicate"]
            and not rec["is_cross_class_dup"]
        )
        rec["is_clean"] = is_clean
        if is_clean:
            by_class[rec["canonical_class"]].append(rec)
        else:
            rec["split"] = "excluded"

    rng = random.Random(seed)
    for cls_name in sorted(by_class):
        items = sorted(by_class[cls_name], key=lambda r: r["rel_path"])
        n = len(items)
        if n < min_class_images:
            for r in items:
                r["split"] = "few_shot"
            continue

        rng.shuffle(items)
        n_val = max(1, round(n * 0.10))
        n_test = max(1, round(n * 0.10))
        if n - n_val - n_test < 1:
            n_val = 1 if n >= 3 else 0
            n_test = 1 if n >= 2 else 0

        for idx, r in enumerate(items):
            if idx < n_val:
                r["split"] = "val"
            elif idx < n_val + n_test:
                r["split"] = "test"
            else:
                r["split"] = "train"

    split_hashes: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        if rec["split"] != "excluded" and rec["dhash"]:
            split_hashes[rec["split"]].add(rec["dhash"])

    active_splits = ("train", "val", "test", "few_shot")
    for i, s1 in enumerate(active_splits):
        for s2 in active_splits[i + 1 :]:
            overlap = split_hashes[s1] & split_hashes[s2]
            if overlap:
                raise AssertionError(
                    f"dHash leakage detected between {s1} and {s2}: {len(overlap)} hashes"
                )


def scan_dataset(data_dir: Path, workers: int) -> tuple[list[dict[str, Any]], int, int]:
    """Scans data_dir, parses folder labels, and runs parallel PIL/dHash auditing."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    items: list[tuple[Path, str, dict[str, Any]]] = []
    folder_count = 0
    non_image_files = 0

    for entry in sorted(os.scandir(data_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        folder_count += 1
        meta = normalize_folder_label(entry.name)
        for fentry in sorted(os.scandir(entry.path), key=lambda e: e.name):
            if fentry.is_dir():
                raise ValueError(f"Unexpected nested directory: {fentry.path}")
            ext = Path(fentry.name).suffix.lower()
            if ext not in IMAGE_EXTENSIONS:
                non_image_files += 1
                continue
            rel_path = f"{entry.name}/{fentry.name}"
            items.append((Path(fentry.path), rel_path, meta))

    records: list[dict[str, Any]] = []
    paths = [it[0] for it in items]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        audits = list(pool.map(audit_image, paths, chunksize=64))

    for (_, rel_path, meta), img_info in zip(items, audits):
        rec = {
            "rel_path": rel_path,
            **meta,
            **img_info,
            "is_duplicate": False,
            "is_cross_class_dup": False,
            "is_clean": False,
            "split": "excluded",
        }
        records.append(rec)

    return records, folder_count, non_image_files


def run_yolo_exterior_probe(
    records: list[dict[str, Any]],
    data_dir: Path,
    out_dir: Path,
    yolo_weights: Path,
    sample_size: int,
    seed: int,
) -> dict[str, Any] | None:
    """Runs YOLOv8n vehicle detection on a deterministic sample of clean images."""
    clean_records = sorted(
        (r for r in records if r["is_clean"]),
        key=lambda r: r["rel_path"],
    )
    if sample_size <= 0 or not clean_records:
        return None
    if not yolo_weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")

    cache_dir = out_dir.parent / ".cache" / "ultralytics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(cache_dir)

    from ultralytics import YOLO

    rng = random.Random(seed)
    n_sample = min(sample_size, len(clean_records))
    sampled = rng.sample(clean_records, n_sample)

    model = YOLO(str(yolo_weights))
    counts = Counter()
    multi_vehicle = 0
    no_vehicle_samples: list[str] = []

    batch_size = 32
    for start in range(0, n_sample, batch_size):
        chunk = sampled[start : start + batch_size]
        imgs: list[Image.Image] = []
        areas: list[float] = []
        valid_recs: list[dict[str, Any]] = []
        for rec in chunk:
            img_path = data_dir / rec["rel_path"]
            im = Image.open(img_path).convert("RGB")
            imgs.append(im)
            areas.append(float(im.size[0] * im.size[1]))
            valid_recs.append(rec)

        try:
            results = model.predict(imgs, classes=[2, 5, 7], conf=0.25, verbose=False)
        finally:
            for im in imgs:
                im.close()

        for rec, img_area, res in zip(valid_recs, areas, results):
            if res.boxes is None or len(res.boxes) == 0:
                counts["no_vehicle"] += 1
                if len(no_vehicle_samples) < 15:
                    no_vehicle_samples.append(rec["rel_path"])
                continue

            xyxy = res.boxes.xyxy.cpu().numpy()
            box_areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            ratios = box_areas / max(img_area, 1.0)
            if int(np.sum(ratios > 0.05)) >= 2:
                multi_vehicle += 1

            if float(np.max(ratios)) > 0.08:
                counts["clean_exterior"] += 1
            else:
                counts["small_vehicle"] += 1

    return {
        "sample_size": n_sample,
        "clean_exterior": counts["clean_exterior"],
        "small_vehicle": counts["small_vehicle"],
        "no_vehicle": counts["no_vehicle"],
        "multi_vehicle": multi_vehicle,
        "no_vehicle_samples": no_vehicle_samples,
    }


def build_class_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregates per-canonical-class statistics and valid year ranges."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec["exclude_reason"] == "":
            grouped[rec["canonical_class"]].append(rec)

    rows: list[dict[str, Any]] = []
    for cls_name, items in grouped.items():
        first = items[0]
        raw_models = sorted({r["raw_model"] for r in items})
        split_counts = Counter(r["split"] for r in items)
        clean_items = [r for r in items if r["is_clean"]]
        valid_years = [r["raw_year"] for r in clean_items if r["year_valid"]]

        rows.append(
            {
                "canonical_class": cls_name,
                "make": first["make"],
                "model": first["model"],
                "raw_models_merged": ";".join(raw_models),
                "total_imgs": len(items),
                "clean_imgs": len(clean_items),
                "train_imgs": split_counts["train"],
                "val_imgs": split_counts["val"],
                "test_imgs": split_counts["test"],
                "few_shot_imgs": split_counts["few_shot"],
                "valid_year_min": min(valid_years) if valid_years else "",
                "valid_year_max": max(valid_years) if valid_years else "",
            }
        )

    rows.sort(key=lambda r: (-r["clean_imgs"], r["canonical_class"]))
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Writes a list of dicts to a UTF-8 CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_audit_report(
    report_path: Path,
    data_dir: Path,
    folder_count: int,
    non_image_files: int,
    records: list[dict[str, Any]],
    class_rows: list[dict[str, Any]],
    min_class_images: int,
    yolo_stats: dict[str, Any] | None,
    elapsed_sec: float,
) -> None:
    """Generates a comprehensive Russian Markdown audit report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    raw_makes = {r["raw_make"] for r in records}
    raw_make_models = {(r["raw_make"], r["raw_model"]) for r in records}
    non_excl = [r for r in records if r["exclude_reason"] == ""]
    norm_makes = {r["make"] for r in non_excl}

    excl_reasons = Counter(r["exclude_reason"] for r in records if r["exclude_reason"])
    n_corrupt = sum(1 for r in records if r["is_corrupt"])
    n_tiny = sum(1 for r in records if r["is_tiny"])
    n_dup = sum(1 for r in records if r["is_duplicate"])
    n_cross = sum(1 for r in records if r["is_cross_class_dup"])
    n_clean = sum(1 for r in records if r["is_clean"])

    split_counts = Counter(r["split"] for r in records)
    clean_invalid_years = [r for r in records if r["is_clean"] and not r["year_valid"]]
    invalid_year_examples = sorted(
        {(r["raw_folder"], r["canonical_class"], r["raw_year"]) for r in clean_invalid_years},
        key=lambda x: (x[2], x[0]),
    )[:15]

    train_classes = sum(1 for c in class_rows if c["train_imgs"] > 0)
    few_shot_classes = sum(1 for c in class_rows if c["clean_imgs"] > 0 and c["train_imgs"] == 0)
    empty_classes = sum(1 for c in class_rows if c["clean_imgs"] == 0)

    lines: list[str] = [
        "# Аудит локального датасета (VMMRdb subset)",
        "",
        f"- **Источник:** `{data_dir}` (режим строго read-only)",
        f"- **Время прогона:** {elapsed_sec:.1f} сек",
        f"- **Папок `<make>_<model>_<year>`:** {folder_count:,}",
        f"- **Всего изображений:** {len(records):,} (пропущено не-изображений: {non_image_files})",
        "",
        "## 1. Нормализация таксономии",
        "",
        "| Показатель | До очистки (raw) | После нормализации |",
        "| :--- | ---: | ---: |",
        f"| Уникальных марок | {len(raw_makes)} | {len(norm_makes)} |",
        f"| Уникальных моделей (`make+model`) | {len(raw_make_models)} | {len(class_rows)} |",
        f"| Классов для основного обучения (≥{min_class_images} чистых фото) | — | {train_classes} |",
        f"| Few-shot классов (1..{min_class_images - 1} чистых фото) | — | {few_shot_classes} |",
        f"| Классов с 0 чистых фото | — | {empty_classes} |",
        "",
        "## 2. Фильтрация шума и дедупликация (64-bit dHash)",
        "",
        "| Категория | Количество фото | Доля |",
        "| :--- | ---: | ---: |",
        f"| Всего файлов | {len(records):,} | 100.0% |",
        f"| Исключено: не легковые (`non_car`) | {excl_reasons['non_car']:,} | {100 * excl_reasons['non_car'] / max(len(records), 1):.2f}% |",
        f"| Исключено: размытые/сводные метки (`ambiguous_label`) | {excl_reasons['ambiguous_label']:,} | {100 * excl_reasons['ambiguous_label'] / max(len(records), 1):.2f}% |",
        f"| Битые файлы (`is_corrupt`) | {n_corrupt:,} | {100 * n_corrupt / max(len(records), 1):.2f}% |",
        f"| Слишком мелкие (<80 px, `is_tiny`) | {n_tiny:,} | {100 * n_tiny / max(len(records), 1):.2f}% |",
        f"| Дубликаты внутри класса (`is_duplicate`) | {n_dup:,} | {100 * n_dup / max(len(records), 1):.2f}% |",
        f"| Коллизии dHash между разными классами (`is_cross_class_dup`) | {n_cross:,} | {100 * n_cross / max(len(records), 1):.2f}% |",
        f"| **Итого чистых уникальных фото (`is_clean`)** | **{n_clean:,}** | **{100 * n_clean / max(len(records), 1):.2f}%** |",
        "",
        "## 3. Проверка корректности годов (`year_valid`)",
        "",
        f"- Чистых фото с подозрительным/ошибочным годом (`year_valid=False`): **{len(clean_invalid_years):,}** "
        "(остаются для обучения/теста марки и модели, но исключаются из регрессии года).",
    ]

    if invalid_year_examples:
        lines.append("- Примеры папок с невалидным годом:")
        for folder, cls_name, yr in invalid_year_examples:
            lines.append(f"  - `{folder}` → `{cls_name}` (год `{yr}`)")

    lines.extend(
        [
            "",
            f"## 4. Разбивка на сплиты (без утечки по dHash, порог `min_class_images={min_class_images}`)",
            "",
            "| Сплит | Фото | Доля от всех |",
            "| :--- | ---: | ---: |",
        ]
    )
    for sp in ("train", "val", "test", "few_shot", "excluded"):
        cnt = split_counts[sp]
        lines.append(f"| `{sp}` | {cnt:,} | {100 * cnt / max(len(records), 1):.2f}% |")

    lines.extend(
        [
            "",
            "## 5. Топ-20 канонических классов по числу чистых фото",
            "",
            "| Класс | Всего | Чистых | Train | Val | Test | Годы (валидные) | Объединённые raw-модели |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |",
        ]
    )
    for row in class_rows[:20]:
        yr_range = (
            f"{row['valid_year_min']}–{row['valid_year_max']}"
            if row["valid_year_min"] != ""
            else "—"
        )
        lines.append(
            f"| `{row['canonical_class']}` | {row['total_imgs']:,} | {row['clean_imgs']:,} | "
            f"{row['train_imgs']:,} | {row['val_imgs']:,} | {row['test_imgs']:,} | "
            f"{yr_range} | `{row['raw_models_merged']}` |"
        )

    lines.extend(["", "## 6. Выборочная проверка экстерьера через YOLOv8n", ""])
    if yolo_stats is None:
        lines.append("- Проверка YOLO пропущена (`--yolo-sample 0`).")
    else:
        n_s = yolo_stats["sample_size"]
        lines.extend(
            [
                f"Проверено **{n_s}** случайных чистых изображений (`classes=[car, bus, truck]`, `conf=0.25`):",
                "",
                "| Категория YOLO | Кол-во | Доля |",
                "| :--- | ---: | ---: |",
                f"| Крупный экстерьер (bbox > 8% площади, `clean_exterior`) | {yolo_stats['clean_exterior']} | {100 * yolo_stats['clean_exterior'] / n_s:.1f}% |",
                f"| Мелкое авто на кадре (bbox ≤ 8% площади, `small_vehicle`) | {yolo_stats['small_vehicle']} | {100 * yolo_stats['small_vehicle'] / n_s:.1f}% |",
                f"| Авто не найдено (салон / детали / мусор, `no_vehicle`) | {yolo_stats['no_vehicle']} | {100 * yolo_stats['no_vehicle'] / n_s:.1f}% |",
                f"| Несколько крупных авто в кадре (≥2 боксов > 5% площади) | {yolo_stats['multi_vehicle']} | {100 * yolo_stats['multi_vehicle'] / n_s:.1f}% |",
            ]
        )
        if yolo_stats["no_vehicle_samples"]:
            lines.append("")
            lines.append("Примеры `no_vehicle` кадров (для визуальной проверки):")
            for rel in yolo_stats["no_vehicle_samples"]:
                lines.append(f"- `{rel}`")

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Audit and index local VMMRdb dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\Projects\bitirme\data"))
    parser.add_argument("--out-dir", type=Path, default=base_dir / "data_manifests")
    parser.add_argument("--report-path", type=Path, default=base_dir / "reports" / "local_data_audit.md")
    parser.add_argument("--yolo-weights", type=Path, default=base_dir / "yolov8n.pt")
    parser.add_argument("--yolo-sample", type=int, default=500)
    parser.add_argument("--min-class-images", type=int, default=15)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    print(f"[1/5] Scanning and hashing images in {args.data_dir} (workers={args.workers})...")
    records, folder_count, non_image_files = scan_dataset(args.data_dir, args.workers)
    print(f"      Found {len(records):,} images across {folder_count:,} folders.")

    print("[2/5] Deduplicating images (within-class & cross-class dHash)...")
    deduplicate_records(records)

    print(f"[3/5] Assigning leakage-free splits (min_class_images={args.min_class_images}, seed={args.seed})...")
    assign_splits(records, min_class_images=args.min_class_images, seed=args.seed)

    yolo_stats = None
    if args.yolo_sample > 0:
        print(f"[4/5] Running YOLOv8n exterior quality probe on {args.yolo_sample} clean images...")
        yolo_stats = run_yolo_exterior_probe(
            records=records,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            yolo_weights=args.yolo_weights,
            sample_size=args.yolo_sample,
            seed=args.seed,
        )
    else:
        print("[4/5] Skipping YOLOv8n exterior probe (--yolo-sample 0).")

    print("[5/5] Writing CSV manifests and audit report...")
    manifest_path = args.out_dir / "local_vmmrdb_manifest.csv"
    classes_path = args.out_dir / "local_vmmrdb_classes.csv"
    class_rows = build_class_summary(records)

    write_csv(manifest_path, MANIFEST_FIELDS, records)
    write_csv(classes_path, CLASS_FIELDS, class_rows)

    elapsed = time.perf_counter() - t0
    write_audit_report(
        report_path=args.report_path,
        data_dir=args.data_dir,
        folder_count=folder_count,
        non_image_files=non_image_files,
        records=records,
        class_rows=class_rows,
        min_class_images=args.min_class_images,
        yolo_stats=yolo_stats,
        elapsed_sec=elapsed,
    )
    print(f"[DONE] Finished in {elapsed:.1f}s -> {manifest_path}, {classes_path}, {args.report_path}")


if __name__ == "__main__":
    main()
