"""Build zero-retraining European & modern vehicle prototypes from DVM-CAR Confirmed_fronts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import shutil
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from multitask_model import CLIP_MEAN, CLIP_STD, AutoVisionMultiTaskNet  # noqa: E402
from v2_taxonomy import canonicalize_make_model  # noqa: E402

DVM_FIGSHARE_URL = "https://ndownloader.figshare.com/files/34792480"
DVM_DROPBOX_URL = "https://www.dropbox.com/s/xmyz2na20f8lth8/Confirmed_fronts.zip?dl=1"

EU_MAKE_DISPLAY: dict[str, tuple[str, str]] = {
    "vauxhall": ("opel", "Opel"),
    "opel": ("opel", "Opel"),
    "citroen": ("citroen", "Citroen"),
    "skoda": ("skoda", "Skoda"),
    "renault": ("renault", "Renault"),
    "peugeot": ("peugeot", "Peugeot"),
    "dacia": ("dacia", "Dacia"),
    "seat": ("seat", "SEAT"),
    "ds": ("ds", "DS Automobiles"),
}

PRIORITY_EU_MAKES = frozenset({"renault", "skoda", "peugeot", "opel", "citroen", "dacia", "seat", "ds"})

# Target EU/modern models missing or outdated in US Phase 1 -> optional min year
PRIORITY_MODERN_EU_SPECS: dict[str, int] = {
    "mazda__2": 2015,
    "ford__fiesta": 2017,
    "ford__focus": 2018,
    "toyota__yaris": 2015,
    "volkswagen__polo": 2012,
    "volkswagen__up": 2012,
    "volkswagen__t-roc": 2017,
    "volkswagen__tiguan": 2017,
    "ford__kuga": 2013,
    "ford__puma": 2019,
    "ford__mondeo": 2014,
    "toyota__auris": 2012,
    "toyota__aygo": 2014,
    "toyota__c-hr": 2016,
    "hyundai__i10": 2013,
    "hyundai__i20": 2014,
    "hyundai__i30": 2014,
    "kia__picanto": 2014,
    "kia__ceed": 2014,
    "kia__sportage": 2016,
    "nissan__qashqai": 2014,
    "nissan__micra": 2014,
    "nissan__note": 2013,
    "honda__jazz": 2014,
    "audi__a1": 2012,
    "audi__q2": 2016,
    "mercedes-benz__b-class": 2012,
    "mercedes-benz__a-class": 2018,
    "volvo__v40": 2012,
    "fiat__500": 2015,
    "fiat__panda": 2012,
    "fiat__tipo": 2015,
    "fiat__punto": 2012,
    "suzuki__swift": 2014,
    "suzuki__vitara": 2015,
    "mg__zs": 2017,
    "mg__mg3": 2014,
    "alfa-romeo__giulietta": 2012,
    "alfa-romeo__mito": 2012,
}

MODEL_SLUG_OVERRIDES: dict[tuple[str, str], tuple[str, str]] = {
    ("mazda", "mazda2"): ("2", "2"),
    ("mazda", "mazda 2"): ("2", "2"),
    ("mazda", "mazda3"): ("3", "3"),
    ("mazda", "mazda 3"): ("3", "3"),
    ("mazda", "mazda6"): ("6", "6"),
    ("mazda", "mazda 6"): ("6", "6"),
    ("volkswagen", "up"): ("up", "up!"),
    ("volkswagen", "up!"): ("up", "up!"),
    ("kia", "cee'd"): ("ceed", "Ceed"),
    ("kia", "ceed"): ("ceed", "Ceed"),
    ("kia", "pro_cee'd"): ("proceed", "ProCeed"),
    ("toyota", "c-hr"): ("c-hr", "C-HR"),
    ("volkswagen", "t-roc"): ("t-roc", "T-Roc"),
    ("mercedes-benz", "b class"): ("b-class", "B-Class"),
    ("mercedes-benz", "b-class"): ("b-class", "B-Class"),
    ("mercedes-benz", "a class"): ("a-class", "A-Class"),
    ("mercedes-benz", "a-class"): ("a-class", "A-Class"),
}


def download_dvm_zip(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(dest):
        print(f"[DVM-CAR] Using existing archive: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    for url in (DVM_FIGSHARE_URL, DVM_DROPBOX_URL):
        try:
            print(f"[DVM-CAR] Downloading Confirmed_fronts.zip from {url}...")
            req = urllib.request.Request(url, headers={"User-Agent": "AutoVision-EU-Proto/1.0"})
            tmp = dest.with_suffix(".tmp")
            with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f, length=1024 * 1024)
            if zipfile.is_zipfile(tmp):
                tmp.replace(dest)
                print(f"[DVM-CAR] Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
                return dest
            tmp.unlink(missing_ok=True)
        except Exception as exc:
            print(f"[DVM-CAR] Download failed from {url}: {exc}")
    raise RuntimeError("Could not download DVM-CAR Confirmed_fronts.zip from any mirror.")


def _parse_dvm_member(member: str) -> dict[str, Any] | None:
    if not member.lower().endswith((".jpg", ".jpeg", ".png")):
        return None
    name = Path(member).name
    stem = Path(name).stem
    raw_make = ""
    raw_model = ""
    year_val = 0
    color_val = "Unknown"

    if "$$" in stem:
        toks = stem.split("$$")
        if len(toks) < 3:
            return None
        raw_make, raw_model = toks[0].strip(), toks[1].strip()
        if toks[2].strip().isdigit():
            year_val = int(toks[2].strip())
        if len(toks) >= 4:
            color_val = toks[3].strip() or "Unknown"
    else:
        parts = [p for p in member.replace("\\", "/").split("/") if p]
        if len(parts) >= 3:
            raw_make, raw_model = parts[0], parts[1]
            if parts[2].isdigit():
                year_val = int(parts[2])
        else:
            return None

    mk_low = raw_make.strip().lower()
    if mk_low in EU_MAKE_DISPLAY:
        mk_slug, mk_disp = EU_MAKE_DISPLAY[mk_low]
        md_low = re.sub(r"\s+", " ", raw_model.strip().lower())
        md_slug = re.sub(r"[^a-z0-9-]+", "_", md_low).strip("_")
        md_disp = raw_model.strip()
    else:
        mk_slug, _, mk_disp, md_disp = canonicalize_make_model(raw_make, raw_model)
        ovr = MODEL_SLUG_OVERRIDES.get((mk_slug, raw_model.strip().lower()))
        if ovr is not None:
            md_slug, md_disp = ovr
        else:
            md_low = re.sub(r"\s+", " ", raw_model.strip().lower())
            md_slug = canonicalize_make_model(raw_make, raw_model)[1]
            if not md_slug:
                md_slug = re.sub(r"[^a-z0-9-]+", "_", md_low).strip("_")

    if not mk_slug or not md_slug:
        return None
    return {
        "member": member,
        "make": mk_slug,
        "make_display": mk_disp,
        "canonical_class": f"{mk_slug}__{md_slug}",
        "model_display": md_disp,
        "year": year_val,
        "color": color_val,
    }


def _select_eu_targets(
    zip_path: Path,
    existing_closed_models: set[str],
    max_classes: int = 75,
    proto_per_class: int = 12,
    val_per_class: int = 3,
    seed: int = 42,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, str], dict[str, str]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    make_displays: dict[str, str] = {}
    model_displays: dict[str, str] = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            rec = _parse_dvm_member(member)
            if rec is None:
                continue
            cls_name = rec["canonical_class"]
            by_class[cls_name].append(rec)
            make_displays.setdefault(rec["make"], rec["make_display"])
            model_displays.setdefault(cls_name, rec["model_display"])

    min_needed = proto_per_class + val_per_class
    chosen_classes: list[str] = []

    # Priority A: top models from missing EU brands
    per_brand_limit = 7
    brand_counts: dict[str, int] = defaultdict(int)
    eu_candidates = [
        c for c, items in by_class.items()
        if c.split("__", 1)[0] in PRIORITY_EU_MAKES and len(items) >= min_needed
    ]
    eu_candidates.sort(key=lambda c: (-len(by_class[c]), c))
    for c in eu_candidates:
        mk = c.split("__", 1)[0]
        if brand_counts[mk] < per_brand_limit:
            chosen_classes.append(c)
            brand_counts[mk] += 1

    # Priority B: modern / non-US European models
    for c, min_yr in PRIORITY_MODERN_EU_SPECS.items():
        if c in chosen_classes or len(chosen_classes) >= max_classes:
            continue
        items = [r for r in by_class.get(c, []) if (r["year"] == 0 or r["year"] >= min_yr)]
        if len(items) < min_needed:
            items = by_class.get(c, [])
        # Skip models already heavily represented as modern closed classes unless explicitly upgraded
        if c in existing_closed_models and min_yr < 2015:
            continue
        if len(items) >= min_needed:
            by_class[c] = items
            chosen_classes.append(c)

    chosen_classes = chosen_classes[:max_classes]
    rng = random.Random(seed)
    proto_samples: dict[str, list[dict[str, Any]]] = {}
    val_samples: dict[str, list[dict[str, Any]]] = {}

    for c in chosen_classes:
        items = list(by_class[c])
        rng.shuffle(items)
        # Spread across colors & prefer newer years
        items.sort(key=lambda r: -(r["year"] if r["year"] >= 2000 else 0))
        by_color: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in items:
            by_color[r["color"]].append(r)

        diverse: list[dict[str, Any]] = []
        color_keys = sorted(by_color.keys())
        while len(diverse) < min_needed:
            added = False
            for ck in color_keys:
                if by_color[ck]:
                    diverse.append(by_color[ck].pop(0))
                    added = True
                    if len(diverse) >= min_needed:
                        break
            if not added:
                break

        proto_samples[c] = diverse[:proto_per_class]
        val_samples[c] = diverse[proto_per_class : proto_per_class + val_per_class]

    return proto_samples, val_samples, make_displays, model_displays


def _embed_and_save_thumbs(
    zip_path: Path,
    records: list[dict[str, Any]],
    net: AutoVisionMultiTaskNet,
    tf: T.Compose,
    device: str,
    thumbs_zf: zipfile.ZipFile | None,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    embs_all: list[np.ndarray] = []
    body_all: list[np.ndarray] = []
    kept_records: list[dict[str, Any]] = []

    with zipfile.ZipFile(zip_path, "r") as src_zf:
        batch_imgs: list[torch.Tensor] = []
        batch_recs: list[dict[str, Any]] = []

        def _flush() -> None:
            if not batch_imgs:
                return
            x = torch.stack(batch_imgs).to(device)
            with torch.inference_mode():
                o1 = net(x, model_targets=None)
                o2 = net(torch.flip(x, dims=[-1]), model_targets=None)
                emb = F.normalize(o1["embedding"].float() + o2["embedding"].float(), dim=-1)
                bd_p = F.softmax(0.5 * (o1["body_logits"].float() + o2["body_logits"].float()), dim=-1)
            embs_all.append(emb.cpu().numpy().astype(np.float32))
            body_all.append(bd_p.cpu().numpy().astype(np.float32))
            kept_records.extend(batch_recs)
            batch_imgs.clear()
            batch_recs.clear()

        for rec in records:
            try:
                raw = src_zf.read(rec["member"])
                with Image.open(io.BytesIO(raw)) as im:
                    rgb = im.convert("RGB")
                    batch_imgs.append(tf(rgb))

                    h_id = hashlib.sha1(rec["member"].encode("utf-8")).hexdigest()[:14]
                    rel_path = f"eu_dvm/{rec['canonical_class']}/{h_id}.jpg"
                    rec_copy = dict(rec)
                    rec_copy["rel_path"] = rel_path
                    batch_recs.append(rec_copy)

                    if thumbs_zf is not None:
                        w, h = rgb.size
                        if max(w, h) > 320:
                            sc = 320.0 / float(max(w, h))
                            rgb_s = rgb.resize((max(16, int(round(w * sc))), max(16, int(round(h * sc)))), Image.Resampling.BICUBIC)
                        else:
                            rgb_s = rgb
                        buf = io.BytesIO()
                        rgb_s.save(buf, format="JPEG", quality=88, optimize=True)
                        thumbs_zf.writestr(rel_path, buf.getvalue())
                        if rgb_s is not rgb:
                            rgb_s.close()
            except Exception:
                continue

            if len(batch_imgs) >= batch_size:
                _flush()
        _flush()

    if not embs_all:
        return np.zeros((0, 512), dtype=np.float32), np.zeros((0, 8), dtype=np.float32), []
    return np.concatenate(embs_all, axis=0), np.concatenate(body_all, axis=0), kept_records


def build_extension(project_root: Path, dvm_zip: Path, out_dir: Path) -> dict[str, Any]:
    p1_dir = project_root / "models" / "v2_phase1"
    lmap = json.loads((p1_dir / "label_maps.json").read_text(encoding="utf-8"))
    ckpt = torch.load(p1_dir / "autovision_multitask_best.pt", map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = AutoVisionMultiTaskNet(
        num_makes=len(lmap["make_classes"]),
        num_models=len(lmap["model_classes"]),
        num_bodies=len(lmap["body_classes"]),
        num_years=int(lmap["year_max"]) - int(lmap["year_min"]) + 1,
        num_colors=len(lmap["color_classes"]),
        pretrained=None,
    ).to(device)
    net.load_state_dict(state, strict=True)
    net.eval()

    tf = T.Compose(
        [
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )

    proto_map, val_map, mk_disp, md_disp = _select_eu_targets(
        dvm_zip, existing_closed_models=set(lmap["model_classes"]), max_classes=75
    )
    all_proto_recs = [r for c in sorted(proto_map) for r in proto_map[c]]
    all_val_recs = [r for c in sorted(val_map) for r in val_map[c]]
    print(f"[EU-Ext] Selected {len(proto_map)} EU/modern classes ({len(all_proto_recs)} gallery, {len(all_val_recs)} val) on {device}...")

    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs_zip_path = out_dir / "eu_gallery_thumbs.zip"
    with zipfile.ZipFile(thumbs_zip_path, "w", compression=zipfile.ZIP_STORED) as tzf:
        gal_embs, gal_bd, gal_recs = _embed_and_save_thumbs(dvm_zip, all_proto_recs, net, tf, device, thumbs_zf=tzf)

    val_embs, _, val_recs = _embed_and_save_thumbs(dvm_zip, all_val_recs, net, tf, device, thumbs_zf=None)

    body_classes = list(lmap["body_classes"])
    by_cls: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(gal_recs):
        by_cls[r["canonical_class"]].append(i)

    p_vecs, p_names, p_makes, p_bodies, p_splits = [], [], [], [], []
    for cls_name in sorted(by_cls):
        idxs = by_cls[cls_name]
        vec = gal_embs[idxs].mean(axis=0)
        vec = vec / max(float(np.linalg.norm(vec)), 1e-12)
        bd_idx = int(np.argmax(gal_bd[idxs].mean(axis=0)))
        rec0 = gal_recs[idxs[0]]
        rec0["inferred_body"] = body_classes[bd_idx]
        for idx in idxs:
            gal_recs[idx]["inferred_body"] = body_classes[bd_idx]
        p_vecs.append(vec.astype(np.float32))
        p_names.append(cls_name)
        p_makes.append(rec0["make"])
        p_bodies.append(body_classes[bd_idx])
        p_splits.append("eu_proto")

    eu_proto_mat = np.stack(p_vecs, axis=0)

    # Evaluate held-out EU shots against combined [base + EU] prototypes
    base_proto_npz = np.load(p1_dir / "prototypes_v2.npz", allow_pickle=False)
    comb_protos = np.vstack([base_proto_npz["prototypes"].astype(np.float32), eu_proto_mat])
    comb_protos /= np.maximum(np.linalg.norm(comb_protos, axis=1, keepdims=True), 1e-12)
    comb_names = [str(s) for s in base_proto_npz["class_names"]] + p_names
    base_proto_npz.close()

    sims = val_embs @ comb_protos.T
    top1_hits = 0
    top3_hits = 0
    for i, r in enumerate(val_recs):
        top3_i = np.argsort(-sims[i])[:3]
        top3_cls = [comb_names[int(k)] for k in top3_i]
        if top3_cls[0] == r["canonical_class"]:
            top1_hits += 1
        if r["canonical_class"] in top3_cls:
            top3_hits += 1

    np.savez_compressed(
        out_dir / "eu_prototypes.npz",
        prototypes=eu_proto_mat,
        class_names=np.array(p_names),
        makes=np.array(p_makes),
        body_types=np.array(p_bodies),
        split_types=np.array(p_splits),
        is_few_shot=np.ones(len(p_names), dtype=bool),
    )
    np.savez_compressed(
        out_dir / "eu_gallery_embeddings.npz",
        embeddings=gal_embs.astype(np.float32),
        rel_paths=np.array([r["rel_path"] for r in gal_recs]),
    )
    with open(out_dir / "eu_gallery_manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rel_path", "canonical_class", "make", "raw_year", "year_valid", "body_type", "split", "is_clean"])
        w.writeheader()
        for r in gal_recs:
            yr = r["year"] if r["year"] >= 1950 else ""
            w.writerow(
                {
                    "rel_path": r["rel_path"],
                    "canonical_class": r["canonical_class"],
                    "make": r["make"],
                    "raw_year": yr,
                    "year_valid": "true" if yr else "false",
                    "body_type": r.get("inferred_body", "Hatchback"),
                    "split": "few_shot",
                    "is_clean": "true",
                }
            )

    disp_payload = {
        "make_display_names": {mk: mk_disp[mk] for mk in p_makes if mk in mk_disp},
        "model_display_names": {c: md_disp[c] for c in p_names if c in md_disp},
    }
    (out_dir / "eu_display_names.json").write_text(json.dumps(disp_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "num_eu_classes": len(p_names),
        "num_eu_gallery_images": len(gal_recs),
        "num_eu_val_images": len(val_recs),
        "eu_heldout_combined_proto_top1": round(top1_hits / max(len(val_recs), 1), 4),
        "eu_heldout_combined_proto_top3": round(top3_hits / max(len(val_recs), 1), 4),
        "eu_classes": p_names,
    }
    (out_dir / "eu_extension_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[EU-Ext] Built {len(p_names)} EU/modern prototypes | Held-out Top-1={summary['eu_heldout_combined_proto_top1']*100:.2f}% | Top-3={summary['eu_heldout_combined_proto_top3']*100:.2f}%")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build zero-retraining EU & modern DVM-CAR prototypes for AutoVision v2.")
    parser.add_argument("--project-root", type=Path, default=BASE_DIR)
    parser.add_argument("--cache-zip", type=Path, default=BASE_DIR / ".cache" / "dvm_confirmed_fronts.zip")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "models" / "v2_eu_extension")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--cleanup-zip", action="store_true")
    args = parser.parse_args()

    if args.download_only:
        download_dvm_zip(args.cache_zip)
        return

    if not args.build_only and not zipfile.is_zipfile(args.cache_zip):
        download_dvm_zip(args.cache_zip)
    if not zipfile.is_zipfile(args.cache_zip):
        raise FileNotFoundError(f"DVM-CAR zip not found: {args.cache_zip}. Run with --download-only first.")

    build_extension(args.project_root, args.cache_zip, args.out_dir)
    if args.cleanup_zip and args.cache_zip.is_file():
        args.cache_zip.unlink()
        print(f"[EU-Ext] Removed temporary archive {args.cache_zip}")


if __name__ == "__main__":
    main()
