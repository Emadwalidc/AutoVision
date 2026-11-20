"""AutoVision - Full-Scale Feature Extraction on Kaggle GPU (89,535 Clean Images).

Run in Kaggle Notebook (Accelerator: GPU T4 x2 / P100, Internet: ON).
First install dependencies in a separate cell if missing:
# !pip install -q ultralytics open_clip_torch

Inputs (attached under /kaggle/input/**):
  1. local_vmmrdb_manifest.csv
  2. Clean image folders (<make>_<model>_<year>/*.jpg) matching rel_path

Outputs (/kaggle/working/autovision_features/):
  - bboxes_yolo.npz                   (YOLOv8n crops computed once for all 89,535 images)
  - features_openclip_vitb16.npz      (89,535 x 512 float16, Flip TTA + L2)
  - features_dinov2_vitb14.npz        (89,535 x 768 float16, Flip TTA + L2)
  - ablation_raw_test_<backbone>.npz  (2,000 test images WITHOUT YOLO crop for thesis ablation)
  - extraction_summary.json           (timings, throughput, YOLO rate, quick val accuracy)
"""

from __future__ import annotations

import csv
import glob
import json
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics_cfg")

import numpy as np
import open_clip
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from ultralytics import YOLO

# --- Configuration ---
BACKBONES = ["openclip_vitb16", "dinov2_vitb14"]
BATCH_YOLO = 64
BATCH_EMBED = 64
NUM_WORKERS = 2
ABLATION_RAW_TEST_N = 2000
CHECKPOINT_EVERY = 10000
SEED = 42
OUT_DIR = Path("/kaggle/working/autovision_features")

BACKBONE_DIMS = {
    "openclip_vitb16": 512,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
}


def find_manifest_and_image_root() -> tuple[Path, Path, list[dict]]:
    """Auto-discovers local_vmmrdb_manifest.csv and image root directory under /kaggle/input."""
    csv_candidates = sorted(glob.glob("/kaggle/input/**/local_vmmrdb_manifest.csv", recursive=True))
    if not csv_candidates:
        raise FileNotFoundError(
            "local_vmmrdb_manifest.csv not found under /kaggle/input/**. "
            "Upload it or the packed Kaggle zip first."
        )
    manifest_path = Path(csv_candidates[0])

    rows = []
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("is_clean", "")).strip().lower() == "true":
                orig_rel = row["rel_path"].replace("\\", "/")
                img_rel = (row.get("kaggle_rel_path") or orig_rel).replace("\\", "/")
                rows.append(
                    {
                        "rel_path": orig_rel,
                        "img_rel_path": img_rel,
                        "make": row["make"],
                        "canonical_class": row["canonical_class"],
                        "split": row["split"],
                    }
                )
    if not rows:
        raise ValueError(f"No clean rows found in {manifest_path}")

    sample_rels = [
        rows[0]["img_rel_path"],
        rows[len(rows) // 2]["img_rel_path"],
        rows[-1]["img_rel_path"],
    ]
    input_root = Path("/kaggle/input")
    candidates = [manifest_path.parent, manifest_path.parent / "images"]
    for root, dirs, _ in os.walk(input_root):
        rel_depth = Path(root).relative_to(input_root).parts
        if len(rel_depth) > 4:
            dirs[:] = []
            continue
        candidates.append(Path(root))

    seen = set()
    for cand in candidates:
        if cand in seen or not cand.is_dir():
            continue
        seen.add(cand)
        if all((cand / rp).is_file() for rp in sample_rels):
            return manifest_path, cand, rows

    raise FileNotFoundError(
        f"Could not locate image root under /kaggle/input matching sample paths: {sample_rels}"
    )


def compute_vehicle_crop_box(
    img_w: int,
    img_h: int,
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    min_area_ratio: float = 0.08,
    pad: float = 0.05,
) -> tuple[tuple[int, int, int, int], list[float], bool]:
    """Selects primary vehicle bounding box with padding, or 85% center fallback."""
    img_area = float(max(img_w * img_h, 1))
    best_conf = 0.0
    best_ratio = 0.0

    if boxes_xyxy.ndim == 2 and len(boxes_xyxy) > 0:
        areas = np.maximum(boxes_xyxy[:, 2] - boxes_xyxy[:, 0], 0.0) * np.maximum(
            boxes_xyxy[:, 3] - boxes_xyxy[:, 1], 0.0
        )
        idx = int(np.argmax(areas))
        bx1, by1, bx2, by2 = [float(v) for v in boxes_xyxy[idx]]
        best_conf = float(confs[idx]) if len(confs) > idx else 0.0
        bw, bh = max(bx2 - bx1, 0.0), max(by2 - by1, 0.0)
        best_ratio = (bw * bh) / img_area

        if best_ratio > min_area_ratio:
            x1 = max(0, int(bx1 - pad * bw))
            y1 = max(0, int(by1 - pad * bh))
            x2 = min(img_w, int(bx2 + pad * bw))
            y2 = min(img_h, int(by2 + pad * bh))
            x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
            return (x1, y1, x2, y2), [float(x1), float(y1), float(x2), float(y2), best_conf, best_ratio], True

    mw, mh = int(img_w * 0.075), int(img_h * 0.075)
    x1, y1 = min(mw, max(img_w - 1, 0)), min(mh, max(img_h - 1, 0))
    x2, y2 = max(x1 + 1, img_w - mw), max(y1 + 1, img_h - mh)
    return (x1, y1, x2, y2), [float(x1), float(y1), float(x2), float(y2), best_conf, best_ratio], False


class RawImageDataset(Dataset):
    def __init__(self, img_root: Path, rel_paths: list[str]) -> None:
        self.img_root = img_root
        self.rel_paths = rel_paths

    def __len__(self) -> int:
        return len(self.rel_paths)

    def __getitem__(self, idx: int):
        with Image.open(self.img_root / self.rel_paths[idx]) as im:
            rgb = im.convert("RGB")
        return idx, rgb


def collate_raw(batch):
    indices, imgs = zip(*batch)
    return list(indices), list(imgs)


class CroppedTensorDataset(Dataset):
    def __init__(
        self,
        img_root: Path,
        rel_paths: list[str],
        bboxes: np.ndarray | None,
        preprocess: T.Compose,
    ) -> None:
        self.img_root = img_root
        self.rel_paths = rel_paths
        self.bboxes = bboxes
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.rel_paths)

    def __getitem__(self, idx: int):
        with Image.open(self.img_root / self.rel_paths[idx]) as im:
            rgb = im.convert("RGB")
        if self.bboxes is not None:
            x1, y1, x2, y2 = [int(v) for v in self.bboxes[idx, :4]]
            w, h = rgb.size
            x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
            x2, y2 = max(x1 + 1, min(x2, w)), max(y1 + 1, min(y2, h))
            rgb = rgb.crop((x1, y1, x2, y2))
        return idx, self.preprocess(rgb)


def run_yolo_stage(
    img_root: Path,
    img_rel_paths: list[str],
    rel_paths: list[str],
    out_dir: Path,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Stage 1: Single YOLOv8n pass over all images (reuses cached bboxes_yolo.npz if matching)."""
    bbox_path = out_dir / "bboxes_yolo.npz"
    if bbox_path.exists():
        with np.load(bbox_path, allow_pickle=False) as d:
            if d["rel_paths"].tolist() == rel_paths:
                print(f"[Stage 1] Reusing cached {bbox_path} (yolo_hit={d['used_yolo'].mean()*100:.2f}%)")
                return d["bboxes"], d["used_yolo"], 0.0

    t0 = time.perf_counter()
    yolo = YOLO("yolov8n.pt")
    n = len(rel_paths)
    bboxes = np.zeros((n, 6), dtype=np.float32)
    used_yolo = np.zeros(n, dtype=bool)

    loader = DataLoader(
        RawImageDataset(img_root, img_rel_paths),
        batch_size=BATCH_YOLO,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_raw,
    )

    for indices, imgs in tqdm(loader, desc="Stage 1: YOLOv8n crops"):
        results = yolo.predict(imgs, classes=[2, 5, 7], conf=0.25, device=device, verbose=False)
        for idx, im, res in zip(indices, imgs, results):
            w, h = im.size
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
            else:
                xyxy = np.empty((0, 4), dtype=np.float32)
                confs = np.empty((0,), dtype=np.float32)
            _, row6, y_flag = compute_vehicle_crop_box(w, h, xyxy, confs)
            bboxes[idx] = row6
            used_yolo[idx] = y_flag
            im.close()

    elapsed = time.perf_counter() - t0
    np.savez(
        bbox_path,
        bboxes=bboxes,
        used_yolo=used_yolo,
        rel_paths=np.asarray(rel_paths, dtype=str),
    )
    print(f"[Stage 1] Done in {elapsed:.1f}s | YOLO detection hit rate: {used_yolo.mean()*100:.2f}%")
    return bboxes, used_yolo, elapsed


def build_backbone(name: str, device: str):
    """Loads OpenCLIP or DINOv2 encoder with direct 224x224 BICUBIC resize."""
    if name == "openclip_vitb16":
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="laion2b_s34b_b88k", device=device
        )
        model.eval()
        prep = T.Compose(
            [
                T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        return model.encode_image, prep, 512, "laion2b_s34b_b88k"

    if name in ("dinov2_vitb14", "dinov2_vitl14"):
        model = torch.hub.load("facebookresearch/dinov2", name).to(device).eval()
        prep = T.Compose(
            [
                T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        return model, prep, BACKBONE_DIMS[name], "facebookresearch/dinov2"

    raise ValueError(f"Unknown backbone: {name}")


@torch.no_grad()
def extract_backbone_features(
    backbone_name: str,
    img_root: Path,
    img_rel_paths: list[str],
    rel_paths: list[str],
    bboxes: np.ndarray | None,
    used_yolo: np.ndarray,
    out_path: Path,
    device: str,
    no_yolo: bool = False,
) -> tuple[np.ndarray, float]:
    """Extracts L2-normalized Flip-TTA float16 embeddings and verifies integrity."""
    forward_fn, prep, dim, pretrained_tag = build_backbone(backbone_name, device)
    n = len(rel_paths)
    partial_path = out_path.with_suffix(".partial.npz")

    embeddings = np.zeros((n, dim), dtype=np.float16)
    start_idx = 0
    if partial_path.exists():
        with np.load(partial_path, allow_pickle=False) as ckpt:
            done = int(ckpt["done_count"].item())
            if ckpt["rel_paths"].tolist() == rel_paths and 0 < done <= n:
                embeddings[:done] = ckpt["embeddings"][:done]
                start_idx = done
                print(f"  [Resume] {backbone_name}: {done:,}/{n:,}")

    t0 = time.perf_counter()
    if start_idx < n:
        sub_bboxes = None if bboxes is None else bboxes[start_idx:]
        loader = DataLoader(
            CroppedTensorDataset(img_root, img_rel_paths[start_idx:], sub_bboxes, prep),
            batch_size=BATCH_EMBED,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=device.startswith("cuda"),
        )
        last_ckpt = start_idx
        processed = start_idx
        for local_idx, x in tqdm(loader, desc=f"Extract {backbone_name}"):
            x = x.to(device, non_blocking=True)
            x_tta = torch.cat([x, torch.flip(x, dims=[-1])], dim=0)
            with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
                feats = forward_fn(x_tta)
            feats = torch.nn.functional.normalize(feats.float(), dim=-1)
            bs = x.shape[0]
            comb = torch.nn.functional.normalize(feats[:bs] + feats[bs:], dim=-1)

            g_idx = start_idx + local_idx.numpy()
            embeddings[g_idx] = comb.cpu().numpy().astype(np.float16)
            processed += bs

            if CHECKPOINT_EVERY > 0 and (processed - last_ckpt) >= CHECKPOINT_EVERY and processed < n:
                np.savez(
                    partial_path,
                    embeddings=embeddings,
                    rel_paths=np.asarray(rel_paths, dtype=str),
                    done_count=np.asarray(processed, dtype=np.int64),
                )
                last_ckpt = processed

    elapsed = time.perf_counter() - t0
    assert embeddings.shape == (n, dim) and np.isfinite(embeddings).all(), "Invalid embeddings!"
    norms = np.linalg.norm(embeddings.astype(np.float32), axis=1)
    assert (norms > 0.9).all(), f"Non-unit norms detected: min={norms.min():.4f}"

    if bboxes is None:
        raw_boxes = np.zeros((n, 6), dtype=np.float32)
        for i, img_rp in enumerate(img_rel_paths):
            with Image.open(img_root / img_rp) as im:
                w, h = im.size
            raw_boxes[i] = [0.0, 0.0, float(w), float(h), 0.0, 1.0]
        save_boxes = raw_boxes
    else:
        save_boxes = bboxes.astype(np.float32)

    meta = {
        "backbone": backbone_name,
        "pretrained": pretrained_tag,
        "embed_dim": dim,
        "img_size": 224,
        "tta": "horizontal_flip",
        "dtype": "float16",
        "no_yolo": bool(no_yolo),
        "num_samples": n,
        "yolo_hit_rate": round(float(used_yolo.mean()), 4) if not no_yolo else 0.0,
        "elapsed_sec": round(elapsed, 2),
    }
    np.savez(
        out_path,
        embeddings=embeddings,
        bboxes=save_boxes,
        used_yolo=np.asarray(used_yolo, dtype=bool),
        rel_paths=np.asarray(rel_paths, dtype=str),
        metadata_json=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )
    if partial_path.exists():
        partial_path.unlink()

    del forward_fn
    torch.cuda.empty_cache()
    print(f"  Saved {out_path.name} ({embeddings.shape}) in {elapsed:.1f}s ({n/max(elapsed, 1e-6):.1f} img/s)")
    return embeddings, elapsed


def quick_val_prototype_score(embeddings: np.ndarray, rows: list[dict], device: str) -> dict:
    """Computes quick GPU Nearest-Prototype Top-1/Top-5 accuracy on val split."""
    train_idx = [i for i, r in enumerate(rows) if r["split"] == "train"]
    val_idx = [i for i, r in enumerate(rows) if r["split"] == "val"]
    by_cls = defaultdict(list)
    for i in train_idx:
        by_cls[rows[i]["canonical_class"]].append(i)

    classes = sorted(by_cls)
    cls_to_id = {c: idx for idx, c in enumerate(classes)}
    emb_f = embeddings.astype(np.float32)

    protos = np.zeros((len(classes), emb_f.shape[1]), dtype=np.float32)
    for idx, c in enumerate(classes):
        m = emb_f[by_cls[c]].mean(axis=0)
        protos[idx] = m / (np.linalg.norm(m) + 1e-12)

    val_embs = torch.from_numpy(emb_f[val_idx]).to(device)
    proto_t = torch.from_numpy(protos).to(device)
    sims = val_embs @ proto_t.T
    top5 = torch.topk(sims, k=min(5, len(classes)), dim=1).indices.cpu().numpy()

    y_val = np.array([cls_to_id.get(rows[i]["canonical_class"], -1) for i in val_idx])
    top1_acc = float((top5[:, 0] == y_val).mean())
    top5_acc = float(np.mean([y in row for y, row in zip(y_val, top5)]))
    return {"val_proto_top1": round(top1_acc, 4), "val_proto_top5": round(top5_acc, 4)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== AutoVision Kaggle Full Extraction (device={device.upper()}) ===")

    manifest_path, img_root, rows = find_manifest_and_image_root()
    rel_paths = [r["rel_path"] for r in rows]
    img_rel_paths = [r["img_rel_path"] for r in rows]
    print(f"Manifest: {manifest_path} | Image root: {img_root} | Clean images: {len(rows):,}")

    # Stage 1: Single YOLO pass
    bboxes, used_yolo, yolo_sec = run_yolo_stage(img_root, img_rel_paths, rel_paths, OUT_DIR, device)

    # Deterministic 2,000 test subset for no-YOLO ablation
    test_indices = [i for i, r in enumerate(rows) if r["split"] == "test"]
    rng = np.random.default_rng(SEED)
    n_abl = min(ABLATION_RAW_TEST_N, len(test_indices))
    abl_idx = np.sort(rng.choice(test_indices, size=n_abl, replace=False))
    abl_rel_paths = [rel_paths[int(i)] for i in abl_idx]
    abl_img_rel_paths = [img_rel_paths[int(i)] for i in abl_idx]
    abl_used_yolo = np.zeros(n_abl, dtype=bool)

    summary = {
        "num_clean_images": len(rows),
        "num_classes": len({r["canonical_class"] for r in rows}),
        "yolo_elapsed_sec": round(yolo_sec, 2),
        "yolo_hit_rate": round(float(used_yolo.mean()), 4),
        "ablation_test_n": n_abl,
        "backbones": {},
    }

    # Stage 2 & 3: Backbones + Raw Ablation
    for bb in BACKBONES:
        print(f"\n[Stage 2] Extracting full {bb} features...")
        feat_path = OUT_DIR / f"features_{bb}.npz"
        embs, bb_sec = extract_backbone_features(
            bb, img_root, img_rel_paths, rel_paths, bboxes, used_yolo, feat_path, device, no_yolo=False
        )

        print(f"[Stage 3] Extracting raw-frame ablation ({n_abl} test images) for {bb}...")
        abl_path = OUT_DIR / f"ablation_raw_test_{bb}.npz"
        _, abl_sec = extract_backbone_features(
            bb, img_root, abl_img_rel_paths, abl_rel_paths, None, abl_used_yolo, abl_path, device, no_yolo=True
        )

        scores = quick_val_prototype_score(embs, rows, device)
        summary["backbones"][bb] = {
            "embed_dim": BACKBONE_DIMS[bb],
            "extract_sec": round(bb_sec, 2),
            "img_per_sec": round(len(rows) / max(bb_sec, 1e-6), 1),
            "ablation_sec": round(abl_sec, 2),
            **scores,
        }
        print(f"  Quick Val Prototype ({bb}): Top-1={scores['val_proto_top1']*100:.2f}%, Top-5={scores['val_proto_top5']*100:.2f}%")

    summary_path = OUT_DIR / "extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== ALL DONE -> {OUT_DIR} ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
