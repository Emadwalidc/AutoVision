"""Full-scale vehicle feature extraction (YOLOv8n crop + OpenCLIP / DINOv2 + Flip TTA).

Supports both local CPU/GPU and Kaggle GPU environments, checkpointing with
--resume, optional --no-yolo ablation, and compact float16 .npz bundle export.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_ULTRALYTICS = BASE_DIR / ".cache" / "ultralytics"
CACHE_ULTRALYTICS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_ULTRALYTICS))

BACKBONE_DIMS: dict[str, int] = {
    "openclip_vitb16": 512,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
}


def load_clean_manifest(
    manifest_path: Path | str,
    splits: str | list[str] | set[str] | None = None,
    max_samples: int = 0,
) -> list[dict[str, Any]]:
    """Loads clean records from local_vmmrdb_manifest.csv with optional split filtering."""
    allowed_splits: set[str] | None = None
    if splits is not None:
        if isinstance(splits, str):
            parsed = {s.strip() for s in splits.split(",") if s.strip()}
            allowed_splits = parsed if parsed else None
        else:
            parsed = {str(s).strip() for s in splits if str(s).strip()}
            allowed_splits = parsed if parsed else None

    rows: list[dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("is_clean", "")).strip().lower() != "true":
                continue
            split = str(row.get("split", "")).strip()
            if allowed_splits is not None and split not in allowed_splits:
                continue
            rows.append(
                {
                    "rel_path": row["rel_path"],
                    "make": row["make"],
                    "model": row["model"],
                    "canonical_class": row["canonical_class"],
                    "raw_year": int(row["raw_year"]),
                    "year_valid": str(row.get("year_valid", "")).strip().lower() == "true",
                    "split": split,
                }
            )
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def compute_vehicle_crop_box(
    img_w: int,
    img_h: int,
    boxes_xyxy: np.ndarray | list[list[float]],
    confs: np.ndarray | list[float],
    min_area_ratio: float = 0.08,
    pad: float = 0.05,
) -> tuple[tuple[int, int, int, int], list[float], bool]:
    """Selects primary vehicle bounding box with padding, or 85% center fallback."""
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    conf_arr = np.asarray(confs, dtype=np.float32)
    img_area = float(max(img_w * img_h, 1))

    best_conf = 0.0
    best_ratio = 0.0
    if boxes.ndim == 2 and len(boxes) > 0:
        areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(
            boxes[:, 3] - boxes[:, 1], 0.0
        )
        best_idx = int(np.argmax(areas))
        bx1, by1, bx2, by2 = [float(v) for v in boxes[best_idx]]
        best_conf = float(conf_arr[best_idx]) if len(conf_arr) > best_idx else 0.0
        bw = max(bx2 - bx1, 0.0)
        bh = max(by2 - by1, 0.0)
        best_ratio = (bw * bh) / img_area

        if best_ratio > min_area_ratio:
            x1 = max(0, int(bx1 - pad * bw))
            y1 = max(0, int(by1 - pad * bh))
            x2 = min(img_w, int(bx2 + pad * bw))
            y2 = min(img_h, int(by2 + pad * bh))
            x2 = max(x1 + 1, x2)
            y2 = max(y1 + 1, y2)
            return (
                (x1, y1, x2, y2),
                [float(x1), float(y1), float(x2), float(y2), best_conf, float(best_ratio)],
                True,
            )

    mw = int(img_w * 0.075)
    mh = int(img_h * 0.075)
    x1 = min(mw, max(img_w - 1, 0))
    y1 = min(mh, max(img_h - 1, 0))
    x2 = max(x1 + 1, img_w - mw)
    y2 = max(y1 + 1, img_h - mh)
    return (
        (x1, y1, x2, y2),
        [float(x1), float(y1), float(x2), float(y2), best_conf, float(best_ratio)],
        False,
    )


def build_preprocess(backbone_name: str, img_size: int = 224) -> T.Compose:
    """Builds direct BICUBIC resize transform (no center crop) for vehicle crops."""
    if backbone_name == "openclip_vitb16":
        mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    elif backbone_name in ("dinov2_vitb14", "dinov2_vitl14"):
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    return T.Compose(
        [
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def load_backbone(
    backbone_name: str,
    device: str | torch.device,
) -> tuple[Callable[[torch.Tensor], np.ndarray], T.Compose, int, dict[str, Any]]:
    """Loads pretrained vision backbone and returns (encode_batch_fn, preprocess, dim, meta)."""
    if backbone_name not in BACKBONE_DIMS:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    dev_str = str(device)
    use_cuda = dev_str.startswith("cuda")
    embed_dim = BACKBONE_DIMS[backbone_name]
    preprocess = build_preprocess(backbone_name, img_size=224)

    if backbone_name == "openclip_vitb16":
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="laion2b_s34b_b88k", device=device
        )
        model.eval()
        forward_fn = model.encode_image
        pretrained_tag = "laion2b_s34b_b88k"
    else:
        model = torch.hub.load("facebookresearch/dinov2", backbone_name)
        model = model.to(device).eval()
        forward_fn = model
        pretrained_tag = "facebookresearch/dinov2"

    @torch.no_grad()
    def encode_batch_fn(tensor_batch: torch.Tensor) -> np.ndarray:
        x = tensor_batch.to(device, non_blocking=True)
        x_tta = torch.cat([x, torch.flip(x, dims=[-1])], dim=0)
        with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
            feats = forward_fn(x_tta)
        feats = feats.float()
        feats = torch.nn.functional.normalize(feats, dim=-1)
        n = x.shape[0]
        combined = feats[:n] + feats[n:]
        combined = torch.nn.functional.normalize(combined, dim=-1)
        return combined.cpu().numpy().astype(np.float16)

    meta = {
        "backbone": backbone_name,
        "pretrained": pretrained_tag,
        "embed_dim": embed_dim,
        "img_size": 224,
        "tta": "horizontal_flip",
        "dtype": "float16",
    }
    return encode_batch_fn, preprocess, embed_dim, meta


def save_feature_bundle(
    out_path: Path | str,
    embeddings: np.ndarray,
    bboxes: np.ndarray,
    used_yolo: np.ndarray,
    rel_paths: list[str] | np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Validates feature arrays (shape, finiteness, unit L2 norms) and saves .npz bundle."""
    emb = np.asarray(embeddings, dtype=np.float16)
    boxes = np.asarray(bboxes, dtype=np.float32)
    yolo_flags = np.asarray(used_yolo, dtype=bool)
    paths_arr = np.asarray(rel_paths, dtype=str)

    n = len(paths_arr)
    if emb.ndim != 2 or emb.shape[0] != n or boxes.shape != (n, 6) or yolo_flags.shape != (n,):
        raise ValueError(
            f"Shape mismatch: emb={emb.shape}, boxes={boxes.shape}, "
            f"used_yolo={yolo_flags.shape}, rel_paths={n}"
        )
    if not np.isfinite(emb).all():
        raise ValueError("Non-finite values detected in embeddings.")
    if n > 0:
        norms = np.linalg.norm(emb.astype(np.float32), axis=1)
        if not bool((norms > 0.9).all()):
            raise ValueError(f"Non-unit L2 norms detected: min={norms.min():.4f}")

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_file,
        embeddings=emb,
        bboxes=boxes,
        used_yolo=yolo_flags,
        rel_paths=paths_arr,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )


def load_feature_bundle(path: Path | str) -> dict[str, Any]:
    """Loads and validates a saved .npz feature bundle."""
    with np.load(path, allow_pickle=False) as data:
        embeddings = data["embeddings"]
        bboxes = data["bboxes"]
        used_yolo = data["used_yolo"]
        rel_paths = data["rel_paths"]
        meta_raw = data["metadata_json"].item() if "metadata_json" in data else "{}"

    metadata = json.loads(meta_raw) if isinstance(meta_raw, str) else {}
    n = len(rel_paths)
    if embeddings.shape[0] != n or bboxes.shape != (n, 6) or len(used_yolo) != n:
        raise ValueError(f"Corrupt bundle shapes in {path}")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"Non-finite values in {path}")
    if n > 0:
        norms = np.linalg.norm(embeddings.astype(np.float32), axis=1)
        if not bool((norms > 0.9).all()):
            raise ValueError(f"Non-unit norms in {path}: min={norms.min():.4f}")

    return {
        "embeddings": embeddings,
        "bboxes": bboxes,
        "used_yolo": used_yolo,
        "rel_paths": rel_paths,
        "metadata": metadata,
    }


class _RawImageDataset(Dataset):
    """Loads RGB images from disk in parallel DataLoader workers."""

    def __init__(self, data_dir: Path, rel_paths: list[str]) -> None:
        self.data_dir = data_dir
        self.rel_paths = rel_paths

    def __len__(self) -> int:
        return len(self.rel_paths)

    def __getitem__(self, idx: int) -> tuple[int, Image.Image]:
        img_path = self.data_dir / self.rel_paths[idx]
        with Image.open(img_path) as im:
            rgb = im.convert("RGB")
        return idx, rgb


def _collate_pil(batch: list[tuple[int, Image.Image]]) -> tuple[list[int], list[Image.Image]]:
    indices, images = zip(*batch)
    return list(indices), list(images)


def extract_features(
    manifest_path: Path,
    data_dir: Path,
    out_dir: Path,
    backbone_name: str,
    yolo_weights: Path,
    no_yolo: bool = False,
    batch_size: int = 64,
    num_workers: int | None = None,
    checkpoint_every: int = 5000,
    resume: bool = False,
    max_samples: int = 0,
    splits: str | None = None,
) -> Path:
    """Runs end-to-end feature extraction and saves the final .npz bundle."""
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 2

    records = load_clean_manifest(manifest_path, splits=splits, max_samples=max_samples)
    if not records:
        raise ValueError(f"No clean records found in {manifest_path} (splits={splits})")

    rel_paths = [r["rel_path"] for r in records]
    n_total = len(rel_paths)
    embed_dim = BACKBONE_DIMS[backbone_name]

    suffix = "_noyolo" if no_yolo else ""
    if splits:
        split_tag = "_".join(sorted(s.strip() for s in splits.split(",") if s.strip()))
        suffix += f"_{split_tag}"
    out_path = out_dir / f"features_{backbone_name}{suffix}.npz"
    partial_path = out_dir / f"features_{backbone_name}{suffix}.partial.npz"

    embeddings = np.zeros((n_total, embed_dim), dtype=np.float16)
    bboxes = np.zeros((n_total, 6), dtype=np.float32)
    used_yolo = np.zeros(n_total, dtype=bool)
    start_idx = 0

    if resume and partial_path.exists():
        with np.load(partial_path, allow_pickle=False) as ckpt:
            ckpt_paths = ckpt["rel_paths"].tolist()
            done = int(ckpt["done_count"].item())
            if ckpt_paths == rel_paths and 0 < done <= n_total:
                embeddings[:done] = ckpt["embeddings"][:done]
                bboxes[:done] = ckpt["bboxes"][:done]
                used_yolo[:done] = ckpt["used_yolo"][:done]
                start_idx = done
                print(f"[RESUME] Loaded {done:,}/{n_total:,} items from {partial_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device={device.upper()}, backbone={backbone_name}, no_yolo={no_yolo}, items={n_total - start_idx:,}")

    yolo_model = None
    if not no_yolo:
        from ultralytics import YOLO

        if not yolo_weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")
        yolo_model = YOLO(str(yolo_weights))

    encode_batch_fn, preprocess, _, meta = load_backbone(backbone_name, device)
    meta["no_yolo"] = bool(no_yolo)
    meta["splits"] = splits or "all_clean"
    meta["num_samples"] = n_total

    t0 = time.perf_counter()
    if start_idx < n_total:
        dataset = _RawImageDataset(data_dir, rel_paths[start_idx:])
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate_pil,
        )

        last_ckpt = start_idx
        processed = start_idx
        pbar = tqdm(loader, desc=f"Extract ({backbone_name})")
        for local_indices, imgs in pbar:
            if yolo_model is not None:
                yolo_results = yolo_model.predict(
                    imgs, classes=[2, 5, 7], conf=0.25, verbose=False
                )
            else:
                yolo_results = [None] * len(imgs)

            tensors = []
            for offset, im, res in zip(local_indices, imgs, yolo_results):
                global_idx = start_idx + offset
                w, h = im.size
                if yolo_model is None:
                    crop_box = (0, 0, w, h)
                    bboxes[global_idx] = [0.0, 0.0, float(w), float(h), 0.0, 1.0]
                    used_yolo[global_idx] = False
                else:
                    if res is not None and res.boxes is not None and len(res.boxes) > 0:
                        xyxy = res.boxes.xyxy.cpu().numpy()
                        confs = res.boxes.conf.cpu().numpy()
                    else:
                        xyxy = np.empty((0, 4), dtype=np.float32)
                        confs = np.empty((0,), dtype=np.float32)
                    crop_box, row6, y_flag = compute_vehicle_crop_box(w, h, xyxy, confs)
                    bboxes[global_idx] = row6
                    used_yolo[global_idx] = y_flag

                cropped = im.crop(crop_box)
                tensors.append(preprocess(cropped))
                im.close()

            batch_tensor = torch.stack(tensors, dim=0)
            batch_emb = encode_batch_fn(batch_tensor)
            g_indices = [start_idx + o for o in local_indices]
            embeddings[g_indices] = batch_emb
            processed += len(local_indices)

            if checkpoint_every > 0 and (processed - last_ckpt) >= checkpoint_every and processed < n_total:
                out_dir.mkdir(parents=True, exist_ok=True)
                np.savez(
                    partial_path,
                    embeddings=embeddings,
                    bboxes=bboxes,
                    used_yolo=used_yolo,
                    rel_paths=np.asarray(rel_paths, dtype=str),
                    done_count=np.asarray(processed, dtype=np.int64),
                )
                last_ckpt = processed

    elapsed = time.perf_counter() - t0
    meta["elapsed_sec"] = round(elapsed, 2)
    meta["yolo_hit_rate"] = round(float(used_yolo.mean()), 4) if not no_yolo else 0.0

    save_feature_bundle(out_path, embeddings, bboxes, used_yolo, rel_paths, meta)
    if partial_path.exists():
        partial_path.unlink()

    print(
        f"[DONE] Saved {out_path} ({n_total:,} x {embed_dim} float16, "
        f"yolo_hit={meta['yolo_hit_rate']*100:.1f}%, time={elapsed:.1f}s)"
    )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract YOLO-cropped vehicle embeddings.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=BASE_DIR / "data_manifests" / "local_vmmrdb_manifest.csv",
    )
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\Projects\bitirme\data"))
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "features")
    parser.add_argument(
        "--backbone",
        choices=list(BACKBONE_DIMS.keys()),
        default="openclip_vitb16",
    )
    parser.add_argument("--yolo-weights", type=Path, default=BASE_DIR / "yolov8n.pt")
    parser.add_argument("--no-yolo", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--splits", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_features(
        manifest_path=args.manifest,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        backbone_name=args.backbone,
        yolo_weights=args.yolo_weights,
        no_yolo=args.no_yolo,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        max_samples=args.max_samples,
        splits=args.splits,
    )


if __name__ == "__main__":
    main()
