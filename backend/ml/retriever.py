"""Cosine prototype matching and top-K visually similar vehicle retrieval."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

import numpy as np

from backend.schemas import FewShotCandidate, SimilarVehicle, format_display_name


class VehicleRetriever(Protocol):
    """Protocol for swappable prototype and visual similarity retrieval backends."""

    @property
    def gallery_size(self) -> int:
        ...

    @property
    def prototype_classes(self) -> list[str]:
        ...

    def match_prototype(self, embedding: np.ndarray) -> FewShotCandidate | None:
        ...

    def search_similar(self, embedding: np.ndarray, top_k: int = 5) -> list[SimilarVehicle]:
        ...


class NumpyCosineRetriever:
    """Fast in-memory exact cosine retriever using normalized Stage 4 arrays."""

    def __init__(self, models_dir: Path, manifest_path: Path, data_dir: Path) -> None:
        self._protos: np.ndarray | None = None
        self._proto_classes: list[str] = []
        self._proto_makes: list[str] = []
        self._proto_bodies: list[str] = []
        self._proto_split_types: list[str] = []

        proto_path = models_dir / "prototypes_342.npz"
        if proto_path.is_file():
            with np.load(proto_path, allow_pickle=False) as d:
                p = d["prototypes"].astype(np.float32)
                p /= np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
                self._protos = p
                self._proto_classes = [str(s) for s in d["class_names"]]
                self._proto_makes = [str(s) for s in d["makes"]]
                self._proto_bodies = [str(s) for s in d["body_types"]]
                self._proto_split_types = [str(s) for s in d["split_types"]]

        self._gallery_embs: np.ndarray | None = None
        self._gallery_meta: list[dict] = []

        feat_path = models_dir / "features_finetuned_openclip_vitb16.npz"
        if feat_path.is_file() and manifest_path.is_file():
            meta_lookup: dict[str, dict] = {}
            with open(manifest_path, "r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    if str(row.get("is_clean", "")).strip().lower() != "true":
                        continue
                    if row.get("split") not in ("train", "few_shot"):
                        continue
                    rp = row["rel_path"].replace("\\", "/")
                    yr_raw = row.get("raw_year", "").strip()
                    yr_valid = str(row.get("year_valid", "")).strip().lower() == "true"
                    meta_lookup[rp] = {
                        "rel_path": rp,
                        "canonical_class": row["canonical_class"],
                        "make": row["make"],
                        "year": int(yr_raw) if (yr_raw.isdigit() and yr_valid) else None,
                        "body_type": row.get("body_type", "") or "",
                    }

            with np.load(feat_path, allow_pickle=False) as d:
                rels = [str(p).replace("\\", "/") for p in d["rel_paths"]]
                keep_idx: list[int] = []
                meta_list: list[dict] = []
                for idx, rp in enumerate(rels):
                    rec = meta_lookup.get(rp)
                    if rec is not None:
                        keep_idx.append(idx)
                        meta_list.append(rec)
                if keep_idx:
                    embs = d["embeddings"][keep_idx].astype(np.float32)
                    embs /= np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
                    self._gallery_embs = embs
                    self._gallery_meta = meta_list

        self._has_data_dir = data_dir.is_dir()

    @property
    def gallery_size(self) -> int:
        return len(self._gallery_meta)

    @property
    def prototype_classes(self) -> list[str]:
        return list(self._proto_classes)

    def match_prototype(self, embedding: np.ndarray) -> FewShotCandidate | None:
        if self._protos is None or len(self._proto_classes) == 0:
            return None
        q = embedding.astype(np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        sims = self._protos @ q
        best_i = int(np.argmax(sims))
        cls_slug = self._proto_classes[best_i]
        return FewShotCandidate(
            canonical_class=cls_slug,
            display_name=format_display_name(cls_slug),
            make=self._proto_makes[best_i],
            body_type=self._proto_bodies[best_i],
            similarity=round(float(np.clip(sims[best_i], -1.0, 1.0)), 4),
            is_few_shot=(self._proto_split_types[best_i] == "few_shot"),
        )

    def search_similar(self, embedding: np.ndarray, top_k: int = 5) -> list[SimilarVehicle]:
        if self._gallery_embs is None or top_k <= 0 or len(self._gallery_meta) == 0:
            return []
        q = embedding.astype(np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        sims = self._gallery_embs @ q
        k_eff = min(int(top_k), len(sims))
        cand_idx = np.argpartition(-sims, kth=k_eff - 1)[:k_eff]
        cand_idx = cand_idx[np.argsort(-sims[cand_idx])]

        items: list[SimilarVehicle] = []
        for idx in cand_idx:
            rec = self._gallery_meta[int(idx)]
            rp = rec["rel_path"]
            cls_slug = rec["canonical_class"]
            items.append(
                SimilarVehicle(
                    rel_path=rp,
                    canonical_class=cls_slug,
                    display_name=format_display_name(cls_slug),
                    make=rec["make"],
                    year=rec["year"],
                    body_type=rec["body_type"],
                    similarity=round(float(np.clip(sims[int(idx)], -1.0, 1.0)), 4),
                    thumbnail_url=f"/api/v1/gallery/thumbnail?rel_path={quote(rp, safe='/')}",
                )
            )
        return items
