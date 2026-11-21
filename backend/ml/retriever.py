"""Cosine prototype matching and top-K visually similar vehicle retrieval."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import numpy as np

from backend.schemas import (
    FewShotCandidate,
    SimilarVehicle,
    format_canonical_display_name,
)


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


def _load_gallery_manifest_subset(
    feat_path: Path,
    manifest_path: Path,
) -> tuple[np.ndarray | None, list[dict[str, Any]]]:
    if not (feat_path.is_file() and manifest_path.is_file()):
        return None, []
    meta_lookup: dict[str, dict[str, Any]] = {}
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
        meta_list: list[dict[str, Any]] = []
        for idx, rp in enumerate(rels):
            rec = meta_lookup.get(rp)
            if rec is not None:
                keep_idx.append(idx)
                meta_list.append(rec)
        if not keep_idx:
            return None, []
        embs = d["embeddings"][keep_idx].astype(np.float32)
        embs /= np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
        return embs, meta_list


class NumpyCosineRetriever:
    """Fast in-memory exact cosine retriever using normalized prototype and gallery arrays."""

    def __init__(
        self,
        models_dir: Path,
        manifest_path: Path,
        data_dir: Path,
        make_display_names: dict[str, str] | None = None,
        model_display_names: dict[str, str] | None = None,
        eu_extension_dir: Path | None = None,
    ) -> None:
        self._protos: np.ndarray | None = None
        self._proto_classes: list[str] = []
        self._proto_makes: list[str] = []
        self._proto_bodies: list[str] = []
        self._proto_split_types: list[str] = []
        self._make_display_names = dict(make_display_names or {})
        self._model_display_names = dict(model_display_names or {})

        proto_v2 = models_dir / "prototypes_v2.npz"
        proto_v1 = models_dir / "prototypes_342.npz"
        proto_path = proto_v2 if proto_v2.is_file() else proto_v1
        if proto_path.is_file():
            with np.load(proto_path, allow_pickle=False) as d:
                p = d["prototypes"].astype(np.float32)
                p /= np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
                self._protos = p
                self._proto_classes = [str(s) for s in d["class_names"]]
                self._proto_makes = [str(s) for s in d["makes"]]
                self._proto_bodies = [str(s) for s in d["body_types"]]
                self._proto_split_types = [str(s) for s in d["split_types"]]

        self._gallery_embs, self._gallery_meta = _load_gallery_manifest_subset(
            models_dir / "features_finetuned_openclip_vitb16.npz",
            manifest_path,
        )

        if eu_extension_dir is not None and eu_extension_dir.is_dir():
            disp_path = eu_extension_dir / "eu_display_names.json"
            if disp_path.is_file():
                disp = json.loads(disp_path.read_text(encoding="utf-8"))
                self._make_display_names.update(disp.get("make_display_names", {}))
                self._model_display_names.update(disp.get("model_display_names", {}))

            eu_proto_path = eu_extension_dir / "eu_prototypes.npz"
            if eu_proto_path.is_file():
                with np.load(eu_proto_path, allow_pickle=False) as d:
                    ep = d["prototypes"].astype(np.float32)
                    ep /= np.maximum(np.linalg.norm(ep, axis=1, keepdims=True), 1e-12)
                    self._protos = np.vstack([self._protos, ep]) if self._protos is not None else ep
                    self._proto_classes.extend(str(s) for s in d["class_names"])
                    self._proto_makes.extend(str(s) for s in d["makes"])
                    self._proto_bodies.extend(str(s) for s in d["body_types"])
                    self._proto_split_types.extend(str(s) for s in d["split_types"])

            eu_embs, eu_meta = _load_gallery_manifest_subset(
                eu_extension_dir / "eu_gallery_embeddings.npz",
                eu_extension_dir / "eu_gallery_manifest.csv",
            )
            if eu_embs is not None and eu_meta:
                self._gallery_embs = (
                    np.vstack([self._gallery_embs, eu_embs])
                    if self._gallery_embs is not None
                    else eu_embs
                )
                self._gallery_meta.extend(eu_meta)

        self._unique_proto_classes = list(dict.fromkeys(self._proto_classes))
        self._has_data_dir = data_dir.is_dir()

    @property
    def gallery_size(self) -> int:
        return len(self._gallery_meta)

    @property
    def prototype_classes(self) -> list[str]:
        return list(self._unique_proto_classes)

    @property
    def make_display_names(self) -> dict[str, str]:
        return dict(self._make_display_names)

    @property
    def model_display_names(self) -> dict[str, str]:
        return dict(self._model_display_names)

    def _to_candidate(self, idx: int, sim: float) -> FewShotCandidate:
        cls_slug = self._proto_classes[idx]
        return FewShotCandidate(
            canonical_class=cls_slug,
            display_name=format_canonical_display_name(
                cls_slug, self._make_display_names, self._model_display_names
            ),
            make=self._proto_makes[idx],
            body_type=self._proto_bodies[idx],
            similarity=round(float(np.clip(sim, -1.0, 1.0)), 4),
            is_few_shot=(self._proto_split_types[idx] in ("few_shot", "eu_proto")),
        )

    def match_prototype(self, embedding: np.ndarray) -> FewShotCandidate | None:
        top_cands, _, _ = self.match_prototypes_detailed(embedding, top_k=1)
        return top_cands[0] if top_cands else None

    def match_prototypes_detailed(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
    ) -> tuple[list[FewShotCandidate], float, FewShotCandidate | None]:
        if self._protos is None or len(self._proto_classes) == 0:
            return [], -1.0, None
        q = embedding.astype(np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        sims = self._protos @ q
        order = np.argsort(-sims)

        top_cands: list[FewShotCandidate] = []
        seen_cls: set[str] = set()
        for raw_i in order:
            idx = int(raw_i)
            cls_slug = self._proto_classes[idx]
            if cls_slug in seen_cls:
                continue
            seen_cls.add(cls_slug)
            top_cands.append(self._to_candidate(idx, float(sims[idx])))
            if len(top_cands) >= max(int(top_k), 1):
                break

        closed_mask = np.array([st == "train" for st in self._proto_split_types], dtype=bool)
        open_mask = ~closed_mask
        best_closed_sim = float(np.max(sims[closed_mask])) if np.any(closed_mask) else -1.0

        best_open_cand: FewShotCandidate | None = None
        if np.any(open_mask):
            open_indices = np.flatnonzero(open_mask)
            best_open_pos = int(open_indices[int(np.argmax(sims[open_indices]))])
            best_open_cand = self._to_candidate(best_open_pos, float(sims[best_open_pos]))

        return top_cands, best_closed_sim, best_open_cand

    def knn_class_votes(
        self,
        embedding: np.ndarray,
        top_k: int = 12,
        min_sim: float = 0.45,
    ) -> dict[str, tuple[float, float]]:
        if self._gallery_embs is None or len(self._gallery_meta) == 0 or top_k <= 0:
            return {}
        q = embedding.astype(np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        sims = self._gallery_embs @ q
        k_eff = min(int(top_k), len(sims))
        cand_idx = np.argpartition(-sims, kth=k_eff - 1)[:k_eff]

        weights: dict[str, float] = {}
        max_sims: dict[str, float] = {}
        total_w = 0.0
        for idx in cand_idx:
            s_val = float(sims[int(idx)])
            if s_val < min_sim:
                continue
            w = s_val * s_val
            cls_name = self._gallery_meta[int(idx)]["canonical_class"]
            weights[cls_name] = weights.get(cls_name, 0.0) + w
            if s_val > max_sims.get(cls_name, -1.0):
                max_sims[cls_name] = s_val
            total_w += w

        if total_w <= 0.0:
            return {}
        return {c: (w / total_w, max_sims[c]) for c, w in weights.items()}

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
                    display_name=format_canonical_display_name(
                        cls_slug, self._make_display_names, self._model_display_names
                    ),
                    make=rec["make"],
                    year=rec["year"],
                    body_type=rec["body_type"],
                    similarity=round(float(np.clip(sims[int(idx)], -1.0, 1.0)), 4),
                    thumbnail_url=f"/api/v1/gallery/thumbnail?rel_path={quote(rp, safe='/')}",
                )
            )
        return items
