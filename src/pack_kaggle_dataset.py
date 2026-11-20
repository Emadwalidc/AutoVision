"""Package clean VMMRdb images and manifest into a Kaggle-safe uncompressed ZIP."""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import zipfile
from pathlib import Path
from typing import Any

from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
KAGGLE_SAFE_RE = re.compile(r"^[A-Za-z0-9_./-]+$")


def build_safe_archive_entries(
    manifest_path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    """Reads clean manifest rows and assigns Kaggle-safe sharded ASCII paths."""
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        base_fields = list(reader.fieldnames or [])
        fieldnames = base_fields if "kaggle_rel_path" in base_fields else [*base_fields, "kaggle_rel_path"]

        clean_rows: list[dict[str, str]] = []
        idx = 0
        for row in reader:
            if str(row.get("is_clean", "")).strip().lower() != "true":
                continue
            rel_path = row["rel_path"].replace("\\", "/")
            ext = Path(rel_path).suffix.lower() or ".jpg"
            kaggle_rel = f"images/{idx // 1000:03d}/{idx:06d}{ext}"
            if not KAGGLE_SAFE_RE.fullmatch(kaggle_rel):
                raise ValueError(f"Generated unsafe archive path: {kaggle_rel!r}")
            row_out = dict(row)
            row_out["rel_path"] = rel_path
            row_out["kaggle_rel_path"] = kaggle_rel
            clean_rows.append(row_out)
            idx += 1

    return fieldnames, clean_rows


def _render_manifest_bytes(fieldnames: list[str], rows: list[dict[str, str]]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def pack_kaggle_dataset(
    manifest_path: Path,
    data_dir: Path,
    out_zip: Path,
    dry_run: bool = False,
    repack_from_zip: Path | None = None,
) -> dict[str, Any]:
    """Packages manifest + clean images into a Kaggle-safe ZIP_STORED archive."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    fieldnames, clean_rows = build_safe_archive_entries(manifest_path)
    manifest_bytes = _render_manifest_bytes(fieldnames, clean_rows)

    missing = 0
    total_bytes = len(manifest_bytes)
    sources: list[tuple[str, str, int]] = []

    if repack_from_zip is not None:
        if not repack_from_zip.is_file():
            raise FileNotFoundError(f"Source ZIP not found for repack: {repack_from_zip}")
        with zipfile.ZipFile(repack_from_zip, "r") as src_zf:
            src_infos = {info.filename: info for info in src_zf.infolist()}
            for row in clean_rows:
                rel = row["rel_path"]
                cand_legacy = f"images/{rel}"
                if cand_legacy in src_infos:
                    src_name = cand_legacy
                elif rel in src_infos:
                    src_name = rel
                elif row["kaggle_rel_path"] in src_infos:
                    src_name = row["kaggle_rel_path"]
                else:
                    missing += 1
                    continue
                sz = src_infos[src_name].file_size
                total_bytes += sz
                sources.append((src_name, row["kaggle_rel_path"], sz))
    else:
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        for row in clean_rows:
            rel = row["rel_path"]
            src = data_dir / rel
            if not src.exists():
                missing += 1
                continue
            sz = int(row["file_size"]) if row.get("file_size") else src.stat().st_size
            total_bytes += sz
            sources.append((str(src), row["kaggle_rel_path"], sz))

    total_gb = total_bytes / (1024**3)
    summary = {
        "clean_images": len(sources),
        "missing_images": missing,
        "total_bytes": total_bytes,
        "total_gb": round(total_gb, 3),
        "out_zip": str(out_zip),
        "dry_run": dry_run,
        "repack": repack_from_zip is not None,
    }

    if dry_run:
        print(
            f"[DRY-RUN] Clean images: {len(sources):,} (missing={missing}) | "
            f"Estimated ZIP size: {total_gb:.2f} GB -> {out_zip}"
        )
        return summary

    if missing > 0:
        raise FileNotFoundError(f"{missing} clean images were missing during pack")

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp_zip = out_zip.with_name(f".{out_zip.stem}.tmp{out_zip.suffix}")
    replaced = False
    try:
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf_out:
            zf_out.writestr("local_vmmrdb_manifest.csv", manifest_bytes)
            if repack_from_zip is not None:
                with zipfile.ZipFile(repack_from_zip, "r") as zf_in:
                    for src_name, dst_name, _ in tqdm(sources, desc="Repacking ZIP"):
                        zf_out.writestr(dst_name, zf_in.read(src_name))
            else:
                for src_path, dst_name, _ in tqdm(sources, desc="Packing ZIP"):
                    zf_out.write(src_path, arcname=dst_name)

        os.replace(tmp_zip, out_zip)
        replaced = True
    finally:
        if not replaced and tmp_zip.exists():
            tmp_zip.unlink()

    actual_gb = out_zip.stat().st_size / (1024**3)
    summary["actual_gb"] = round(actual_gb, 3)
    print(f"[DONE] Packed {len(sources):,} images + manifest into {out_zip} ({actual_gb:.2f} GB)")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack clean VMMRdb subset for Kaggle upload.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=BASE_DIR / "data_manifests" / "local_vmmrdb_manifest.csv",
    )
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\Projects\bitirme\data"))
    parser.add_argument(
        "--out-zip",
        type=Path,
        default=BASE_DIR / "kaggle" / "vmmrdb_clean_89k.zip",
    )
    parser.add_argument(
        "--repack-from-zip",
        type=Path,
        default=None,
        help="Optional existing ZIP to fast-repack into Kaggle-safe paths.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pack_kaggle_dataset(
        manifest_path=args.manifest,
        data_dir=args.data_dir,
        out_zip=args.out_zip,
        dry_run=args.dry_run,
        repack_from_zip=args.repack_from_zip,
    )


if __name__ == "__main__":
    main()
