"""Safe resized gallery thumbnail serving endpoint with bounded LRU cache."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import io
import threading
import zipfile
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Query, Request, Response
from PIL import Image, ImageOps

from backend.security import normalize_safe_gallery_rel_path, resolve_safe_gallery_path

router = APIRouter(prefix="/api/v1/gallery", tags=["Gallery"])


def _resize_and_encode_jpeg(src: Image.Image, max_width: int, quality: int) -> bytes:
    im: Image.Image | None = None
    resized: Image.Image | None = None
    try:
        im = ImageOps.exif_transpose(src).convert("RGB")
        w, h = im.size
        target = im
        if w > max_width:
            new_h = max(1, int(round(h * (max_width / float(w)))))
            resized = im.resize((max_width, new_h), Image.Resampling.LANCZOS)
            target = resized
        buf = io.BytesIO()
        target.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    finally:
        if resized is not None:
            resized.close()
        if im is not None and im is not src:
            im.close()


def _render_thumbnail_from_bytes(raw_bytes: bytes, max_width: int, quality: int) -> bytes:
    src: Image.Image | None = None
    try:
        src = Image.open(io.BytesIO(raw_bytes))
        return _resize_and_encode_jpeg(src, max_width=max_width, quality=quality)
    finally:
        if src is not None:
            src.close()


@lru_cache(maxsize=256)
def _render_thumbnail_jpeg(path_str: str, max_width: int, quality: int) -> bytes:
    src: Image.Image | None = None
    try:
        src = Image.open(Path(path_str))
        return _resize_and_encode_jpeg(src, max_width=max_width, quality=quality)
    finally:
        if src is not None:
            src.close()


class GalleryThumbnailStore:
    """Thread-safe gallery thumbnail loader supporting multi-zip archives and directory fallback."""

    def __init__(
        self,
        data_dir: Path,
        gallery_zip_path: Path | None = None,
        extra_zip_paths: tuple[Path, ...] | list[Path] = (),
        cache_maxsize: int = 256,
    ) -> None:
        self.data_dir = data_dir
        self.gallery_zip_path = gallery_zip_path
        self._cache_maxsize = max(int(cache_maxsize), 1)
        self._cache: OrderedDict[tuple[str, int, int], bytes] = OrderedDict()
        self._lock = threading.Lock()
        self._zips: list[tuple[zipfile.ZipFile, set[str]]] = []

        candidates: list[Path] = []
        if gallery_zip_path is not None:
            candidates.append(gallery_zip_path)
        candidates.extend(extra_zip_paths)

        for zp in candidates:
            if zp is not None and zp.is_file() and zipfile.is_zipfile(zp):
                zf = zipfile.ZipFile(zp, "r")
                self._zips.append((zf, set(zf.namelist())))

    def close(self) -> None:
        with self._lock:
            for zf, members in self._zips:
                zf.close()
                members.clear()
            self._zips.clear()
            self._cache.clear()

    def get_thumbnail_jpeg(self, rel_path: str, max_width: int, quality: int) -> bytes:
        norm_rel = normalize_safe_gallery_rel_path(rel_path)
        cache_key = (norm_rel, int(max_width), int(quality))

        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                return cached

            raw_bytes: bytes | None = None
            for zf, members in self._zips:
                if norm_rel in members:
                    raw_bytes = zf.read(norm_rel)
                    break

        if raw_bytes is not None:
            rendered = _render_thumbnail_from_bytes(raw_bytes, max_width=max_width, quality=quality)
        else:
            safe_path = resolve_safe_gallery_path(self.data_dir, norm_rel)
            rendered = _render_thumbnail_jpeg(str(safe_path), int(max_width), int(quality))

        with self._lock:
            self._cache[cache_key] = rendered
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self._cache_maxsize:
                self._cache.popitem(last=False)
        return rendered


@router.get("/thumbnail", summary="Serve safe resized JPEG thumbnail for a similar reference vehicle")
async def get_gallery_thumbnail(
    request: Request,
    rel_path: str = Query(..., description="Manifest-relative gallery image path"),
    max_width: int = Query(360, ge=64, le=1024, description="Maximum thumbnail pixel width"),
) -> Response:
    settings = request.app.state.settings
    store: GalleryThumbnailStore | None = getattr(request.app.state, "thumbnail_store", None)
    temp_store: GalleryThumbnailStore | None = None
    if store is None:
        extra_zips = (
            (settings.eu_extension_dir / "eu_gallery_thumbs.zip",)
            if getattr(settings, "eu_extension_dir", None) is not None
            else ()
        )
        temp_store = GalleryThumbnailStore(
            settings.data_dir, settings.gallery_zip_path, extra_zip_paths=extra_zips
        )
        store = temp_store
    try:
        jpeg_bytes = await asyncio.to_thread(
            store.get_thumbnail_jpeg,
            rel_path,
            int(max_width),
            int(settings.thumbnail_jpeg_quality),
        )
    finally:
        if temp_store is not None:
            temp_store.close()
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )
