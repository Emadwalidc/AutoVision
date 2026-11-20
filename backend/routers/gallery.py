"""Safe resized gallery thumbnail serving endpoint with bounded LRU cache."""

from __future__ import annotations

import asyncio
import io
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Query, Request, Response
from PIL import Image, ImageOps

from backend.security import resolve_safe_gallery_path

router = APIRouter(prefix="/api/v1/gallery", tags=["Gallery"])


@lru_cache(maxsize=256)
def _render_thumbnail_jpeg(path_str: str, max_width: int, quality: int) -> bytes:
    src: Image.Image | None = None
    im: Image.Image | None = None
    resized: Image.Image | None = None
    try:
        src = Image.open(Path(path_str))
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
        if src is not None:
            src.close()


@router.get("/thumbnail", summary="Serve safe resized JPEG thumbnail for a similar reference vehicle")
async def get_gallery_thumbnail(
    request: Request,
    rel_path: str = Query(..., description="Manifest-relative gallery image path"),
    max_width: int = Query(360, ge=64, le=1024, description="Maximum thumbnail pixel width"),
) -> Response:
    settings = request.app.state.settings
    safe_path = resolve_safe_gallery_path(settings.data_dir, rel_path)
    jpeg_bytes = await asyncio.to_thread(
        _render_thumbnail_jpeg,
        str(safe_path),
        int(max_width),
        int(settings.thumbnail_jpeg_quality),
    )
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )
