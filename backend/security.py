"""Boundary validation, upload sanitization, and path traversal guards."""

from __future__ import annotations

import io
from pathlib import Path, PurePosixPath, PureWindowsPath

from PIL import Image, ImageOps, UnidentifiedImageError


class AutoVisionApiError(Exception):
    """Typed domain exception mapped to structured ErrorResponse JSON."""

    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.error_code = error_code
        self.message = message
        self.detail = detail


class PayloadTooLargeError(AutoVisionApiError):
    def __init__(self, message: str = "Uploaded file exceeds maximum allowed size.") -> None:
        super().__init__(status_code=413, error_code="PAYLOAD_TOO_LARGE", message=message)


class InvalidImageError(AutoVisionApiError):
    def __init__(self, message: str = "Invalid or corrupted image upload.", detail: str | None = None) -> None:
        super().__init__(status_code=400, error_code="INVALID_IMAGE", message=message, detail=detail)


class UnsafePathError(AutoVisionApiError):
    def __init__(self, message: str = "Invalid or unsafe relative path.") -> None:
        super().__init__(status_code=403, error_code="UNSAFE_PATH", message=message)


class ResourceNotFoundError(AutoVisionApiError):
    def __init__(self, message: str = "Requested resource was not found.") -> None:
        super().__init__(status_code=404, error_code="NOT_FOUND", message=message)


def decode_uploaded_image(
    raw_bytes: bytes,
    max_bytes: int,
    max_pixels: int,
) -> Image.Image:
    """Validates byte size and resolution, applies EXIF orientation, and returns RGB image."""
    if not raw_bytes:
        raise InvalidImageError("Uploaded image file is empty.")
    if len(raw_bytes) > max_bytes:
        raise PayloadTooLargeError(f"Uploaded file ({len(raw_bytes):,} bytes) exceeds {max_bytes:,} byte limit.")

    src_im: Image.Image | None = None
    transposed_im: Image.Image | None = None
    try:
        src_im = Image.open(io.BytesIO(raw_bytes))
        w, h = src_im.size
        if w <= 0 or h <= 0:
            raise InvalidImageError("Decoded image has non-positive dimensions.")
        if (w * h) > max_pixels:
            raise InvalidImageError(
                f"Image resolution {w}x{h} ({w * h:,} px) exceeds {max_pixels:,} pixel limit."
            )
        transposed_im = ImageOps.exif_transpose(src_im)
        rgb = transposed_im.convert("RGB")
        return rgb
    except AutoVisionApiError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
        raise InvalidImageError("Uploaded bytes could not be decoded as a supported image.", detail=str(exc)) from exc
    finally:
        if transposed_im is not None and transposed_im is not src_im:
            transposed_im.close()
        if src_im is not None:
            src_im.close()


def normalize_safe_gallery_rel_path(rel_path: str) -> str:
    """Validates and normalizes a relative gallery path, rejecting absolute or traversal paths."""
    if not rel_path or "\x00" in rel_path:
        raise UnsafePathError("Gallery path is empty or contains null bytes.")

    norm_rel = rel_path.replace("\\", "/").strip()
    if not norm_rel or norm_rel.startswith("/") or ":" in norm_rel:
        raise UnsafePathError(f"Absolute or drive-qualified paths are forbidden: {rel_path!r}")

    posix = PurePosixPath(norm_rel)
    win = PureWindowsPath(rel_path)
    if posix.is_absolute() or win.is_absolute() or ".." in posix.parts or posix.as_posix() in ("", "."):
        raise UnsafePathError(f"Path traversal is forbidden: {rel_path!r}")
    return posix.as_posix()


def resolve_safe_gallery_path(data_dir: Path, rel_path: str) -> Path:
    """Safely resolves a relative gallery image path strictly inside data_dir."""
    norm_rel = normalize_safe_gallery_rel_path(rel_path)
    posix = PurePosixPath(norm_rel)

    base = data_dir.resolve()
    resolved = (base / posix).resolve()
    if not resolved.is_relative_to(base):
        raise UnsafePathError(f"Resolved path escapes gallery directory: {rel_path!r}")
    if not resolved.is_file():
        raise ResourceNotFoundError(f"Gallery image not found: {norm_rel}")
    return resolved
