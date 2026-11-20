"""AutoVision FastAPI ML backend package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.main import create_app

__all__ = ["create_app"]


def __getattr__(name: str):
    if name == "create_app":
        from backend.main import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
