"""API routers for AutoVision FastAPI backend."""

from backend.routers.analyze import router as analyze_router
from backend.routers.gallery import router as gallery_router
from backend.routers.health import router as health_router

__all__ = ["analyze_router", "gallery_router", "health_router"]
