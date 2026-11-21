"""FastAPI application factory and error boundaries for AutoVision backend."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import BackendSettings, load_settings
from backend.ml.pipeline import AutoVisionPipeline
from backend.routers import analyze_router, gallery_router, health_router
from backend.routers.gallery import GalleryThumbnailStore
from backend.schemas import ErrorResponse
from backend.security import AutoVisionApiError


def create_app(
    settings: BackendSettings | None = None,
    pipeline: AutoVisionPipeline | None = None,
) -> FastAPI:
    """Builds a configured AutoVision FastAPI application instance."""
    cfg = settings or load_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.settings = cfg
        application.state.inference_semaphore = asyncio.Semaphore(cfg.max_concurrent_inferences)
        extra_zips = (
            (cfg.eu_extension_dir / "eu_gallery_thumbs.zip",)
            if cfg.eu_extension_dir is not None
            else ()
        )
        thumbnail_store = GalleryThumbnailStore(
            cfg.data_dir, cfg.gallery_zip_path, extra_zip_paths=extra_zips
        )
        application.state.thumbnail_store = thumbnail_store
        try:
            if pipeline is not None:
                application.state.pipeline = pipeline
            else:
                application.state.pipeline = await asyncio.to_thread(AutoVisionPipeline.from_settings, cfg)
            yield
        finally:
            application.state.pipeline = None
            application.state.thumbnail_store = None
            thumbnail_store.close()

    application = FastAPI(
        title="AutoVision Recognition API",
        description="Multi-task vehicle detection and 5-attribute recognition API (Make, Model, Year, Body Type, Color + Similar Cars).",
        version="2.0.0",
        lifespan=lifespan,
    )
    application.state.settings = cfg
    application.state.inference_semaphore = asyncio.Semaphore(cfg.max_concurrent_inferences)
    application.state.pipeline = pipeline
    application.state.thumbnail_store = None

    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @application.exception_handler(AutoVisionApiError)
    async def handle_api_error(_request: Request, exc: AutoVisionApiError) -> JSONResponse:
        payload = ErrorResponse(error_code=exc.error_code, message=exc.message, detail=exc.detail)
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @application.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        payload = ErrorResponse(
            error_code="INTERNAL_ERROR",
            message="Unexpected internal server error.",
            detail=type(exc).__name__,
        )
        return JSONResponse(status_code=500, content=payload.model_dump())

    application.include_router(health_router)
    application.include_router(analyze_router)
    application.include_router(gallery_router)
    return application


app = create_app()
