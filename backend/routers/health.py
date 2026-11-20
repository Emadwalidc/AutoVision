"""Health check and label catalog endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from backend.ml.pipeline import AutoVisionPipeline
from backend.schemas import CatalogResponse, HealthResponse
from backend.security import AutoVisionApiError

router = APIRouter(tags=["System & Catalog"])


def get_pipeline_from_request(request: Request) -> AutoVisionPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise AutoVisionApiError(
            status_code=503,
            error_code="MODEL_NOT_READY",
            message="AutoVision inference pipeline is not initialized.",
        )
    return pipeline


@router.get("/health", response_model=HealthResponse, summary="Service health and model readiness")
async def health_check(request: Request) -> HealthResponse:
    return get_pipeline_from_request(request).get_health()


@router.get("/api/v1/catalog", response_model=CatalogResponse, summary="Supported makes, models, body types, colors, and year range")
async def get_catalog(request: Request) -> CatalogResponse:
    return get_pipeline_from_request(request).get_catalog()
