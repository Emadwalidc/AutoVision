"""Vehicle image analysis endpoint."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, File, Query, Request, UploadFile

from backend.routers.health import get_pipeline_from_request
from backend.schemas import AnalyzeResponse
from backend.security import PayloadTooLargeError

router = APIRouter(prefix="/api/v1", tags=["Vehicle Recognition"])


@router.post("/analyze", response_model=AnalyzeResponse, summary="Detect vehicles and recognize 5 attributes + similar cars")
async def analyze_vehicle_image(
    request: Request,
    file: UploadFile = File(..., description="JPEG, PNG, or WebP vehicle photo"),
    top_k_similar: int = Query(5, ge=0, le=12, description="Number of visually similar reference cars"),
    use_tta: bool = Query(True, description="Enable horizontal-flip test-time augmentation"),
) -> AnalyzeResponse:
    settings = request.app.state.settings
    pipeline = get_pipeline_from_request(request)
    semaphore: asyncio.Semaphore = request.app.state.inference_semaphore

    try:
        raw_bytes = await file.read(settings.max_upload_bytes + 1)
    finally:
        await file.close()

    if len(raw_bytes) > settings.max_upload_bytes:
        raise PayloadTooLargeError(
            f"Uploaded file exceeds maximum allowed size of {settings.max_upload_bytes:,} bytes."
        )

    async with semaphore:
        return await asyncio.to_thread(pipeline.analyze_bytes, raw_bytes, top_k_similar, use_tta)
