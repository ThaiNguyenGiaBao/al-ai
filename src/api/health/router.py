# health.router.py
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Optional

from .schema import DetectRequest, DetectResponse, DetectBytesRequest
from .service import HealthService

router = APIRouter(prefix="/api/health", tags=["health"])

# create a singleton HealthService instance (or use DI in your app startup)
health_service = HealthService()


@router.post("/detect", response_model=DetectResponse)
async def detect_endpoint(payload: DetectRequest = None):
    if payload is None or not payload.image_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="image_url is required"
        )

    try:
        print("Received detection request for URL:", payload.image_url)
        analysis_vn = health_service.detect_from_url(payload.image_url)

        return DetectResponse(analysis_vn=analysis_vn)
    except HTTPException:
        raise
    except Exception as exc:
        # log exception server-side in production
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}")


@router.post("/detect-bytes", response_model=DetectResponse)
async def detect_bytes_endpoint(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type"
        )

    try:
        analysis_vn = health_service.detect_from_bytes(await image.read())
        return DetectResponse(analysis_vn=analysis_vn)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}")
