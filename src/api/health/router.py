# health.router.py
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status


from .schema import (
    DetectRequest,
    DetectResponse,
    QuizGenerationRequest,
    QuizGenerationResponse,
    TreeJourneyRequest,
    TreeJourneyResponse,
)
from .service import HealthService

router = APIRouter(
    prefix="/api",
)

# create a singleton HealthService instance (or use DI in your app startup)
health_service = HealthService()


@router.post("/health/detect", response_model=DetectResponse)
async def detect_endpoint(payload: DetectRequest = None):
    if payload is None or not payload.image_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="image_url is required"
        )

    try:
        print("Received detection request for URL:", payload.image_url)
        analysis_vn = health_service.tree_disease_diagnosis_from_url(payload.image_url)

        return DetectResponse(analysis_vn=analysis_vn)
    except HTTPException:
        raise
    except Exception as exc:
        # log exception server-side in production
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}")


@router.post("/health/detect-bytes", response_model=DetectResponse)
async def detect_bytes_endpoint(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type"
        )

    try:
        analysis_vn = health_service.tree_disease_diagnosis_from_bytes(
            await image.read()
        )
        return DetectResponse(analysis_vn=analysis_vn)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}")


@router.post("/quiz", response_model=QuizGenerationResponse)
async def quiz_endpoint(payload: QuizGenerationRequest):
    if (
        not payload.source_text
        or payload.num_questions < 1
        or payload.num_questions > 30
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="source_text is required and num_questions must be in [1,30]",
        )
    try:
        result = health_service.quiz_generation(
            payload.source_text, payload.num_questions
        )
        return QuizGenerationResponse.model_validate(result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quiz generation failed: {exc}",
        )


@router.post("/tree-journey", response_model=TreeJourneyResponse)
async def extract_tree_journey_endpoint(payload: TreeJourneyRequest):
    try:
        result = health_service.extract_tree_journey(payload.raw_data)
        return TreeJourneyResponse.model_validate(result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tree journey extraction failed: {exc}",
        )
