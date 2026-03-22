from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Detection(BaseModel):
    label: str
    confidence: Optional[float] = None
    box_2d: List[float] = Field(
        ..., description="[ymin, xmin, ymax, xmax] normalized 0..1000"
    )


class DetectRequest(BaseModel):
    image_url: str = Field(..., description="URL of the image to analyze")


class DetectBytesRequest(BaseModel):
    image_bytes: bytes = Field(..., description="Bytes of the image to analyze")


class DetectResponse(BaseModel):
    analysis_vn: Optional[dict] = Field(
        None, description="Analysis results from the model"
    )


class QuizGenerationRequest(BaseModel):
    source_text: str
    num_questions: int


class QuizQuestion(BaseModel):
    question: str
    choices: List[str]
    correct_index: int
    explanations: List[str]


class QuizGenerationResponse(BaseModel):
    questions: List[QuizQuestion]


class TreeJourneyRequest(BaseModel):
    raw_data:str


class GrowthStage(BaseModel):
    index: int = 1
    stageDisplayName: str = ""
    stageImage: str = ""
    stageImageCaption: str = ""
    studentStory: str = ""
    gardenNote: str = ""
    layoutClass: str = ""
    contentSpacingClass: str = ""
    emptySpacingClass: str = ""
    cardAccentClass: str = ""
    timelineDotClass: str = ""
    noteMarginClass: str = ""


class TreeJourneyResponse(BaseModel):
    heroImage: str = ""
    heroAlt: str = ""
    heroBadge: str = ""
    heroTitle: str = ""
    heroDescription: str = ""
    breed: str = ""
    cultivar: str = ""
    farmingMethod: str = ""
    growDuration: int = 0
    growthStages: List[GrowthStage] = Field(default_factory=list)
    harvestTitle: str = ""
    estimatedFruitWeight: int = 0
    fruitWeight: int = 0
    harvestBadge: str = ""
    harvestQuote: str = ""
    footerTitle: str = ""
    footerDescription: str = ""

