from pydantic import BaseModel, Field
from typing import Literal


class DiagnosisResponse(BaseModel):
    """Response model for skin diagnosis results."""
    status: Literal["healthy", "unhealthy"]
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    disease: str = Field(description="Name of the disease if unhealthy, 'None' if healthy")
    disease_confidence: float = Field(ge=0.0, le=1.0, description="Disease confidence score between 0.0 and 1.0")


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str = Field(description="Error message")
