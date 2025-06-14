"""Health check endpoints for media-to-text microservice."""

from fastapi import APIRouter
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    service: str


class ReadinessResponse(BaseModel):
    """Readiness check response model."""
    status: str
    dependencies: dict[str, str]


router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        service="media-to-text"
    )


@router.get("/readyz", response_model=ReadinessResponse)
async def readiness_check() -> ReadinessResponse:
    """Readiness check with dependency verification."""
    # TODO: Add actual dependency checks (Redis, OpenAI API, etc.)
    dependencies = {
        "redis": "connected",  # Will be implemented later
        "openai": "available",  # Will be implemented later
        "ffmpeg": "installed"   # Will be implemented later
    }
    
    return ReadinessResponse(
        status="ready",
        dependencies=dependencies
    )