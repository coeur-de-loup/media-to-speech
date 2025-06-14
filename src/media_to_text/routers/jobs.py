"""Job management endpoints for media-to-text microservice."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    state: str
    progress: float
    chunks_done: int
    chunks_total: int
    error_message: Optional[str] = None


class JobCancelResponse(BaseModel):
    """Response model for job cancellation."""
    job_id: str
    message: str


router = APIRouter()


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    stream: bool = Query(default=False, description="Enable SSE streaming")
) -> JobStatusResponse:
    """Get job status by ID."""
    # TODO: Implement actual job status retrieval from Redis
    # This is a placeholder implementation
    
    return JobStatusResponse(
        job_id=job_id,
        state="PROCESSING",
        progress=0.42,
        chunks_done=5,
        chunks_total=12
    )


@router.delete("/{job_id}", response_model=JobCancelResponse)
async def cancel_job(job_id: str) -> JobCancelResponse:
    """Cancel a job by ID."""
    # TODO: Implement actual job cancellation
    # This is a placeholder implementation
    
    return JobCancelResponse(
        job_id=job_id,
        message="Job cancellation requested"
    )