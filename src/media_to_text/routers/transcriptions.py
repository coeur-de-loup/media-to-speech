"""Transcription endpoints for media-to-text microservice."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


class TranscriptionRequest(BaseModel):
    """Request model for transcription jobs."""
    file_path: str
    language: str = "en"
    async_processing: bool = True


class TranscriptionResponse(BaseModel):
    """Response model for transcription job creation."""
    job_id: str
    state: str


router = APIRouter()


@router.post("/", response_model=TranscriptionResponse)
async def create_transcription(request: TranscriptionRequest) -> TranscriptionResponse:
    """Create a new transcription job."""
    # TODO: Implement actual transcription job creation
    # This is a placeholder implementation
    
    # Validate file path exists
    import os
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=400, detail="File not found")
    
    # Generate mock job ID (will be replaced with actual implementation)
    import uuid
    job_id = str(uuid.uuid4())
    
    return TranscriptionResponse(
        job_id=job_id,
        state="QUEUED"
    )