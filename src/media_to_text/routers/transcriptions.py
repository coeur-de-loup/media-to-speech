"""Transcription endpoints for media-to-text microservice."""

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse

from media_to_text.models import (
    TranscriptionRequest, 
    TranscriptionResponse, 
    JobState
)
from media_to_text.services.redis_service import get_redis_service, RedisService


router = APIRouter()


async def validate_file_exists(file_path: str) -> Path:
    """Validate that the file exists and is readable."""
    try:
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File not found: {file_path}"
            )
        
        # Check if it's a file (not directory)
        if not path.is_file():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Path is not a file: {file_path}"
            )
        
        # Check read permissions
        if not os.access(path, os.R_OK):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"File is not readable: {file_path}"
            )
        
        # Check file size (basic validation)
        file_size = path.stat().st_size
        if file_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty"
            )
        
        # Check for supported file extensions
        supported_extensions = {
            '.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm',
            '.avi', '.mov', '.wmv', '.flv', '.mkv', '.aac', '.flac',
            '.ogg', '.wma', '.3gp'
        }
        
        if path.suffix.lower() not in supported_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {path.suffix}. Supported types: {', '.join(sorted(supported_extensions))}"
            )
        
        return path
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error validating file: {str(e)}"
        )


@router.post("/", response_model=TranscriptionResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_transcription(
    request: TranscriptionRequest,
    redis_service: RedisService = Depends(get_redis_service)
) -> TranscriptionResponse:
    """
    Create a new transcription job.
    
    This endpoint validates the input file, creates a job entry in Redis,
    and returns a job ID for tracking progress.
    
    Args:
        request: Transcription request containing file path and options
        redis_service: Redis service dependency for job management
    
    Returns:
        TranscriptionResponse with job_id and initial state
    
    Raises:
        HTTPException: If file validation fails or Redis operations fail
    """
    try:
        # Validate file existence and permissions
        file_path = await validate_file_exists(request.file_path)
        
        # Create job in Redis
        job_metadata = await redis_service.create_job(
            file_path=str(file_path),
            language=request.language
        )
        
        # Return response
        return TranscriptionResponse(
            job_id=job_metadata.job_id,
            state=job_metadata.state
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/health")
async def transcription_health_check():
    """Health check endpoint for transcription service."""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "healthy",
            "service": "transcriptions",
            "endpoints": [
                "POST /transcriptions - Create transcription job",
                "GET /transcriptions/health - Health check"
            ]
        }
    )