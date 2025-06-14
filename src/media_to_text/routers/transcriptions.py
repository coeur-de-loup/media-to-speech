"""Transcription endpoints for the media-to-text API."""

import os
import uuid
from typing import Dict, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends
from fastapi.responses import JSONResponse

from media_to_text.config import Settings
from media_to_text.logging import get_logger, set_trace_id, set_request_id
from media_to_text.models import TranscriptionRequest, TranscriptionResponse
from media_to_text.services.redis_service import get_redis_service, RedisService

router = APIRouter()


async def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


@router.post("/", response_model=TranscriptionResponse)
async def create_transcription(
    file: UploadFile = File(...),
    language: str = Form(default="en"),
    settings: Settings = Depends(get_settings),
    redis_service: RedisService = Depends(get_redis_service)
) -> TranscriptionResponse:
    """
    Create a new transcription job.
    
    Args:
        file: Audio/video file to transcribe
        language: Language code for transcription (default: "en")
        settings: Application settings
        redis_service: Redis service for job management
    
    Returns:
        TranscriptionResponse with job details
    """
    # Set up request tracing
    request_id = set_request_id()
    trace_id = set_trace_id()
    
    logger = get_logger("transcriptions")
    logger.info("Transcription request received", 
                request_id=request_id,
                trace_id=trace_id,
                filename=file.filename,
                content_type=file.content_type,
                language=language)
    
    try:
        # Validate file
        if not file.filename:
            logger.warning("File upload missing filename", 
                          request_id=request_id)
            raise HTTPException(
                status_code=400, 
                detail="File must have a filename"
            )
        
        # Check file size (basic check before processing)
        if hasattr(file, 'size') and file.size:
            file_size_mb = file.size / (1024 * 1024)
            logger.info("File size check", 
                       request_id=request_id,
                       file_size_mb=file_size_mb)
            
            # Basic size limit check (500MB)
            if file_size_mb > 500:
                logger.warning("File too large", 
                              request_id=request_id,
                              file_size_mb=file_size_mb)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large: {file_size_mb:.1f}MB. Maximum size is 500MB."
                )
        
        # Validate file type
        allowed_extensions = {'.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            logger.warning("Invalid file type", 
                          request_id=request_id,
                          filename=file.filename,
                          file_extension=file_ext,
                          allowed_extensions=list(allowed_extensions))
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Supported types: {', '.join(allowed_extensions)}"
            )
        
        # Validate language code
        supported_languages = [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 
            'ar', 'hi', 'tr', 'pl', 'nl', 'sv', 'da', 'no', 'fi'
        ]
        if language not in supported_languages:
            logger.warning("Unsupported language", 
                          request_id=request_id,
                          language=language,
                          supported_languages=supported_languages)
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language: {language}. Supported: {', '.join(supported_languages)}"
            )
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Create temp directory for file
        temp_dir = os.path.join(settings.temp_dir, job_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(temp_dir, file.filename)
        
        logger.info("Saving uploaded file", 
                   request_id=request_id,
                   job_id=job_id,
                   file_path=file_path)
        
        try:
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Verify file was saved
            if not os.path.exists(file_path):
                raise RuntimeError("Failed to save uploaded file")
            
            actual_size = os.path.getsize(file_path)
            logger.info("File saved successfully", 
                       request_id=request_id,
                       job_id=job_id,
                       file_path=file_path,
                       file_size_bytes=actual_size,
                       file_size_mb=actual_size / (1024 * 1024))
            
        except Exception as e:
            logger.error("Failed to save uploaded file", 
                        request_id=request_id,
                        job_id=job_id,
                        error=str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save uploaded file: {str(e)}"
            )
        
        # Verify file accessibility and permissions
        try:
            # Check if file is readable
            with open(file_path, "rb") as f:
                f.read(1)  # Try to read first byte
            
            logger.debug("File accessibility verified", 
                        request_id=request_id,
                        job_id=job_id)
            
        except Exception as e:
            logger.error("File accessibility check failed", 
                        request_id=request_id,
                        job_id=job_id,
                        error=str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Saved file is not accessible: {str(e)}"
            )
        
        # Create job in Redis
        try:
            await redis_service.create_job(
                job_id=job_id,
                file_path=file_path,
                language=language,
                original_filename=file.filename,
                request_id=request_id,
                trace_id=trace_id
            )
            
            logger.info("Transcription job created", 
                       request_id=request_id,
                       job_id=job_id,
                       file_path=file_path,
                       language=language)
            
        except Exception as e:
            logger.error("Failed to create job in Redis", 
                        request_id=request_id,
                        job_id=job_id,
                        error=str(e))
            
            # Clean up file on Redis failure
            try:
                os.remove(file_path)
                os.rmdir(temp_dir)
            except:
                pass
            
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create transcription job: {str(e)}"
            )
        
        # Return response
        response = TranscriptionResponse(
            job_id=job_id,
            status="queued",
            message="Transcription job created successfully",
            original_filename=file.filename,
            language=language,
            request_id=request_id,
            trace_id=trace_id
        )
        
        logger.info("Transcription job response created", 
                   request_id=request_id,
                   job_id=job_id,
                   status=response.status)
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error("Unexpected error in transcription endpoint", 
                    request_id=request_id,
                    error=str(e),
                    error_type=type(e).__name__)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/formats")
async def get_supported_formats() -> Dict[str, Any]:
    """
    Get supported file formats and languages.
    
    Returns:
        Dictionary with supported formats, languages, and limits
    """
    logger = get_logger("transcriptions")
    logger.debug("Supported formats requested")
    
    return {
        "supported_formats": [
            ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"
        ],
        "supported_languages": [
            "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh",
            "ar", "hi", "tr", "pl", "nl", "sv", "da", "no", "fi"
        ],
        "max_file_size_mb": 500,
        "max_chunk_size_mb": 25,
        "max_parallel_requests": 8,
        "rate_limiting": {
            "requests_per_second": 4,
            "burst_size": 8
        }
    }