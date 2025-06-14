"""Job management endpoints for media-to-text microservice."""

import asyncio
import json
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Depends, status
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse

from media_to_text.models import (
    JobStatusResponse, 
    JobCancelResponse,
    JobState
)
from media_to_text.services.redis_service import get_redis_service, RedisService
from media_to_text.services.job_worker import get_job_worker, JobWorker


router = APIRouter()


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    stream: bool = Query(default=False, description="Enable SSE streaming"),
    redis_service: RedisService = Depends(get_redis_service)
) -> JobStatusResponse:
    """
    Get job status and progress.
    
    Args:
        job_id: The job identifier
        stream: If True, return Server-Sent Events stream for real-time updates
        redis_service: Redis service dependency
    
    Returns:
        JobStatusResponse with current job status and progress
    
    Raises:
        HTTPException: If job not found
    """
    job = await redis_service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    if stream:
        # Return streaming response for real-time updates
        return StreamingResponse(
            _job_status_stream(job_id, redis_service),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    else:
        # Return current status
        progress = 0.0
        if job.chunks_total > 0:
            progress = job.chunks_done / job.chunks_total
        
        return JobStatusResponse(
            job_id=job.job_id,
            state=job.state,
            progress=progress,
            chunks_done=job.chunks_done,
            chunks_total=job.chunks_total,
            error_message=job.error_message,
            created_at=job.created_at
        )


@router.get("/{job_id}/transcript")
async def get_transcription_result(
    job_id: str,
    redis_service: RedisService = Depends(get_redis_service)
) -> JSONResponse:
    """
    Get the completed transcription result for a job.
    
    Args:
        job_id: The job identifier
        redis_service: Redis service dependency
    
    Returns:
        JSON response with transcription text and metadata
    
    Raises:
        HTTPException: If job not found or not completed
    """
    # Check if job exists
    job = await redis_service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    # Check if job is completed
    if job.state != JobState.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job {job_id} is not completed (current state: {job.state})"
        )
    
    # Get transcription result from Redis
    result_key = f"transcript:{job_id}"
    result_data = await redis_service.redis.get(result_key)
    
    if not result_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transcription result not found for job {job_id}"
        )
    
    try:
        result = json.loads(result_data)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=result
        )
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to parse transcription result"
        )


async def _job_status_stream(job_id: str, redis_service: RedisService):
    """Generate Server-Sent Events for job status updates."""
    last_update_id = "0"
    
    try:
        while True:
            # Get latest job status
            job = await redis_service.get_job(job_id)
            if not job:
                yield f"event: error\ndata: Job {job_id} not found\n\n"
                break
            
            # Get updates from Redis stream
            updates = await redis_service.get_job_updates(job_id, last_update_id)
            
            for update in updates:
                # Send update as SSE
                yield f"event: job_update\ndata: {json.dumps(update)}\n\n"
                last_update_id = update["id"]
            
            # If job is in final state, send final status and close
            if job.state in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED]:
                progress = 0.0
                if job.chunks_total > 0:
                    progress = job.chunks_done / job.chunks_total
                
                final_status = {
                    "job_id": job.job_id,
                    "state": job.state.value,
                    "progress": progress,
                    "chunks_done": job.chunks_done,
                    "chunks_total": job.chunks_total,
                    "error_message": job.error_message,
                    "final": True
                }
                
                yield f"event: job_complete\ndata: {json.dumps(final_status)}\n\n"
                break
            
            # Wait before next poll
            await asyncio.sleep(1.0)
            
    except asyncio.CancelledError:
        # Client disconnected
        pass
    except Exception as e:
        yield f"event: error\ndata: Stream error: {str(e)}\n\n"


@router.delete("/{job_id}", response_model=JobCancelResponse)
async def cancel_job(
    job_id: str,
    redis_service: RedisService = Depends(get_redis_service)
) -> JobCancelResponse:
    """
    Cancel a running job.
    
    Args:
        job_id: The job identifier
        redis_service: Redis service dependency
    
    Returns:
        JobCancelResponse with cancellation status
    
    Raises:
        HTTPException: If job not found or cannot be cancelled
    """
    job = await redis_service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    # Check if job can be cancelled
    if job.state in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job {job_id} is already in final state: {job.state}"
        )
    
    # Cancel the job
    await redis_service.update_job_state(
        job_id, 
        JobState.CANCELLED, 
        error_message="Job cancelled by user request"
    )
    
    return JobCancelResponse(
        job_id=job_id,
        message=f"Job {job_id} has been cancelled"
    )


@router.get("/", response_model=List[JobStatusResponse])
async def list_jobs(
    state: Optional[JobState] = Query(default=None, description="Filter by job state"),
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of jobs to return"),
    redis_service: RedisService = Depends(get_redis_service)
) -> List[JobStatusResponse]:
    """
    List jobs with optional filtering by state.
    
    Args:
        state: Optional job state filter
        limit: Maximum number of jobs to return
        redis_service: Redis service dependency
    
    Returns:
        List of JobStatusResponse objects
    """
    jobs = await redis_service.list_jobs(state=state)
    
    # Limit results
    jobs = jobs[:limit]
    
    # Convert to response models
    job_responses = []
    for job in jobs:
        progress = 0.0
        if job.chunks_total > 0:
            progress = job.chunks_done / job.chunks_total
        
        job_responses.append(JobStatusResponse(
            job_id=job.job_id,
            state=job.state,
            progress=progress,
            chunks_done=job.chunks_done,
            chunks_total=job.chunks_total,
            error_message=job.error_message,
            created_at=job.created_at
        ))
    
    return job_responses


@router.post("/{job_id}/process")
async def manually_process_job(
    job_id: str,
    job_worker: JobWorker = Depends(get_job_worker)
) -> JSONResponse:
    """
    Manually trigger processing of a specific job (for testing/debugging).
    
    Args:
        job_id: The job identifier
        job_worker: Job worker dependency
    
    Returns:
        JSON response with processing status
    
    Raises:
        HTTPException: If job not found or cannot be processed
    """
    success = await job_worker.process_job_by_id(job_id)
    
    if success:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": f"Job {job_id} processing initiated",
                "job_id": job_id
            }
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not process job {job_id}. Check if job exists and is in QUEUED state."
        )


@router.get("/health")
async def job_service_health_check():
    """Health check endpoint for job service."""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "healthy",
            "service": "jobs",
            "endpoints": [
                "GET /jobs - List all jobs",
                "GET /jobs/{job_id} - Get job status",
                "GET /jobs/{job_id}?stream=true - Stream job updates",
                "GET /jobs/{job_id}/transcript - Get transcription result",
                "DELETE /jobs/{job_id} - Cancel job",
                "POST /jobs/{job_id}/process - Manually process job",
                "GET /jobs/health - Health check"
            ]
        }
    )