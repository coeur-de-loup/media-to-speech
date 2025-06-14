"""Data models for media-to-text microservice."""

import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class JobState(str, Enum):
    """Job processing states."""
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobStatus(Enum):
    """Job status enumeration (alias for JobState for compatibility)."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TranscriptionRequest(BaseModel):
    """Request model for creating a transcription job."""
    file_path: str = Field(..., description="Absolute or container-relative path to media file")
    language: str = Field(default="en", description="Language code for transcription")
    async_processing: bool = Field(default=True, description="Process asynchronously")

    @validator('file_path')
    def validate_file_path(cls, v):
        """Validate file path for security."""
        # Check for directory traversal attempts
        if '..' in v or '~' in v:
            raise ValueError("Directory traversal attempts not allowed")
        
        # Ensure it's an absolute path or starts with /app/data
        if not (v.startswith('/') or v.startswith('/app/data')):
            raise ValueError("File path must be absolute or within /app/data")
        
        return v

    @validator('language')
    def validate_language(cls, v):
        """Validate language code."""
        # Basic language code validation (ISO 639-1)
        if len(v) < 2 or len(v) > 5:
            raise ValueError("Language code must be 2-5 characters")
        return v.lower()


class TranscriptionResponse(BaseModel):
    """Response model for transcription job creation."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Current job status")
    message: str = Field(..., description="Status message")
    original_filename: Optional[str] = Field(None, description="Original uploaded filename")
    language: str = Field(..., description="Language code for transcription")
    request_id: Optional[str] = Field(None, description="Request ID for tracing")
    trace_id: Optional[str] = Field(None, description="Trace ID for observability")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Job creation timestamp")


class JobStatusResponse(BaseModel):
    """Response model for job status queries."""
    job_id: str = Field(..., description="Unique job identifier")
    state: JobState = Field(..., description="Current job state")
    progress: float = Field(default=0.0, description="Progress percentage (0.0-1.0)")
    chunks_done: int = Field(default=0, description="Number of chunks processed")
    chunks_total: int = Field(default=0, description="Total number of chunks")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    created_at: Optional[datetime] = Field(default=None, description="Job creation timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Job completion timestamp")


class TranscriptChunk(BaseModel):
    """Model for a single transcript chunk."""
    index: int = Field(..., description="Chunk index")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text")


class TranscriptResult(BaseModel):
    """Model for complete transcript result."""
    job_id: str = Field(..., description="Job identifier")
    chunks: List[TranscriptChunk] = Field(..., description="List of transcript chunks")
    full_text: str = Field(..., description="Complete concatenated transcript")
    duration: float = Field(..., description="Total media duration in seconds")
    language: str = Field(..., description="Detected/specified language")


class JobCancelResponse(BaseModel):
    """Response model for job cancellation."""
    job_id: str = Field(..., description="Job identifier")
    message: str = Field(..., description="Cancellation message")


class JobMetadata(BaseModel):
    """Model for job metadata stored in Redis."""
    job_id: str = Field(..., description="Unique job identifier")
    file_path: str = Field(..., description="Original file path")
    language: str = Field(..., description="Language code")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    state: JobState = Field(default=JobState.QUEUED, description="Current state")
    chunks_total: int = Field(default=0, description="Total chunks to process")
    chunks_done: int = Field(default=0, description="Chunks completed")
    error_message: Optional[str] = Field(default=None, description="Error message")
    updated_at: Optional[datetime] = None
    
    @property
    def progress(self) -> float:
        """Calculate progress percentage."""
        if self.chunks_total == 0:
            return 0.0
        return (self.chunks_done / self.chunks_total) * 100
    
    def to_dict(self) -> dict:
        """Convert to dictionary for Redis storage."""
        return {
            "job_id": self.job_id,
            "file_path": self.file_path,
            "language": self.language,
            "created_at": self.created_at.isoformat(),
            "state": self.state.value,
            "chunks_total": str(self.chunks_total),
            "chunks_done": str(self.chunks_done),
            "error_message": self.error_message or "",
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "progress": self.progress
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "JobMetadata":
        """Create from dictionary retrieved from Redis."""
        return cls(
            job_id=data["job_id"],
            file_path=data["file_path"],
            language=data["language"],
            created_at=datetime.fromisoformat(data["created_at"]),
            state=JobState(data["state"]),
            chunks_total=int(data["chunks_total"]),
            chunks_done=int(data["chunks_done"]),
            error_message=data["error_message"] if data["error_message"] else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
        )


class JobResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str
    progress: float = 0.0
    message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    error_message: Optional[str] = None
    transcript: Optional[str] = None
    metadata: Optional[dict] = None


class JobListResponse(BaseModel):
    """Response model for job listing."""
    jobs: List[JobResponse]
    total: int
    status_filter: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "0.1.0"
    uptime_seconds: Optional[float] = None


class ReadinessResponse(BaseModel):
    """Response model for readiness checks."""
    ready: bool
    checks: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MetricsResponse(BaseModel):
    """Response model for metrics."""
    system: dict
    redis: dict
    jobs: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)