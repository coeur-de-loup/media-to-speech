"""Configuration settings for media-to-text microservice."""

import os
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_max_chunk_size_mb: int = Field(default=25, env="OPENAI_MAX_CHUNK_SIZE_MB")
    openai_max_parallel_requests: int = Field(default=8, env="OPENAI_MAX_PARALLEL_REQUESTS")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_jobs_prefix: str = Field(default="transcribe:jobs", env="REDIS_JOBS_PREFIX")
    redis_ttl_days: int = Field(default=7, env="REDIS_TTL_DAYS")
    
    # File Processing Configuration
    temp_dir: str = Field(default="/tmp", env="TEMP_DIR")
    cleanup_temp_files: bool = Field(default=True, env="CLEANUP_TEMP_FILES")
    
    # Security Configuration
    jwt_secret_key: Optional[str] = Field(default=None, env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=30, env="JWT_EXPIRE_MINUTES")
    
    # Monitoring Configuration
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # FFmpeg Configuration
    ffmpeg_binary: str = Field(default="ffmpeg", env="FFMPEG_BINARY")
    ffprobe_binary: str = Field(default="ffprobe", env="FFPROBE_BINARY")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"