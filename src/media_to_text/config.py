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
    redis_url: str = Field(default="redis://redis:6379", env="REDIS_URL")
    redis_jobs_prefix: str = Field(default="jobs", env="REDIS_JOBS_PREFIX")
    redis_ttl_days: int = Field(default=7, env="REDIS_TTL_DAYS")
    
    # File Processing Configuration
    temp_dir: str = Field(default="/tmp/media-to-text", env="TEMP_DIR")
    cleanup_temp_files: bool = Field(default=True, env="CLEANUP_TEMP_FILES")
    
    # Monitoring Configuration
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # FFmpeg Configuration (will connect to ffmpeg container)
    ffmpeg_container_url: str = Field(default="http://ffmpeg:8080", env="FFMPEG_CONTAINER_URL")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    enable_structured_logging: bool = Field(default=True, env="ENABLE_STRUCTURED_LOGGING")
    enable_axiom: bool = Field(default=False, env="ENABLE_AXIOM")
    axiom_token: Optional[str] = Field(default=None, env="AXIOM_TOKEN")
    axiom_dataset: str = Field(default="media-to-text-logs", env="AXIOM_DATASET")
    log_sensitive_data: bool = Field(default=False, env="LOG_SENSITIVE_DATA")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False