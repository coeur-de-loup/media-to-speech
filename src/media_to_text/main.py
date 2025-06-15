"""Main FastAPI application for media-to-text microservice."""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from media_to_text.config import Settings
from media_to_text.logging import setup_logging, get_logger, LoggingMiddleware
from media_to_text.routers import transcriptions, jobs, health
from media_to_text.services.redis_service import init_redis_service, close_redis_service, get_redis_service
from media_to_text.services.job_worker import init_job_worker, close_job_worker
from media_to_text.services.cleanup_service import init_cleanup_service
from media_to_text.services.ffmpeg_service import init_ffmpeg_service
from media_to_text.services.openai_service import init_openai_service
from media_to_text.services.transcript_service import init_transcript_processor


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Initialize settings first
    settings = Settings()
    
    # Setup structured logging as early as possible
    setup_logging(
        log_level=settings.log_level,
        enable_axiom=settings.enable_axiom,
        axiom_token=settings.axiom_token,
        axiom_dataset=settings.axiom_dataset
    )
    
    logger = get_logger("startup")
    logger.info("Starting media-to-text microservice", version="0.1.0")
    
    # Initialize services
    try:
        # Initialize Redis service
        logger.info("Initializing Redis service", redis_url=settings.redis_url)
        redis_service = await init_redis_service(settings)
        logger.info("Redis service initialized successfully")
        
        # Initialize FFmpeg service (required by job worker)
        logger.info("Initializing FFmpeg service")
        ffmpeg_service = init_ffmpeg_service(settings)
        logger.info("FFmpeg service initialized successfully")
        
        # Initialize OpenAI service (required by job worker)
        logger.info("Initializing OpenAI service")
        openai_service = init_openai_service(settings)
        logger.info("OpenAI service initialized successfully")
        
        # Initialize transcript processor (required by job worker)
        logger.info("Initializing transcript processor")
        transcript_processor = init_transcript_processor()
        logger.info("Transcript processor initialized successfully")
        
        # Initialize cleanup service
        logger.info("Initializing cleanup service")
        cleanup_service = await init_cleanup_service(settings, redis_service)
        logger.info("Cleanup service initialized successfully")
        
        # Initialize job worker (depends on all above services)
        logger.info("Initializing job worker")
        job_worker = await init_job_worker(settings, redis_service)
        
        # Connect cleanup service to job worker
        await job_worker.set_cleanup_service(cleanup_service)
        logger.info("Job worker initialized with cleanup service")
        
        # Run initial maintenance cycle to clean up any orphaned files
        try:
            logger.info("Running initial cleanup maintenance cycle")
            maintenance_stats = await cleanup_service.run_maintenance_cycle()
            logger.info("Initial cleanup maintenance completed", **maintenance_stats)
        except Exception as e:
            logger.warning("Initial cleanup maintenance failed", error=str(e))
        
    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        raise
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down media-to-text microservice")
    
    try:
        # Close job worker first
        logger.info("Closing job worker")
        await close_job_worker()
        logger.info("Job worker closed successfully")
        
        # Close Redis service
        logger.info("Closing Redis service")
        await close_redis_service()
        logger.info("Redis service closed successfully")
        
    except Exception as e:
        logger.warning("Error during shutdown", error=str(e))
    
    logger.info("Application shutdown complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure FastAPI application."""
    if settings is None:
        settings = Settings()

    app = FastAPI(
        title="Media-to-Text Microservice",
        description="Convert audio and video files to text using OpenAI Speech-to-Text API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )

    # Add logging middleware for request tracing (add this first)
    app.add_middleware(LoggingMiddleware)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(
        transcriptions.router,
        prefix="/transcriptions",
        tags=["Transcriptions"]
    )
    app.include_router(
        jobs.router,
        prefix="/jobs",
        tags=["Jobs"]
    )
    app.include_router(
        health.router,
        tags=["Health"]
    )

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with service information."""
        logger = get_logger("api")
        logger.info("Root endpoint accessed")
        
        return {
            "service": "media-to-text",
            "version": "0.1.0",
            "description": "Convert audio and video files to text",
            "docs": "/docs",
            "health": "/healthz",
            "architecture": {
                "containers": ["api", "redis", "ffmpeg"],
                "processing": "Docker Compose with separate containers",
                "features": [
                    "Automatic media type detection",
                    "Format conversion to 16-bit PCM WAV",
                    "Large file chunking with FFmpeg segments",
                    "Parallel OpenAI transcription with rate limiting",
                    "Real-time progress updates via Redis streams",
                    "Timestamp normalization and transcript aggregation",
                    "Enhanced job cleanup and resource management",
                    "Redis-based job management with crash recovery",
                    "Structured logging with Axiom integration",
                    "Request tracing and observability"
                ]
            },
            "endpoints": {
                "transcriptions": "/transcriptions",
                "jobs": "/jobs",
                "health": "/healthz",
                "metrics": "/metrics"
            },
            "logging": {
                "structured": settings.enable_structured_logging,
                "axiom_enabled": settings.enable_axiom,
                "log_level": settings.log_level
            }
        }

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = Settings()
    
    # Setup logging for CLI run
    setup_logging(
        log_level=settings.log_level,
        enable_axiom=settings.enable_axiom,
        axiom_token=settings.axiom_token,
        axiom_dataset=settings.axiom_dataset
    )
    
    logger = get_logger("cli")
    logger.info("Starting development server", 
                host=settings.api_host, 
                port=settings.api_port,
                debug=settings.debug)
    
    uvicorn.run(
        "media_to_text.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info"
    )