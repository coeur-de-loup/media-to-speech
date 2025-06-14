"""Main FastAPI application for media-to-text microservice."""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from media_to_text.config import Settings
from media_to_text.routers import transcriptions, jobs, health
from media_to_text.services.redis_service import init_redis_service, close_redis_service, get_redis_service
from media_to_text.services.job_worker import init_job_worker, close_job_worker
from media_to_text.services.cleanup_service import init_cleanup_service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    print("Starting media-to-text microservice...")
    
    # Initialize services
    settings = Settings()
    
    try:
        # Initialize Redis service
        redis_service = await init_redis_service(settings)
        print("✅ Redis service initialized successfully")
        
        # Initialize cleanup service
        cleanup_service = await init_cleanup_service(settings, redis_service)
        print("✅ Cleanup service initialized successfully")
        
        # Initialize job worker
        job_worker = await init_job_worker(settings, redis_service)
        
        # Connect cleanup service to job worker
        await job_worker.set_cleanup_service(cleanup_service)
        print("✅ Job worker initialized with cleanup service")
        
        # Run initial maintenance cycle to clean up any orphaned files
        try:
            await cleanup_service.run_maintenance_cycle()
        except Exception as e:
            print(f"⚠️  Warning: Initial cleanup maintenance failed: {e}")
        
    except Exception as e:
        print(f"❌ Failed to initialize services: {e}")
        raise
    
    print("🚀 Application startup complete")
    
    yield
    
    # Shutdown
    print("Shutting down media-to-text microservice...")
    
    try:
        # Close job worker first
        await close_job_worker()
        print("✅ Job worker closed successfully")
        
        # Close Redis service
        await close_redis_service()
        print("✅ Redis service closed successfully")
        
    except Exception as e:
        print(f"⚠️  Error during shutdown: {e}")
    
    print("🛑 Application shutdown complete")


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
                    "Redis-based job management with crash recovery"
                ]
            },
            "endpoints": {
                "transcriptions": "/transcriptions",
                "jobs": "/jobs",
                "health": "/healthz"
            }
        }

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = Settings()
    uvicorn.run(
        "media_to_text.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info"
    )