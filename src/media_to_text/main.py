"""Main FastAPI application for media-to-text microservice."""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from media_to_text.config import Settings
from media_to_text.routers import transcriptions, jobs, health


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    print("Starting media-to-text microservice...")
    yield
    # Shutdown
    print("Shutting down media-to-text microservice...")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure FastAPI application."""
    if settings is None:
        settings = Settings()
    
    app = FastAPI(
        title="Media-to-Text Microservice",
        description="Convert audio/video files to text using OpenAI Speech-to-Text API",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(transcriptions.router, prefix="/transcriptions", tags=["transcriptions"])
    app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
    app.include_router(health.router, tags=["health"])
    
    return app


# Create app instance
settings = Settings()
app = create_app(settings)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )