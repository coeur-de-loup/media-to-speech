"""Health, readiness, and metrics endpoints for media-to-text microservice."""

import asyncio
import json
import os
import psutil
import time
from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel

from media_to_text.config import Settings
from media_to_text.services.redis_service import get_redis_service, RedisService


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    service: str
    timestamp: str
    uptime_seconds: float


class ReadinessResponse(BaseModel):
    """Readiness check response model."""
    status: str
    checks: Dict[str, Dict[str, str]]
    ready: bool


class MetricsResponse(BaseModel):
    """Metrics response model."""
    system: Dict[str, float]
    application: Dict[str, int]
    redis: Dict[str, int]
    jobs: Dict[str, int]


router = APIRouter()

# Global startup time for uptime calculation
startup_time = time.time()


@router.get("/healthz", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Basic health check endpoint.
    
    Returns basic service information and uptime.
    This endpoint should always return 200 OK if the service is running.
    """
    uptime = time.time() - startup_time
    
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        service="media-to-text",
        timestamp=datetime.utcnow().isoformat() + "Z",
        uptime_seconds=round(uptime, 2)
    )


@router.get("/readyz", response_model=ReadinessResponse)
async def readiness_check(
    redis_service: RedisService = Depends(get_redis_service)
) -> ReadinessResponse:
    """
    Readiness check endpoint with dependency validation.
    
    Checks if the service is ready to handle requests by validating:
    - Redis connectivity
    - FFmpeg container availability
    - Disk space availability
    - Memory availability
    
    Returns 200 if ready, 503 if not ready.
    """
    checks = {}
    all_ready = True
    
    # Check Redis connectivity
    try:
        await redis_service.redis.ping()
        checks["redis"] = {
            "status": "ready",
            "message": "Redis connection successful"
        }
    except Exception as e:
        checks["redis"] = {
            "status": "not_ready",
            "message": f"Redis connection failed: {str(e)}"
        }
        all_ready = False
    
    # Check FFmpeg container availability
    try:
        # Try to execute a simple FFmpeg command via Docker
        process = await asyncio.create_subprocess_exec(
            "docker", "exec", "media-to-text-ffmpeg", "ffmpeg", "-version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5.0)
        
        if process.returncode == 0:
            checks["ffmpeg"] = {
                "status": "ready",
                "message": "FFmpeg container accessible"
            }
        else:
            checks["ffmpeg"] = {
                "status": "not_ready",
                "message": f"FFmpeg command failed: {stderr.decode()}"
            }
            all_ready = False
            
    except asyncio.TimeoutError:
        checks["ffmpeg"] = {
            "status": "not_ready",
            "message": "FFmpeg container timeout"
        }
        all_ready = False
    except Exception as e:
        checks["ffmpeg"] = {
            "status": "not_ready",
            "message": f"FFmpeg container check failed: {str(e)}"
        }
        all_ready = False
    
    # Check disk space
    try:
        from media_to_text.config import Settings
        settings = Settings()
        
        # Check temp directory disk space
        statvfs = os.statvfs(settings.temp_dir)
        free_bytes = statvfs.f_frsize * statvfs.f_bavail
        total_bytes = statvfs.f_frsize * statvfs.f_blocks
        free_percent = (free_bytes / total_bytes) * 100
        
        if free_percent > 10:  # At least 10% free space required
            checks["disk_space"] = {
                "status": "ready",
                "message": f"Sufficient disk space: {free_percent:.1f}% free"
            }
        else:
            checks["disk_space"] = {
                "status": "not_ready",
                "message": f"Low disk space: {free_percent:.1f}% free"
            }
            all_ready = False
            
    except Exception as e:
        checks["disk_space"] = {
            "status": "not_ready",
            "message": f"Disk space check failed: {str(e)}"
        }
        all_ready = False
    
    # Check memory availability
    try:
        memory = psutil.virtual_memory()
        if memory.percent < 90:  # Less than 90% memory usage
            checks["memory"] = {
                "status": "ready",
                "message": f"Memory usage: {memory.percent:.1f}%"
            }
        else:
            checks["memory"] = {
                "status": "not_ready",
                "message": f"High memory usage: {memory.percent:.1f}%"
            }
            all_ready = False
            
    except Exception as e:
        checks["memory"] = {
            "status": "not_ready",
            "message": f"Memory check failed: {str(e)}"
        }
        all_ready = False
    
    # Determine overall status
    if all_ready:
        response = ReadinessResponse(
            status="ready",
            checks=checks,
            ready=True
        )
        return response
    else:
        response = ReadinessResponse(
            status="not_ready",
            checks=checks,
            ready=False
        )
        # Return 503 Service Unavailable if not ready
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.dict()
        )


@router.get("/metrics", response_model=MetricsResponse)
async def metrics_endpoint(
    redis_service: RedisService = Depends(get_redis_service)
) -> MetricsResponse:
    """
    Metrics endpoint for monitoring and observability.
    
    Returns system metrics, application metrics, and job statistics.
    """
    # System metrics
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_metrics = {
            "cpu_usage_percent": round(cpu_percent, 2),
            "memory_usage_percent": round(memory.percent, 2),
            "memory_used_bytes": memory.used,
            "memory_total_bytes": memory.total,
            "disk_usage_percent": round((disk.used / disk.total) * 100, 2),
            "disk_free_bytes": disk.free,
            "disk_total_bytes": disk.total,
            "uptime_seconds": round(time.time() - startup_time, 2)
        }
    except Exception:
        system_metrics = {
            "cpu_usage_percent": 0.0,
            "memory_usage_percent": 0.0,
            "memory_used_bytes": 0,
            "memory_total_bytes": 0,
            "disk_usage_percent": 0.0,
            "disk_free_bytes": 0,
            "disk_total_bytes": 0,
            "uptime_seconds": round(time.time() - startup_time, 2)
        }
    
    # Application metrics (example - would be collected from actual app state)
    application_metrics = {
        "active_connections": 0,  # Would track active HTTP connections
        "total_requests": 0,      # Would track total requests handled
        "error_rate": 0           # Would track error percentage
    }
    
    # Redis metrics
    try:
        redis_info = await redis_service.redis.info()
        redis_metrics = {
            "connected_clients": int(redis_info.get("connected_clients", 0)),
            "used_memory": int(redis_info.get("used_memory", 0)),
            "keyspace_hits": int(redis_info.get("keyspace_hits", 0)),
            "keyspace_misses": int(redis_info.get("keyspace_misses", 0)),
            "total_commands_processed": int(redis_info.get("total_commands_processed", 0))
        }
    except Exception:
        redis_metrics = {
            "connected_clients": 0,
            "used_memory": 0,
            "keyspace_hits": 0,
            "keyspace_misses": 0,
            "total_commands_processed": 0
        }
    
    # Job metrics
    try:
        from media_to_text.models import JobState
        
        # Count jobs by state
        all_jobs = await redis_service.list_jobs()
        
        job_metrics = {
            "total_jobs": len(all_jobs),
            "queued_jobs": len([j for j in all_jobs if j.state == JobState.QUEUED]),
            "processing_jobs": len([j for j in all_jobs if j.state == JobState.PROCESSING]),
            "completed_jobs": len([j for j in all_jobs if j.state == JobState.COMPLETED]),
            "failed_jobs": len([j for j in all_jobs if j.state == JobState.FAILED]),
            "cancelled_jobs": len([j for j in all_jobs if j.state == JobState.CANCELLED])
        }
    except Exception:
        job_metrics = {
            "total_jobs": 0,
            "queued_jobs": 0,
            "processing_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "cancelled_jobs": 0
        }
    
    return MetricsResponse(
        system=system_metrics,
        application=application_metrics,
        redis=redis_metrics,
        jobs=job_metrics
    )


@router.get("/metrics/prometheus")
async def prometheus_metrics(
    redis_service: RedisService = Depends(get_redis_service)
) -> Response:
    """
    Prometheus-compatible metrics endpoint.
    
    Returns metrics in Prometheus exposition format.
    """
    try:
        # Get metrics data
        metrics_data = await metrics_endpoint(redis_service)
        
        # Convert to Prometheus format
        prometheus_lines = []
        
        # System metrics
        prometheus_lines.append("# HELP media_to_text_cpu_usage_percent CPU usage percentage")
        prometheus_lines.append("# TYPE media_to_text_cpu_usage_percent gauge")
        prometheus_lines.append(f"media_to_text_cpu_usage_percent {metrics_data.system['cpu_usage_percent']}")
        
        prometheus_lines.append("# HELP media_to_text_memory_usage_percent Memory usage percentage")
        prometheus_lines.append("# TYPE media_to_text_memory_usage_percent gauge")
        prometheus_lines.append(f"media_to_text_memory_usage_percent {metrics_data.system['memory_usage_percent']}")
        
        prometheus_lines.append("# HELP media_to_text_uptime_seconds Service uptime in seconds")
        prometheus_lines.append("# TYPE media_to_text_uptime_seconds counter")
        prometheus_lines.append(f"media_to_text_uptime_seconds {metrics_data.system['uptime_seconds']}")
        
        # Job metrics
        for state, count in metrics_data.jobs.items():
            prometheus_lines.append(f"# HELP media_to_text_{state} Number of {state.replace('_', ' ')}")
            prometheus_lines.append(f"# TYPE media_to_text_{state} gauge")
            prometheus_lines.append(f"media_to_text_{state} {count}")
        
        # Redis metrics
        prometheus_lines.append("# HELP media_to_text_redis_connected_clients Number of Redis connected clients")
        prometheus_lines.append("# TYPE media_to_text_redis_connected_clients gauge")
        prometheus_lines.append(f"media_to_text_redis_connected_clients {metrics_data.redis['connected_clients']}")
        
        prometheus_output = "\n".join(prometheus_lines) + "\n"
        
        return Response(
            content=prometheus_output,
            media_type="text/plain; charset=utf-8"
        )
    
    except Exception as e:
        # Return basic error metric if metrics collection fails
        error_output = f"""# HELP media_to_text_metrics_error Metrics collection error
# TYPE media_to_text_metrics_error gauge
media_to_text_metrics_error 1
# Error: {str(e)}
"""
        return Response(
            content=error_output,
            media_type="text/plain; charset=utf-8"
        )


@router.get("/status")
async def service_status(
    redis_service: RedisService = Depends(get_redis_service)
) -> Dict:
    """
    Comprehensive service status endpoint.
    
    Combines health, readiness, and metrics into a single response.
    """
    try:
        health = await health_check()
        try:
            readiness = await readiness_check(redis_service)
            ready = True
        except HTTPException:
            # Get the readiness data even if not ready
            readiness_data = await readiness_check.__wrapped__(redis_service)
            readiness = readiness_data
            ready = False
        
        metrics = await metrics_endpoint(redis_service)
        
        return {
            "health": health.dict(),
            "readiness": {
                **readiness.dict(),
                "ready": ready
            },
            "metrics": metrics.dict(),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        return {
            "error": f"Failed to collect status: {str(e)}",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }