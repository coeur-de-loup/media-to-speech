"""Redis service for job management and pub/sub."""

import json
import uuid
from typing import Dict, List, Optional

import redis.asyncio as redis
from redis.asyncio import Redis

from media_to_text.config import Settings
from media_to_text.models import JobMetadata, JobState


class RedisService:
    """Service for Redis operations."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.redis: Optional[Redis] = None
        self.jobs_prefix = settings.redis_jobs_prefix
        self.ttl_seconds = settings.redis_ttl_days * 24 * 60 * 60
    
    async def connect(self) -> None:
        """Connect to Redis."""
        self.redis = redis.from_url(
            self.settings.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        # Test connection
        await self.redis.ping()
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
    
    def _get_meta_key(self, job_id: str) -> str:
        """Get Redis key for job metadata."""
        return f"{self.jobs_prefix}:{job_id}:meta"
    
    def _get_stream_key(self, job_id: str) -> str:
        """Get Redis key for job stream."""
        return f"{self.jobs_prefix}:{job_id}:stream"
    
    async def create_job(self, file_path: str, language: str) -> JobMetadata:
        """Create a new job in Redis."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        job_id = str(uuid.uuid4())
        job_metadata = JobMetadata(
            job_id=job_id,
            file_path=file_path,
            language=language,
            state=JobState.QUEUED
        )
        
        # Store metadata
        meta_key = self._get_meta_key(job_id)
        await self.redis.hset(meta_key, mapping=job_metadata.to_dict())
        await self.redis.expire(meta_key, self.ttl_seconds)
        
        # Publish initial state to stream
        await self.publish_job_update(job_id, {
            "state": JobState.QUEUED.value,
            "message": "Job created and queued for processing"
        })
        
        return job_metadata
    
    async def get_job(self, job_id: str) -> Optional[JobMetadata]:
        """Get job metadata from Redis."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        meta_key = self._get_meta_key(job_id)
        job_data = await self.redis.hgetall(meta_key)
        
        if not job_data:
            return None
        
        return JobMetadata.from_dict(job_data)
    
    async def update_job_state(self, job_id: str, state: JobState, error_message: Optional[str] = None) -> None:
        """Update job state in Redis."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        meta_key = self._get_meta_key(job_id)
        updates = {"state": state.value}
        
        if error_message:
            updates["error_message"] = error_message
        
        await self.redis.hset(meta_key, mapping=updates)
        
        # Publish state change to stream
        await self.publish_job_update(job_id, {
            "state": state.value,
            "error_message": error_message
        })
    
    async def update_job_progress(self, job_id: str, chunks_done: int, chunks_total: int) -> None:
        """Update job progress in Redis."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        meta_key = self._get_meta_key(job_id)
        await self.redis.hset(meta_key, mapping={
            "chunks_done": str(chunks_done),
            "chunks_total": str(chunks_total)
        })
        
        # Calculate and publish progress
        progress = chunks_done / chunks_total if chunks_total > 0 else 0.0
        await self.publish_job_update(job_id, {
            "progress": progress,
            "chunks_done": chunks_done,
            "chunks_total": chunks_total
        })
    
    async def publish_job_update(self, job_id: str, data: Dict) -> None:
        """Publish job update to Redis stream."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        stream_key = self._get_stream_key(job_id)
        
        # Add timestamp and job_id to data
        stream_data = {
            "job_id": job_id,
            "timestamp": str(redis.time.time()),
            **data
        }
        
        await self.redis.xadd(stream_key, stream_data)
        await self.redis.expire(stream_key, self.ttl_seconds)
    
    async def get_job_updates(self, job_id: str, start_id: str = "0") -> List[Dict]:
        """Get job updates from Redis stream."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        stream_key = self._get_stream_key(job_id)
        
        try:
            messages = await self.redis.xread({stream_key: start_id}, count=100)
            updates = []
            
            for stream, msgs in messages:
                for msg_id, fields in msgs:
                    updates.append({
                        "id": msg_id,
                        **fields
                    })
            
            return updates
        except redis.RedisError:
            return []
    
    async def delete_job(self, job_id: str) -> bool:
        """Delete job metadata and stream from Redis."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        meta_key = self._get_meta_key(job_id)
        stream_key = self._get_stream_key(job_id)
        
        # Delete both keys
        deleted = await self.redis.delete(meta_key, stream_key)
        return deleted > 0
    
    async def list_jobs(self, state: Optional[JobState] = None) -> List[JobMetadata]:
        """List all jobs, optionally filtered by state."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        # Scan for all job metadata keys
        pattern = f"{self.jobs_prefix}:*:meta"
        jobs = []
        
        async for key in self.redis.scan_iter(match=pattern):
            job_data = await self.redis.hgetall(key)
            if job_data:
                job_metadata = JobMetadata.from_dict(job_data)
                if state is None or job_metadata.state == state:
                    jobs.append(job_metadata)
        
        return sorted(jobs, key=lambda x: x.created_at, reverse=True)


# Global Redis service instance
redis_service: Optional[RedisService] = None


async def get_redis_service() -> RedisService:
    """Get Redis service instance."""
    global redis_service
    if redis_service is None:
        raise RuntimeError("Redis service not initialized")
    return redis_service


async def init_redis_service(settings: Settings) -> RedisService:
    """Initialize Redis service."""
    global redis_service
    redis_service = RedisService(settings)
    await redis_service.connect()
    return redis_service


async def close_redis_service() -> None:
    """Close Redis service."""
    global redis_service
    if redis_service:
        await redis_service.disconnect()
        redis_service = None