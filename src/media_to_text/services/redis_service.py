"""Redis service for job management and real-time updates."""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import redis.asyncio as redis
from redis.asyncio.client import Redis

from media_to_text.config import Settings
from media_to_text.models import JobState, JobStatus, JobMetadata
from media_to_text.logging import LoggerMixin, get_logger


class RedisService(LoggerMixin):
    """Redis service for job management, state tracking, and pub/sub."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.redis: Optional[Redis] = None
        self.redis_url = settings.redis_url
        self.ttl_seconds = settings.redis_ttl_days * 24 * 60 * 60
        self.jobs_prefix = settings.redis_jobs_prefix
        
    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self.redis = redis.from_url(self.redis_url, decode_responses=True)
            # Test connection
            await self.redis.ping()
            self.logger.info("Redis connection established", redis_url=self.redis_url)
        except Exception as e:
            self.logger.error("Failed to connect to Redis", 
                            redis_url=self.redis_url, 
                            error=str(e))
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis:
            try:
                await self.redis.aclose()
                self.logger.info("Redis connection closed")
            except Exception as e:
                self.logger.warning("Error closing Redis connection", error=str(e))
            finally:
                self.redis = None
    
    def _job_key(self, job_id: str) -> str:
        """Get Redis key for job data."""
        return f"{self.jobs_prefix}:{job_id}"
    
    def _stream_key(self, job_id: str) -> str:
        """Get Redis key for job updates stream."""
        return f"{self.jobs_prefix}:stream:{job_id}"
    
    async def create_job(self, job_id: str, file_path: str, **metadata) -> None:
        """Create a new job in Redis."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        # Create JobMetadata object
        job_metadata = JobMetadata(
            job_id=job_id,
            file_path=file_path,
            language=metadata.get("language", "en"),
            state=JobState.QUEUED,
            **{k: v for k, v in metadata.items() if k in ['original_filename', 'request_id', 'trace_id']}
        )
        
        # Convert to dict for Redis storage
        job_data = job_metadata.to_dict()
        
        job_key = self._job_key(job_id)
        
        try:
            # Store job data as hash
            await self.redis.hset(job_key, mapping=job_data)
            await self.redis.expire(job_key, self.ttl_seconds)
            
            # Add to stream for real-time updates
            await self._publish_update(job_id, JobState.QUEUED, **job_data)
            
            self.logger.info("Job created in Redis", 
                           job_id=job_id, 
                           file_path=file_path,
                           ttl_seconds=self.ttl_seconds)
        except Exception as e:
            self.logger.error("Failed to create job in Redis", 
                            job_id=job_id, 
                            error=str(e))
            raise
    
    async def update_job_state(self, job_id: str, state: JobState, **updates) -> None:
        """Update job state and metadata."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        job_key = self._job_key(job_id)
        
        try:
            # Check if job exists
            if not await self.redis.exists(job_key):
                self.logger.warning("Attempted to update non-existent job", job_id=job_id)
                return
            
            # Update job data
            update_data = {
                "state": state.value,
                "updated_at": datetime.utcnow().isoformat(),
                **updates
            }
            
            await self.redis.hset(job_key, mapping=update_data)
            
            # Publish update to stream
            await self._publish_update(job_id, state, **update_data)
            
            self.logger.debug("Job state updated", 
                            job_id=job_id, 
                            state=state.value,
                            updates=list(updates.keys()))
        except Exception as e:
            self.logger.error("Failed to update job state", 
                            job_id=job_id, 
                            state=state.value,
                            error=str(e))
            raise
    
    async def get_job(self, job_id: str) -> Optional[JobMetadata]:
        """Get job data by ID."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        job_key = self._job_key(job_id)
        
        try:
            job_data = await self.redis.hgetall(job_key)
            if not job_data:
                self.logger.debug("Job not found", job_id=job_id)
                return None
            
            # Convert to JobMetadata object
            try:
                job_metadata = JobMetadata.from_dict(job_data)
                self.logger.debug("Job retrieved", job_id=job_id, state=job_metadata.state.value)
                return job_metadata
            except Exception as e:
                self.logger.error("Failed to parse job metadata", job_id=job_id, error=str(e))
                return None
                
        except Exception as e:
            self.logger.error("Failed to get job", job_id=job_id, error=str(e))
            raise
    
    async def list_jobs(self, 
                       state_filter: Optional[JobState] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """List jobs with optional state filtering."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        try:
            # Get all job keys
            pattern = f"{self.jobs_prefix}:*"
            # Exclude stream keys
            all_keys = await self.redis.keys(pattern)
            job_keys = [key for key in all_keys if ":stream:" not in key]
            
            jobs = []
            for job_key in job_keys[:limit]:
                job_data = await self.redis.hgetall(job_key)
                if job_data:
                    # Convert numeric fields
                    if "progress" in job_data:
                        job_data["progress"] = float(job_data["progress"])
                    
                    # Apply state filter
                    if state_filter is None or job_data.get("state") == state_filter.value:
                        jobs.append(job_data)
            
            # Sort by created_at descending
            jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            self.logger.debug("Jobs listed", 
                            count=len(jobs), 
                            state_filter=state_filter.value if state_filter else None)
            return jobs
        except Exception as e:
            self.logger.error("Failed to list jobs", 
                            state_filter=state_filter.value if state_filter else None,
                            error=str(e))
            raise
    
    async def delete_job(self, job_id: str) -> bool:
        """Delete a job and its stream."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        job_key = self._job_key(job_id)
        stream_key = self._stream_key(job_id)
        
        try:
            # Delete job data and stream
            deleted_count = 0
            if await self.redis.exists(job_key):
                await self.redis.delete(job_key)
                deleted_count += 1
            
            if await self.redis.exists(stream_key):
                await self.redis.delete(stream_key)
                deleted_count += 1
            
            success = deleted_count > 0
            if success:
                self.logger.info("Job deleted from Redis", 
                               job_id=job_id, 
                               deleted_items=deleted_count)
            else:
                self.logger.warning("Attempted to delete non-existent job", job_id=job_id)
            
            return success
        except Exception as e:
            self.logger.error("Failed to delete job", job_id=job_id, error=str(e))
            raise
    
    async def _publish_update(self, job_id: str, state: JobState, **data) -> None:
        """Publish job update to Redis stream."""
        if not self.redis:
            return
        
        stream_key = self._stream_key(job_id)
        
        try:
            update_data = {
                "job_id": job_id,
                "state": state.value,
                "timestamp": datetime.utcnow().isoformat(),
                **{k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) 
                   for k, v in data.items()}
            }
            
            await self.redis.xadd(stream_key, update_data)
            await self.redis.expire(stream_key, self.ttl_seconds)
            
            self.logger.debug("Update published to stream", 
                            job_id=job_id, 
                            state=state.value)
        except Exception as e:
            self.logger.warning("Failed to publish update to stream", 
                              job_id=job_id,
                              error=str(e))
    
    async def get_job_updates(self, job_id: str, last_id: str = "0") -> List[Dict[str, Any]]:
        """Get job updates from stream."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        stream_key = self._stream_key(job_id)
        
        try:
            # Read from stream
            result = await self.redis.xread({stream_key: last_id}, count=100)
            
            updates = []
            if result:
                for stream, messages in result:
                    for message_id, fields in messages:
                        update = {"id": message_id, **fields}
                        # Parse JSON fields back
                        for key, value in update.items():
                            if key in ["id", "job_id", "state", "timestamp"]:
                                continue
                            try:
                                update[key] = json.loads(value)
                            except (json.JSONDecodeError, TypeError):
                                pass  # Keep as string
                        updates.append(update)
            
            self.logger.debug("Job updates retrieved", 
                            job_id=job_id, 
                            count=len(updates),
                            last_id=last_id)
            return updates
        except Exception as e:
            self.logger.error("Failed to get job updates", 
                            job_id=job_id, 
                            error=str(e))
            raise
    
    async def get_queued_jobs(self) -> List[Dict[str, Any]]:
        """Get all jobs in QUEUED state for processing."""
        return await self.list_jobs(state_filter=JobState.QUEUED)
    
    async def cleanup_expired_jobs(self) -> int:
        """Clean up expired jobs and return count of cleaned items."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        try:
            # Get all job keys
            pattern = f"{self.jobs_prefix}:*"
            all_keys = await self.redis.keys(pattern)
            
            cleaned_count = 0
            cutoff_time = datetime.utcnow() - timedelta(days=self.settings.redis_ttl_days)
            
            for key in all_keys:
                try:
                    # Check TTL
                    ttl = await self.redis.ttl(key)
                    if ttl == -1:  # No expiration set
                        # Check if it's old enough to clean
                        if ":stream:" not in key:  # Only check job data, not streams
                            job_data = await self.redis.hgetall(key)
                            if job_data and "created_at" in job_data:
                                created_at = datetime.fromisoformat(job_data["created_at"])
                                if created_at < cutoff_time:
                                    await self.redis.delete(key)
                                    cleaned_count += 1
                    elif ttl == -2:  # Key doesn't exist
                        continue
                except Exception as e:
                    self.logger.warning("Failed to check/clean key", key=key, error=str(e))
                    continue
            
            if cleaned_count > 0:
                self.logger.info("Cleaned up expired jobs", count=cleaned_count)
            
            return cleaned_count
        except Exception as e:
            self.logger.error("Failed to cleanup expired jobs", error=str(e))
            raise
    
    async def publish_job_update(self, job_id: str, update_data: Dict[str, Any]) -> None:
        """Publish job update event to Redis stream."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        stream_key = self._stream_key(job_id)
        
        try:
            # Prepare update data for stream
            stream_data = {
                "job_id": job_id,
                "timestamp": datetime.utcnow().isoformat(),
                **{k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) 
                   for k, v in update_data.items()}
            }
            
            await self.redis.xadd(stream_key, stream_data)
            await self.redis.expire(stream_key, self.ttl_seconds)
            
            self.logger.debug("Job update published to stream", 
                            job_id=job_id, 
                            event=update_data.get("event", "unknown"))
        except Exception as e:
            self.logger.warning("Failed to publish job update", 
                              job_id=job_id,
                              error=str(e))
    
    async def update_job_progress(self, job_id: str, chunks_done: int, chunks_total: int) -> None:
        """Update job progress information."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        job_key = self._job_key(job_id)
        
        try:
            # Check if job exists
            if not await self.redis.exists(job_key):
                self.logger.warning("Attempted to update progress for non-existent job", job_id=job_id)
                return
            
            # Calculate progress percentage
            progress = (chunks_done / chunks_total * 100) if chunks_total > 0 else 0
            
            # Update job progress
            update_data = {
                "chunks_done": chunks_done,
                "chunks_total": chunks_total,
                "progress": progress,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            await self.redis.hset(job_key, mapping=update_data)
            
            # Publish progress update to stream
            await self.publish_job_update(job_id, {
                "event": "progress_update",
                "chunks_done": chunks_done,
                "chunks_total": chunks_total,
                "progress": progress
            })
            
            self.logger.debug("Job progress updated", 
                            job_id=job_id, 
                            chunks_done=chunks_done,
                            chunks_total=chunks_total,
                            progress=f"{progress:.1f}%")
        except Exception as e:
            self.logger.error("Failed to update job progress", 
                            job_id=job_id, 
                            error=str(e))
            raise


# Global instance
_redis_service: Optional[RedisService] = None


async def init_redis_service(settings: Settings) -> RedisService:
    """Initialize Redis service."""
    global _redis_service
    logger = get_logger("redis_init")
    
    try:
        _redis_service = RedisService(settings)
        await _redis_service.connect()
        logger.info("Redis service initialized successfully")
        return _redis_service
    except Exception as e:
        logger.error("Failed to initialize Redis service", error=str(e))
        raise


async def close_redis_service() -> None:
    """Close Redis service."""
    global _redis_service
    logger = get_logger("redis_close")
    
    if _redis_service:
        try:
            await _redis_service.disconnect()
            _redis_service = None
            logger.info("Redis service closed successfully")
        except Exception as e:
            logger.warning("Error closing Redis service", error=str(e))


def get_redis_service() -> RedisService:
    """Get the global Redis service instance."""
    if _redis_service is None:
        raise RuntimeError("Redis service not initialized")
    return _redis_service