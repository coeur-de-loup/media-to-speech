"""Cleanup service for managing job resources and temporary files."""

import asyncio
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from media_to_text.config import Settings
from media_to_text.models import JobState
from media_to_text.services.redis_service import RedisService
from media_to_text.logging import LoggerMixin


class CleanupService(LoggerMixin):
    """Service for cleaning up job resources and temporary files."""
    
    def __init__(self, settings: Settings, redis_service: RedisService):
        self.settings = settings
        self.redis_service = redis_service
        self.cleanup_prefix = f"{settings.redis_jobs_prefix}:cleanup"
        
    async def schedule_job_cleanup(self, job_id: str, delay_seconds: int = 300) -> None:
        """
        Schedule cleanup for a job after a delay.
        
        Args:
            job_id: Job identifier
            delay_seconds: Delay before cleanup (default 5 minutes)
        """
        cleanup_time = int(time.time()) + delay_seconds
        cleanup_key = f"{self.cleanup_prefix}:{job_id}"
        
        cleanup_metadata = {
            "job_id": job_id,
            "scheduled_at": str(datetime.utcnow().isoformat()),
            "cleanup_time": str(cleanup_time),
            "status": "scheduled"
        }
        
        # Store cleanup metadata
        await self.redis_service.redis.hset(cleanup_key, mapping=cleanup_metadata)
        await self.redis_service.redis.expire(cleanup_key, delay_seconds + 3600)  # Extra hour buffer
        
        self.logger.info(f"🗑️  Scheduled cleanup for job {job_id} in {delay_seconds} seconds")
    
    async def trigger_immediate_cleanup(self, job_id: str, job_state: JobState) -> bool:
        """
        Trigger immediate cleanup for a job.
        
        Args:
            job_id: Job identifier
            job_state: Final job state
            
        Returns:
            True if cleanup was successful
        """
        self.logger.info(f"🧹 Starting immediate cleanup for job {job_id} (state: {job_state})")
        
        try:
            # Get job directory
            job_dir = os.path.join(self.settings.temp_dir, f"job_{job_id}")
            
            if not os.path.exists(job_dir):
                self.logger.info(f"✅ Job directory {job_dir} does not exist, cleanup not needed")
                return True
            
            # Perform cleanup based on job state
            if job_state == JobState.COMPLETED:
                success = await self._cleanup_completed_job(job_id, job_dir)
            elif job_state in [JobState.FAILED, JobState.CANCELLED]:
                success = await self._cleanup_failed_job(job_id, job_dir)
            else:
                self.logger.warning(f"⚠️  Unexpected job state for cleanup: {job_state}")
                success = await self._cleanup_all_files(job_id, job_dir)
            
            # Update cleanup metadata
            if success:
                await self._update_cleanup_status(job_id, "completed")
            else:
                await self._update_cleanup_status(job_id, "failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup failed for job {job_id}: {e}")
            await self._update_cleanup_status(job_id, "failed", str(e))
            return False
    
    async def _cleanup_completed_job(self, job_id: str, job_dir: str) -> bool:
        """Cleanup completed job, preserving transcript file temporarily."""
        try:
            # For completed jobs, preserve transcript.json for a while
            transcript_file = os.path.join(job_dir, "transcript.json")
            preserved_files = []
            
            if os.path.exists(transcript_file):
                preserved_files.append("transcript.json")
                self.logger.info(f"📄 Preserving transcript.json for job {job_id}")
            
            # Remove all other files and directories
            removed_count = 0
            for item in os.listdir(job_dir):
                if item not in preserved_files:
                    item_path = os.path.join(job_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        removed_count += 1
                        self.logger.info(f"🗂️  Removed directory: {item}")
                    else:
                        os.remove(item_path)
                        removed_count += 1
                        self.logger.info(f"📄 Removed file: {item}")
            
            # If no files were preserved, remove the entire directory
            if not preserved_files:
                shutil.rmtree(job_dir)
                self.logger.info(f"🗂️  Removed entire job directory: {job_dir}")
            
            self.logger.info(f"✅ Completed job cleanup: removed {removed_count} items")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to cleanup completed job {job_id}: {e}")
            return False
    
    async def _cleanup_failed_job(self, job_id: str, job_dir: str) -> bool:
        """Cleanup failed job, removing all files immediately."""
        try:
            # For failed jobs, remove everything immediately
            shutil.rmtree(job_dir)
            self.logger.info(f"🗂️  Removed entire failed job directory: {job_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to cleanup failed job {job_id}: {e}")
            return False
    
    async def _cleanup_all_files(self, job_id: str, job_dir: str) -> bool:
        """Remove all files for a job (fallback cleanup)."""
        try:
            shutil.rmtree(job_dir)
            self.logger.info(f"🗂️  Removed entire job directory: {job_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to cleanup job directory {job_id}: {e}")
            return False
    
    async def _update_cleanup_status(self, job_id: str, status: str, error_message: Optional[str] = None) -> None:
        """Update cleanup status in Redis."""
        cleanup_key = f"{self.cleanup_prefix}:{job_id}"
        
        updates = {
            "status": status,
            "completed_at": str(datetime.utcnow().isoformat())
        }
        
        if error_message:
            updates["error_message"] = error_message
        
        await self.redis_service.redis.hset(cleanup_key, mapping=updates)
    
    async def cleanup_orphaned_jobs(self) -> int:
        """
        Find and cleanup orphaned job directories (crash recovery).
        
        Returns:
            Number of orphaned jobs cleaned up
        """
        self.logger.info("🔍 Scanning for orphaned job directories...")
        
        orphaned_count = 0
        temp_dir = Path(self.settings.temp_dir)
        
        if not temp_dir.exists():
            return 0
        
        # Get all job directories
        job_dirs = [d for d in temp_dir.iterdir() if d.is_dir() and d.name.startswith("job_")]
        
        for job_dir in job_dirs:
            try:
                # Extract job ID from directory name
                job_id = job_dir.name.replace("job_", "")
                
                # Check if job still exists in Redis
                job = await self.redis_service.get_job(job_id)
                
                if job is None:
                    # Job not in Redis, it's orphaned
                    self.logger.info(f"🧹 Found orphaned job directory: {job_dir}")
                    shutil.rmtree(job_dir)
                    orphaned_count += 1
                elif job.state in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED]:
                    # Job is completed but directory still exists, check age
                    dir_age = time.time() - job_dir.stat().st_mtime
                    if dir_age > 3600:  # Older than 1 hour
                        self.logger.info(f"🧹 Cleaning up old completed job directory: {job_dir}")
                        await self.trigger_immediate_cleanup(job_id, job.state)
                        orphaned_count += 1
                        
            except Exception as e:
                self.logger.warning(f"⚠️  Error processing directory {job_dir}: {e}")
        
        if orphaned_count > 0:
            self.logger.info(f"✅ Cleaned up {orphaned_count} orphaned job directories")
        else:
            self.logger.info("✅ No orphaned job directories found")
        
        return orphaned_count
    
    async def cleanup_expired_redis_data(self) -> int:
        """
        Clean up expired Redis data that wasn't automatically removed.
        
        Returns:
            Number of expired keys cleaned up
        """
        self.logger.info("🔍 Scanning for expired Redis data...")
        
        cleaned_count = 0
        
        try:
            # Get all job metadata keys
            pattern = f"{self.redis_service.jobs_prefix}:*:meta"
            
            async for key in self.redis_service.redis.scan_iter(match=pattern):
                try:
                    # Check TTL
                    ttl = await self.redis_service.redis.ttl(key)
                    
                    if ttl == -1:  # No expiration set
                        # Get job data to check age
                        job_data = await self.redis_service.redis.hgetall(key)
                        if job_data and "created_at" in job_data:
                            created_at = datetime.fromisoformat(job_data["created_at"])
                            age = datetime.utcnow() - created_at
                            
                            if age > timedelta(days=self.redis_service.ttl_seconds // (24 * 60 * 60)):
                                # Older than TTL period, set expiration
                                await self.redis_service.redis.expire(key, 3600)  # Expire in 1 hour
                                cleaned_count += 1
                                
                except Exception as e:
                    self.logger.warning(f"⚠️  Error processing Redis key {key}: {e}")
            
            if cleaned_count > 0:
                self.logger.info(f"✅ Set expiration for {cleaned_count} Redis keys")
            else:
                self.logger.info("✅ No expired Redis data found")
            
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up Redis data: {e}")
        
        return cleaned_count
    
    async def get_cleanup_status(self, job_id: str) -> Optional[Dict]:
        """Get cleanup status for a job."""
        cleanup_key = f"{self.cleanup_prefix}:{job_id}"
        cleanup_data = await self.redis_service.redis.hgetall(cleanup_key)
        
        if not cleanup_data:
            return None
        
        return cleanup_data
    
    async def run_maintenance_cycle(self) -> Dict[str, int]:
        """
        Run a complete maintenance cycle (cleanup orphaned files and Redis data).
        
        Returns:
            Dictionary with cleanup statistics
        """
        self.logger.info("🔧 Starting cleanup maintenance cycle...")
        
        stats = {
            "orphaned_directories": await self.cleanup_orphaned_jobs(),
            "expired_redis_keys": await self.cleanup_expired_redis_data()
        }
        
        self.logger.info(f"✅ Maintenance cycle complete: {stats}")
        return stats


# Global cleanup service instance
cleanup_service: Optional[CleanupService] = None


def get_cleanup_service() -> CleanupService:
    """Get cleanup service instance."""
    global cleanup_service
    if cleanup_service is None:
        raise RuntimeError("Cleanup service not initialized")
    return cleanup_service


def init_cleanup_service(settings: Settings, redis_service: RedisService) -> CleanupService:
    """Initialize cleanup service."""
    global cleanup_service
    cleanup_service = CleanupService(settings, redis_service)
    return cleanup_service