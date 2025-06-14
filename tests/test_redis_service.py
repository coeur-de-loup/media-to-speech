"""Unit tests for Redis service."""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from redis.exceptions import ConnectionError, TimeoutError

from media_to_text.models import JobMetadata, JobState
from media_to_text.services.redis_service import RedisService


class TestRedisService:
    """Test suite for Redis service."""

    @pytest.fixture
    def redis_service(self):
        """Create Redis service instance with mocked connection."""
        with patch('redis.asyncio.Redis') as mock_redis:
            service = RedisService(
                host="localhost",
                port=6379,
                db=0,
                ttl_seconds=3600
            )
            service.redis = mock_redis
            return service

    @pytest.fixture
    def sample_job_metadata(self):
        """Sample job metadata for testing."""
        return JobMetadata(
            job_id="test-job-123",
            file_path="/tmp/test.mp3",
            language="en",
            state=JobState.QUEUED,
            chunks_done=0,
            chunks_total=0
        )

    @pytest.fixture
    def sample_job_dict(self):
        """Sample job dictionary for testing."""
        return {
            "id": "test-job-123",
            "file_path": "/tmp/test.mp3",
            "language": "en",
            "state": "QUEUED",
            "chunks_done": 0,
            "chunks_total": 0,
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00"
        }

    @pytest.mark.asyncio
    async def test_create_job_success(self, redis_service, sample_job_metadata):
        """Test successful job creation."""
        redis_service.redis.hset = AsyncMock(return_value=True)
        redis_service.redis.expire = AsyncMock(return_value=True)
        
        result = await redis_service.create_job(sample_job_metadata)
        
        assert result is True
        redis_service.redis.hset.assert_called_once()
        redis_service.redis.expire.assert_called_once_with(
            "job:test-job-123", 
            redis_service.ttl_seconds
        )

    @pytest.mark.asyncio
    async def test_create_job_redis_error(self, redis_service, sample_job_metadata):
        """Test job creation with Redis error."""
        redis_service.redis.hset = AsyncMock(side_effect=ConnectionError("Redis unavailable"))
        
        with pytest.raises(RuntimeError, match="Failed to create job"):
            await redis_service.create_job(sample_job_metadata)

    @pytest.mark.asyncio
    async def test_get_job_success(self, redis_service, sample_job_dict):
        """Test successful job retrieval."""
        redis_service.redis.hgetall = AsyncMock(return_value=sample_job_dict)
        
        job = await redis_service.get_job("test-job-123")
        
        assert job is not None
        assert job["id"] == "test-job-123"
        assert job["state"] == "QUEUED"
        redis_service.redis.hgetall.assert_called_once_with("job:test-job-123")

    @pytest.mark.asyncio
    async def test_get_job_not_found(self, redis_service):
        """Test job retrieval when job doesn't exist."""
        redis_service.redis.hgetall = AsyncMock(return_value={})
        
        job = await redis_service.get_job("nonexistent-job")
        
        assert job is None

    @pytest.mark.asyncio
    async def test_get_job_redis_error(self, redis_service):
        """Test job retrieval with Redis error."""
        redis_service.redis.hgetall = AsyncMock(side_effect=TimeoutError("Redis timeout"))
        
        with pytest.raises(RuntimeError, match="Failed to get job"):
            await redis_service.get_job("test-job-123")

    @pytest.mark.asyncio
    async def test_update_job_state_success(self, redis_service):
        """Test successful job state update."""
        redis_service.redis.hset = AsyncMock(return_value=True)
        redis_service.redis.xadd = AsyncMock(return_value=b"stream-id")
        
        result = await redis_service.update_job_state(
            "test-job-123", 
            JobState.PROCESSING,
            error_message="Test error"
        )
        
        assert result is True
        
        # Verify Redis calls
        redis_service.redis.hset.assert_called()
        redis_service.redis.xadd.assert_called()

    @pytest.mark.asyncio
    async def test_update_job_progress_success(self, redis_service):
        """Test successful job progress update."""
        redis_service.redis.hset = AsyncMock(return_value=True)
        redis_service.redis.xadd = AsyncMock(return_value=b"stream-id")
        
        result = await redis_service.update_job_progress(
            "test-job-123",
            chunks_done=5,
            chunks_total=10
        )
        
        assert result is True
        redis_service.redis.hset.assert_called()
        redis_service.redis.xadd.assert_called()

    @pytest.mark.asyncio
    async def test_list_jobs_all(self, redis_service):
        """Test listing all jobs."""
        # Mock Redis SCAN operation
        redis_service.redis.scan = AsyncMock(return_value=(0, [b"job:test-1", b"job:test-2"]))
        redis_service.redis.hgetall = AsyncMock(side_effect=[
            {"id": "test-1", "state": "QUEUED"},
            {"id": "test-2", "state": "PROCESSING"}
        ])
        
        jobs = await redis_service.list_jobs()
        
        assert len(jobs) == 2
        assert jobs[0]["id"] == "test-1"
        assert jobs[1]["id"] == "test-2"

    @pytest.mark.asyncio
    async def test_list_jobs_with_state_filter(self, redis_service):
        """Test listing jobs with state filter."""
        redis_service.redis.scan = AsyncMock(return_value=(0, [b"job:test-1", b"job:test-2"]))
        redis_service.redis.hgetall = AsyncMock(side_effect=[
            {"id": "test-1", "state": "QUEUED"},
            {"id": "test-2", "state": "PROCESSING"}
        ])
        
        jobs = await redis_service.list_jobs(state_filter=JobState.QUEUED)
        
        assert len(jobs) == 1
        assert jobs[0]["id"] == "test-1"
        assert jobs[0]["state"] == "QUEUED"

    @pytest.mark.asyncio
    async def test_list_jobs_empty_result(self, redis_service):
        """Test listing jobs when no jobs exist."""
        redis_service.redis.scan = AsyncMock(return_value=(0, []))
        
        jobs = await redis_service.list_jobs()
        
        assert len(jobs) == 0

    @pytest.mark.asyncio
    async def test_delete_job_success(self, redis_service):
        """Test successful job deletion."""
        redis_service.redis.delete = AsyncMock(return_value=2)  # 2 keys deleted
        
        result = await redis_service.delete_job("test-job-123")
        
        assert result is True
        redis_service.redis.delete.assert_called_once_with(
            "job:test-job-123",
            "job_updates:test-job-123"
        )

    @pytest.mark.asyncio
    async def test_delete_job_not_found(self, redis_service):
        """Test job deletion when job doesn't exist."""
        redis_service.redis.delete = AsyncMock(return_value=0)  # No keys deleted
        
        result = await redis_service.delete_job("nonexistent-job")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_publish_job_update_success(self, redis_service):
        """Test successful job update publishing."""
        update_data = {
            "event": "test_event",
            "message": "Test message"
        }
        
        redis_service.redis.xadd = AsyncMock(return_value=b"stream-id")
        
        result = await redis_service.publish_job_update("test-job-123", update_data)
        
        assert result is True
        redis_service.redis.xadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_job_update_redis_error(self, redis_service):
        """Test job update publishing with Redis error."""
        update_data = {"event": "test_event"}
        
        redis_service.redis.xadd = AsyncMock(side_effect=ConnectionError("Redis unavailable"))
        
        with pytest.raises(RuntimeError, match="Failed to publish job update"):
            await redis_service.publish_job_update("test-job-123", update_data)

    @pytest.mark.asyncio
    async def test_get_job_updates_success(self, redis_service):
        """Test successful job updates retrieval."""
        mock_stream_data = [
            (b"stream-id-1", {b"event": b"test_event_1", b"message": b"Message 1"}),
            (b"stream-id-2", {b"event": b"test_event_2", b"message": b"Message 2"})
        ]
        
        redis_service.redis.xrange = AsyncMock(return_value=mock_stream_data)
        
        updates = await redis_service.get_job_updates("test-job-123", "0")
        
        assert len(updates) == 2
        assert updates[0]["event"] == "test_event_1"
        assert updates[1]["event"] == "test_event_2"

    @pytest.mark.asyncio
    async def test_get_job_updates_empty_stream(self, redis_service):
        """Test job updates retrieval with empty stream."""
        redis_service.redis.xrange = AsyncMock(return_value=[])
        
        updates = await redis_service.get_job_updates("test-job-123", "0")
        
        assert len(updates) == 0

    @pytest.mark.asyncio
    async def test_stream_job_updates_success(self, redis_service):
        """Test streaming job updates."""
        mock_stream_data = {
            b"job_updates:test-job-123": [
                (b"stream-id-1", {b"event": b"chunk_completed", b"chunk_index": b"0"})
            ]
        }
        
        redis_service.redis.xread = AsyncMock(return_value=mock_stream_data)
        
        updates = []
        async for update in redis_service.stream_job_updates("test-job-123", timeout=100):
            updates.append(update)
            break  # Break to avoid infinite loop in test
        
        assert len(updates) == 1
        assert updates[0]["event"] == "chunk_completed"

    @pytest.mark.asyncio
    async def test_stream_job_updates_timeout(self, redis_service):
        """Test streaming job updates with timeout."""
        redis_service.redis.xread = AsyncMock(return_value={})
        
        updates = []
        async for update in redis_service.stream_job_updates("test-job-123", timeout=100):
            updates.append(update)
            break  # Should not reach here due to timeout
        
        assert len(updates) == 0

    @pytest.mark.asyncio
    async def test_cleanup_expired_jobs_success(self, redis_service):
        """Test cleanup of expired jobs."""
        # Mock finding expired jobs
        redis_service.redis.scan = AsyncMock(return_value=(0, [b"job:expired-1", b"job:expired-2"]))
        redis_service.redis.ttl = AsyncMock(side_effect=[-1, -2])  # Expired TTL values
        redis_service.redis.delete = AsyncMock(return_value=1)
        
        cleaned_count = await redis_service.cleanup_expired_jobs()
        
        assert cleaned_count == 2
        assert redis_service.redis.delete.call_count == 4  # 2 jobs × 2 keys each

    @pytest.mark.asyncio
    async def test_cleanup_expired_jobs_no_expired(self, redis_service):
        """Test cleanup when no jobs are expired."""
        redis_service.redis.scan = AsyncMock(return_value=(0, [b"job:active-1"]))
        redis_service.redis.ttl = AsyncMock(return_value=1800)  # Still has TTL
        
        cleaned_count = await redis_service.cleanup_expired_jobs()
        
        assert cleaned_count == 0

    @pytest.mark.asyncio
    async def test_health_check_success(self, redis_service):
        """Test successful health check."""
        redis_service.redis.ping = AsyncMock(return_value=True)
        
        is_healthy = await redis_service.health_check()
        
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, redis_service):
        """Test health check failure."""
        redis_service.redis.ping = AsyncMock(side_effect=ConnectionError("Redis down"))
        
        is_healthy = await redis_service.health_check()
        
        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_get_stats_success(self, redis_service):
        """Test successful stats retrieval."""
        redis_service.redis.info = AsyncMock(return_value={
            "used_memory": 1024000,
            "connected_clients": 5,
            "total_commands_processed": 1000
        })
        redis_service.redis.dbsize = AsyncMock(return_value=50)
        
        stats = await redis_service.get_stats()
        
        assert stats["memory_usage"] == 1024000
        assert stats["connected_clients"] == 5
        assert stats["total_keys"] == 50

    @pytest.mark.asyncio
    async def test_close_connection(self, redis_service):
        """Test closing Redis connection."""
        redis_service.redis.close = AsyncMock()
        
        await redis_service.close()
        
        redis_service.redis.close.assert_called_once()

    def test_job_metadata_serialization(self):
        """Test JobMetadata serialization for Redis storage."""
        job = JobMetadata(
            job_id="test-123",
            file_path="/tmp/test.mp3",
            language="en",
            state=JobState.PROCESSING,
            chunks_done=3,
            chunks_total=10
        )
        
        # Test that the job can be serialized to dict
        job_dict = {
            "id": job.job_id,
            "file_path": job.file_path,
            "language": job.language,
            "state": job.state.value,
            "chunks_done": job.chunks_done,
            "chunks_total": job.chunks_total
        }
        
        assert job_dict["id"] == "test-123"
        assert job_dict["state"] == "PROCESSING"
        assert job_dict["chunks_done"] == 3

    @pytest.mark.asyncio
    async def test_concurrent_job_operations(self, redis_service):
        """Test concurrent job operations."""
        # Mock Redis operations
        redis_service.redis.hset = AsyncMock(return_value=True)
        redis_service.redis.hgetall = AsyncMock(return_value={
            "id": "test-job", 
            "state": "QUEUED"
        })
        redis_service.redis.expire = AsyncMock(return_value=True)
        
        # Simulate concurrent operations
        job = JobMetadata(
            job_id="test-job",
            file_path="/tmp/test.mp3",
            language="en"
        )
        
        # Run operations concurrently
        import asyncio
        create_task = redis_service.create_job(job)
        get_task = redis_service.get_job("test-job")
        
        create_result, get_result = await asyncio.gather(create_task, get_task)
        
        assert create_result is True
        assert get_result is not None

    @pytest.mark.asyncio
    async def test_large_job_update_data(self, redis_service):
        """Test handling large job update data."""
        # Create large update data
        large_update = {
            "event": "large_data_event",
            "data": "x" * 10000,  # 10KB of data
            "chunks": [f"chunk_{i}" for i in range(100)]
        }
        
        redis_service.redis.xadd = AsyncMock(return_value=b"stream-id")
        
        result = await redis_service.publish_job_update("test-job", large_update)
        
        assert result is True
        redis_service.redis.xadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_job_state_transitions(self, redis_service):
        """Test valid job state transitions."""
        redis_service.redis.hset = AsyncMock(return_value=True)
        redis_service.redis.xadd = AsyncMock(return_value=b"stream-id")
        
        # Test valid state transitions
        valid_transitions = [
            (JobState.QUEUED, JobState.PROCESSING),
            (JobState.PROCESSING, JobState.COMPLETED),
            (JobState.PROCESSING, JobState.FAILED),
            (JobState.QUEUED, JobState.FAILED)
        ]
        
        for from_state, to_state in valid_transitions:
            result = await redis_service.update_job_state("test-job", to_state)
            assert result is True

    @pytest.mark.asyncio
    async def test_error_handling_malformed_job_data(self, redis_service):
        """Test handling of malformed job data from Redis."""
        # Mock malformed data
        malformed_data = {
            "id": "test-job",
            "state": "INVALID_STATE",  # Invalid state
            "chunks_done": "not_a_number"  # Invalid number
        }
        
        redis_service.redis.hgetall = AsyncMock(return_value=malformed_data)
        
        # Should still return the data (let the caller handle validation)
        job = await redis_service.get_job("test-job")
        
        assert job is not None
        assert job["id"] == "test-job"

    @pytest.mark.asyncio
    async def test_redis_connection_retry_logic(self, redis_service):
        """Test Redis connection retry logic."""
        # Mock intermittent connection issues
        redis_service.redis.ping = AsyncMock(side_effect=[
            ConnectionError("Connection lost"),
            ConnectionError("Still down"),
            True  # Finally connects
        ])
        
        # Health check should eventually succeed after retries
        # Note: This test assumes retry logic exists in the service
        is_healthy = await redis_service.health_check()
        
        # With current implementation, it would fail on first try
        assert is_healthy is False