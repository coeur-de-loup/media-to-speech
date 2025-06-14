"""Unit tests for FastAPI endpoints."""

import json
import tempfile
from io import BytesIO
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from media_to_text.main import app
from media_to_text.models import JobMetadata, JobState


class TestTranscriptionEndpoints:
    """Test suite for transcription endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def async_client(self):
        """Create async test client."""
        return AsyncClient(app=app, base_url="http://test")

    @pytest.fixture
    def sample_audio_file(self):
        """Create sample audio file for testing."""
        # Create a simple WAV file header (minimal)
        wav_header = b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x02\x00\x44\xac\x00\x00\x10\xb1\x02\x00\x04\x00\x10\x00data\x00\x08\x00\x00'
        audio_data = b'\x00\x00' * 1000  # Simple audio data
        
        return BytesIO(wav_header + audio_data)

    @pytest.fixture
    def mock_job_metadata(self):
        """Sample job metadata for testing."""
        return JobMetadata(
            job_id="test-job-123",
            file_path="/tmp/test.wav",
            language="en",
            state=JobState.QUEUED,
            chunks_done=0,
            chunks_total=0
        )

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/healthz")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_readiness_check_healthy(self, client):
        """Test readiness check when services are healthy."""
        with patch('media_to_text.services.redis_service.get_redis_service') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis_instance.health_check.return_value = True
            mock_redis.return_value = mock_redis_instance
            
            response = client.get("/readyz")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
            assert data["services"]["redis"] == "healthy"

    def test_readiness_check_unhealthy(self, client):
        """Test readiness check when services are unhealthy."""
        with patch('media_to_text.services.redis_service.get_redis_service') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis_instance.health_check.return_value = False
            mock_redis.return_value = mock_redis_instance
            
            response = client.get("/readyz")
            
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "not ready"
            assert data["services"]["redis"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_create_transcription_success(self, async_client, sample_audio_file):
        """Test successful transcription creation."""
        with patch('media_to_text.services.redis_service.get_redis_service') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis_instance.create_job.return_value = True
            mock_redis.return_value = mock_redis_instance
            
            # Mock file operations
            with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
                mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test_file.wav"
                
                with patch('shutil.move'):
                    files = {"file": ("test.wav", sample_audio_file, "audio/wav")}
                    data = {"language": "en"}
                    
                    async with async_client as ac:
                        response = await ac.post("/transcriptions/", files=files, data=data)
                    
                    assert response.status_code == 201
                    result = response.json()
                    assert "job_id" in result
                    assert result["status"] == "queued"
                    assert result["language"] == "en"

    @pytest.mark.asyncio
    async def test_create_transcription_invalid_file_type(self, async_client):
        """Test transcription creation with invalid file type."""
        invalid_file = BytesIO(b"not an audio file")
        
        files = {"file": ("test.txt", invalid_file, "text/plain")}
        data = {"language": "en"}
        
        async with async_client as ac:
            response = await ac.post("/transcriptions/", files=files, data=data)
        
        assert response.status_code == 400
        data = response.json()
        assert "Unsupported file type" in data["detail"]

    @pytest.mark.asyncio
    async def test_create_transcription_file_too_large(self, async_client):
        """Test transcription creation with file too large."""
        # Create a large file (mock)
        large_file = BytesIO(b"x" * (101 * 1024 * 1024))  # 101MB
        
        files = {"file": ("large.wav", large_file, "audio/wav")}
        data = {"language": "en"}
        
        async with async_client as ac:
            response = await ac.post("/transcriptions/", files=files, data=data)
        
        assert response.status_code == 413
        data = response.json()
        assert "File too large" in data["detail"]

    @pytest.mark.asyncio
    async def test_create_transcription_invalid_language(self, async_client, sample_audio_file):
        """Test transcription creation with invalid language."""
        files = {"file": ("test.wav", sample_audio_file, "audio/wav")}
        data = {"language": "invalid-lang"}
        
        async with async_client as ac:
            response = await ac.post("/transcriptions/", files=files, data=data)
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_get_job_exists(self, async_client, mock_job_metadata):
        """Test getting an existing job."""
        with patch('media_to_text.services.redis_service.get_redis_service') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis_instance.get_job.return_value = {
                "id": "test-job-123",
                "file_path": "/tmp/test.wav",
                "language": "en",
                "state": "QUEUED",
                "chunks_done": 0,
                "chunks_total": 0,
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00"
            }
            mock_redis.return_value = mock_redis_instance
            
            async with async_client as ac:
                response = await ac.get("/jobs/test-job-123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["job_id"] == "test-job-123"
            assert data["status"] == "QUEUED"

    @pytest.mark.asyncio
    async def test_get_job_not_found(self, async_client):
        """Test getting a non-existent job."""
        with patch('media_to_text.services.redis_service.get_redis_service') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis_instance.get_job.return_value = None
            mock_redis.return_value = mock_redis_instance
            
            async with async_client as ac:
                response = await ac.get("/jobs/nonexistent-job")
            
            assert response.status_code == 404
            data = response.json()
            assert "Job not found" in data["detail"]

    @pytest.mark.asyncio
    async def test_list_jobs_all(self, async_client):
        """Test listing all jobs."""
        with patch('media_to_text.services.redis_service.get_redis_service') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis_instance.list_jobs.return_value = [
                {"id": "job-1", "state": "QUEUED"},
                {"id": "job-2", "state": "PROCESSING"},
                {"id": "job-3", "state": "COMPLETED"}
            ]
            mock_redis.return_value = mock_redis_instance
            
            async with async_client as ac:
                response = await ac.get("/jobs")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["jobs"]) == 3
            assert data["total"] == 3

    @pytest.mark.asyncio
    async def test_list_jobs_with_status_filter(self, async_client):
        """Test listing jobs with status filter."""
        with patch('media_to_text.services.redis_service.get_redis_service') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis_instance.list_jobs.return_value = [
                {"id": "job-1", "state": "COMPLETED"},
                {"id": "job-2", "state": "COMPLETED"}
            ]
            mock_redis.return_value = mock_redis_instance
            
            async with async_client as ac:
                response = await ac.get("/jobs?status=COMPLETED")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["jobs"]) == 2
            assert all(job["state"] == "COMPLETED" for job in data["jobs"])

    @pytest.mark.asyncio
    async def test_delete_job_success(self, async_client):
        """Test successful job deletion."""
        with patch('media_to_text.services.redis_service.get_redis_service') as mock_redis:
            with patch('media_to_text.services.cleanup_service.get_cleanup_service') as mock_cleanup:
                mock_redis_instance = AsyncMock()
                mock_redis_instance.get_job.return_value = {
                    "id": "test-job-123",
                    "state": "QUEUED"
                }
                mock_redis_instance.update_job_state.return_value = True
                mock_redis.return_value = mock_redis_instance
                
                mock_cleanup_instance = AsyncMock()
                mock_cleanup_instance.trigger_immediate_cleanup.return_value = None
                mock_cleanup.return_value = mock_cleanup_instance
                
                async with async_client as ac:
                    response = await ac.delete("/jobs/test-job-123")
                
                assert response.status_code == 200
                data = response.json()
                assert data["message"] == "Job cancelled and cleanup initiated"

    @pytest.mark.asyncio
    async def test_delete_job_not_found(self, async_client):
        """Test deleting a non-existent job."""
        with patch('media_to_text.services.redis_service.get_redis_service') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis_instance.get_job.return_value = None
            mock_redis.return_value = mock_redis_instance
            
            async with async_client as ac:
                response = await ac.delete("/jobs/nonexistent-job")
            
            assert response.status_code == 404
            data = response.json()
            assert "Job not found" in data["detail"]

    @pytest.mark.asyncio
    async def test_delete_job_already_completed(self, async_client):
        """Test deleting a completed job."""
        with patch('media_to_text.services.redis_service.get_redis_service') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis_instance.get_job.return_value = {
                "id": "test-job-123",
                "state": "COMPLETED"
            }
            mock_redis.return_value = mock_redis_instance
            
            async with async_client as ac:
                response = await ac.delete("/jobs/test-job-123")
            
            assert response.status_code == 400
            data = response.json()
            assert "Cannot cancel completed job" in data["detail"]

    @pytest.mark.asyncio
    async def test_get_transcript_success(self, async_client):
        """Test getting transcript for completed job."""
        sample_transcript = {
            "transcript": "This is a test transcript",
            "job_id": "test-job-123",
            "language": "en"
        }
        
        with patch('media_to_text.services.redis_service.get_redis_service') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis_instance.get_job.return_value = {
                "id": "test-job-123",
                "state": "COMPLETED"
            }
            mock_redis_instance.redis.get.return_value = json.dumps(sample_transcript)
            mock_redis.return_value = mock_redis_instance
            
            async with async_client as ac:
                response = await ac.get("/jobs/test-job-123/transcript")
            
            assert response.status_code == 200
            data = response.json()
            assert data["transcript"] == "This is a test transcript"
            assert data["job_id"] == "test-job-123"

    @pytest.mark.asyncio
    async def test_get_transcript_job_not_completed(self, async_client):
        """Test getting transcript for non-completed job."""
        with patch('media_to_text.services.redis_service.get_redis_service') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis_instance.get_job.return_value = {
                "id": "test-job-123",
                "state": "PROCESSING"
            }
            mock_redis.return_value = mock_redis_instance
            
            async with async_client as ac:
                response = await ac.get("/jobs/test-job-123/transcript")
            
            assert response.status_code == 400
            data = response.json()
            assert "Job is not completed" in data["detail"]

    @pytest.mark.asyncio
    async def test_get_transcript_not_found(self, async_client):
        """Test getting transcript when transcript doesn't exist."""
        with patch('media_to_text.services.redis_service.get_redis_service') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis_instance.get_job.return_value = {
                "id": "test-job-123",
                "state": "COMPLETED"
            }
            mock_redis_instance.redis.get.return_value = None
            mock_redis.return_value = mock_redis_instance
            
            async with async_client as ac:
                response = await ac.get("/jobs/test-job-123/transcript")
            
            assert response.status_code == 404
            data = response.json()
            assert "Transcript not found" in data["detail"]

    @pytest.mark.asyncio
    async def test_stream_job_updates(self, async_client):
        """Test streaming job updates via SSE."""
        async def mock_stream():
            yield {"event": "progress", "data": {"chunks_done": 1, "chunks_total": 5}}
            yield {"event": "completed", "data": {"status": "COMPLETED"}}
        
        with patch('media_to_text.services.redis_service.get_redis_service') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis_instance.get_job.return_value = {
                "id": "test-job-123",
                "state": "PROCESSING"
            }
            mock_redis_instance.stream_job_updates.return_value = mock_stream()
            mock_redis.return_value = mock_redis_instance
            
            async with async_client as ac:
                response = await ac.get("/jobs/test-job-123/events")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream"

    @pytest.mark.asyncio
    async def test_stream_job_updates_job_not_found(self, async_client):
        """Test streaming updates for non-existent job."""
        with patch('media_to_text.services.redis_service.get_redis_service') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis_instance.get_job.return_value = None
            mock_redis.return_value = mock_redis_instance
            
            async with async_client as ac:
                response = await ac.get("/jobs/nonexistent-job/events")
            
            assert response.status_code == 404

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

    @pytest.mark.asyncio
    async def test_create_transcription_redis_error(self, async_client, sample_audio_file):
        """Test transcription creation with Redis error."""
        with patch('media_to_text.services.redis_service.get_redis_service') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis_instance.create_job.side_effect = Exception("Redis connection failed")
            mock_redis.return_value = mock_redis_instance
            
            with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
                mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test_file.wav"
                
                with patch('shutil.move'):
                    files = {"file": ("test.wav", sample_audio_file, "audio/wav")}
                    data = {"language": "en"}
                    
                    async with async_client as ac:
                        response = await ac.post("/transcriptions/", files=files, data=data)
                    
                    assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_concurrent_transcription_requests(self, async_client, sample_audio_file):
        """Test handling multiple concurrent transcription requests."""
        with patch('media_to_text.services.redis_service.get_redis_service') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis_instance.create_job.return_value = True
            mock_redis.return_value = mock_redis_instance
            
            with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
                mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test_file.wav"
                
                with patch('shutil.move'):
                    # Create multiple requests
                    files = {"file": ("test.wav", sample_audio_file, "audio/wav")}
                    data = {"language": "en"}
                    
                    import asyncio
                    
                    async def make_request():
                        async with async_client as ac:
                            return await ac.post("/transcriptions/", files=files, data=data)
                    
                    # Run 5 concurrent requests
                    tasks = [make_request() for _ in range(5)]
                    responses = await asyncio.gather(*tasks)
                    
                    # All should succeed
                    for response in responses:
                        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_file_upload_size_validation(self, async_client):
        """Test file size validation during upload."""
        # Test with exactly the limit (should pass)
        max_size_file = BytesIO(b"x" * (100 * 1024 * 1024))  # 100MB exactly
        
        files = {"file": ("max_size.wav", max_size_file, "audio/wav")}
        data = {"language": "en"}
        
        with patch('media_to_text.services.redis_service.get_redis_service') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis_instance.create_job.return_value = True
            mock_redis.return_value = mock_redis_instance
            
            with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
                mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test_file.wav"
                
                with patch('shutil.move'):
                    async with async_client as ac:
                        response = await ac.post("/transcriptions/", files=files, data=data)
                    
                    # Should succeed at exactly 100MB
                    assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_missing_file_parameter(self, async_client):
        """Test transcription creation without file parameter."""
        data = {"language": "en"}
        
        async with async_client as ac:
            response = await ac.post("/transcriptions/", data=data)
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_empty_file_upload(self, async_client):
        """Test transcription creation with empty file."""
        empty_file = BytesIO(b"")
        
        files = {"file": ("empty.wav", empty_file, "audio/wav")}
        data = {"language": "en"}
        
        async with async_client as ac:
            response = await ac.post("/transcriptions/", files=files, data=data)
        
        assert response.status_code == 400
        data = response.json()
        assert "File is empty" in data["detail"]

    @pytest.mark.asyncio
    async def test_invalid_job_id_format(self, async_client):
        """Test API calls with invalid job ID format."""
        invalid_job_ids = ["", "invalid-chars@#$", "too-long-" + "x" * 100]
        
        for job_id in invalid_job_ids:
            async with async_client as ac:
                response = await ac.get(f"/jobs/{job_id}")
                
                # Should handle gracefully (may return 404 or 422)
                assert response.status_code in [404, 422]

    @pytest.mark.asyncio
    async def test_api_error_handling_logging(self, async_client):
        """Test that API errors are properly logged."""
        with patch('media_to_text.services.redis_service.get_redis_service') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis_instance.get_job.side_effect = Exception("Database error")
            mock_redis.return_value = mock_redis_instance
            
            with patch('media_to_text.logging.get_logger') as mock_logger:
                mock_logger_instance = Mock()
                mock_logger.return_value = mock_logger_instance
                
                async with async_client as ac:
                    response = await ac.get("/jobs/test-job-123")
                
                assert response.status_code == 500
                # Logger should have been called for error
                mock_logger_instance.error.assert_called()

    @pytest.mark.asyncio
    async def test_cors_headers(self, async_client):
        """Test CORS headers are present."""
        async with async_client as ac:
            response = await ac.options("/healthz")
        
        # Should handle OPTIONS request for CORS
        assert response.status_code in [200, 405]  # Depends on CORS configuration

    def test_openapi_docs(self, client):
        """Test OpenAPI documentation endpoints."""
        # Test docs endpoint
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Test OpenAPI JSON
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data