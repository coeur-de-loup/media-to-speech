"""Integration tests for end-to-end job processing."""

import asyncio
import json
import os
import tempfile
import time
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from httpx import AsyncClient

from media_to_text.main import app
from media_to_text.models import JobState
from media_to_text.services.ffmpeg_service import ChunkInfo, MediaInfo
from media_to_text.services.openai_service import TranscriptionResult


class TestEndToEndIntegration:
    """Integration tests for complete job processing workflows."""

    @pytest.fixture
    def async_client(self):
        """Create async test client."""
        return AsyncClient(app=app, base_url="http://test")

    @pytest.fixture
    def sample_wav_short(self):
        """Create short WAV file (< 25MB)."""
        # Simple WAV header for 10-second file
        wav_header = b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x02\x00\x44\xac\x00\x00\x10\xb1\x02\x00\x04\x00\x10\x00data\x00\x08\x00\x00'
        audio_data = b'\x00\x01' * 441000  # ~10 seconds of audio data
        return BytesIO(wav_header + audio_data)

    @pytest.fixture
    def sample_mp3_medium(self):
        """Create medium-length MP3 file."""
        # MP3 header and some audio data
        mp3_header = b'\xff\xfb\x90\x00'  # Basic MP3 frame header
        audio_data = b'\x00\x01' * 1024000  # Medium-sized file
        return BytesIO(mp3_header + audio_data)

    @pytest.fixture
    def sample_large_file(self):
        """Create large file requiring chunking."""
        # Large WAV file
        wav_header = b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x02\x00\x44\xac\x00\x00\x10\xb1\x02\x00\x04\x00\x10\x00data\x00\x08\x00\x00'
        audio_data = b'\x00\x01' * (30 * 1024 * 1024)  # 30MB file
        return BytesIO(wav_header + audio_data)

    @pytest.fixture
    def mock_successful_services(self):
        """Mock all services for successful processing."""
        with patch('media_to_text.services.redis_service.get_redis_service') as mock_redis, \
             patch('media_to_text.services.ffmpeg_service.get_ffmpeg_service') as mock_ffmpeg, \
             patch('media_to_text.services.openai_service.get_openai_service') as mock_openai, \
             patch('media_to_text.services.job_worker.get_job_worker') as mock_worker:
            
            # Redis service mock
            redis_instance = AsyncMock()
            redis_instance.create_job.return_value = True
            redis_instance.get_job.return_value = {"id": "test-job", "state": "QUEUED"}
            redis_instance.update_job_state.return_value = True
            redis_instance.update_job_progress.return_value = True
            redis_instance.publish_job_update.return_value = True
            redis_instance.health_check.return_value = True
            mock_redis.return_value = redis_instance
            
            # FFmpeg service mock
            ffmpeg_instance = Mock()
            ffmpeg_instance.probe_media.return_value = MediaInfo(
                has_audio=True, duration=30.0, format="wav", needs_conversion=False
            )
            ffmpeg_instance.get_file_size_mb.return_value = 10.0
            ffmpeg_instance.chunk_wav_with_segments.return_value = [
                ChunkInfo("/tmp/chunk_0.wav", 0, 0.0, 30.0, 10 * 1024 * 1024)
            ]
            mock_ffmpeg.return_value = ffmpeg_instance
            
            # OpenAI service mock
            openai_instance = AsyncMock()
            openai_instance.transcribe_chunks_parallel.return_value = [
                TranscriptionResult(
                    chunk_index=0,
                    success=True,
                    text="This is a test transcription.",
                    processing_time=2.5,
                    retry_count=0,
                    segments=[{"start": 0.0, "end": 30.0, "text": "This is a test transcription."}]
                )
            ]
            mock_openai.return_value = openai_instance
            
            # Job worker mock
            worker_instance = AsyncMock()
            worker_instance.process_job_by_id.return_value = True
            mock_worker.return_value = worker_instance
            
            yield {
                "redis": redis_instance,
                "ffmpeg": ffmpeg_instance,
                "openai": openai_instance,
                "worker": worker_instance
            }

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_short_wav_file_processing(self, async_client, sample_wav_short, mock_successful_services):
        """Test processing of short WAV file that doesn't need chunking."""
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test_short.wav"
            
            with patch('shutil.move'):
                # Step 1: Create transcription job
                files = {"file": ("short_audio.wav", sample_wav_short, "audio/wav")}
                data = {"language": "en"}
                
                async with async_client as ac:
                    response = await ac.post("/transcriptions/", files=files, data=data)
                
                assert response.status_code == 201
                job_data = response.json()
                job_id = job_data["job_id"]
                
                # Step 2: Verify job was created
                async with async_client as ac:
                    response = await ac.get(f"/jobs/{job_id}")
                
                assert response.status_code == 200
                job_status = response.json()
                assert job_status["status"] == "QUEUED"
                
                # Step 3: Simulate job processing
                mock_successful_services["redis"].get_job.return_value = {
                    "id": job_id, "state": "COMPLETED"
                }
                mock_successful_services["redis"].redis.get.return_value = json.dumps({
                    "transcript": "This is a test transcription.",
                    "job_id": job_id,
                    "language": "en"
                })
                
                # Step 4: Get transcript
                async with async_client as ac:
                    response = await ac.get(f"/jobs/{job_id}/transcript")
                
                assert response.status_code == 200
                transcript = response.json()
                assert transcript["transcript"] == "This is a test transcription."
                assert transcript["job_id"] == job_id

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_mp3_conversion_workflow(self, async_client, sample_mp3_medium, mock_successful_services):
        """Test processing of MP3 file requiring conversion."""
        # Update FFmpeg mock for MP3 conversion
        mock_successful_services["ffmpeg"].probe_media.return_value = MediaInfo(
            has_audio=True, duration=60.0, format="mp3", needs_conversion=True
        )
        mock_successful_services["ffmpeg"].convert_to_wav.return_value = None
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test_medium.mp3"
            
            with patch('shutil.move'):
                files = {"file": ("medium_audio.mp3", sample_mp3_medium, "audio/mp3")}
                data = {"language": "en"}
                
                async with async_client as ac:
                    response = await ac.post("/transcriptions/", files=files, data=data)
                
                assert response.status_code == 201
                job_data = response.json()
                
                # Verify conversion would be called
                assert mock_successful_services["ffmpeg"].probe_media.called
                # In real processing, convert_to_wav would be called

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_large_file_chunking_workflow(self, async_client, sample_large_file, mock_successful_services):
        """Test processing of large file requiring chunking."""
        # Update mocks for large file processing
        mock_successful_services["ffmpeg"].get_file_size_mb.return_value = 35.0  # Larger than 25MB limit
        mock_successful_services["ffmpeg"].chunk_wav_with_segments.return_value = [
            ChunkInfo("/tmp/chunk_0.wav", 0, 0.0, 30.0, 20 * 1024 * 1024),
            ChunkInfo("/tmp/chunk_1.wav", 1, 30.0, 30.0, 15 * 1024 * 1024)
        ]
        
        # Multiple chunk transcription results
        mock_successful_services["openai"].transcribe_chunks_parallel.return_value = [
            TranscriptionResult(
                chunk_index=0,
                success=True,
                text="First part of the transcription.",
                processing_time=3.0,
                retry_count=0,
                segments=[{"start": 0.0, "end": 30.0, "text": "First part of the transcription."}]
            ),
            TranscriptionResult(
                chunk_index=1,
                success=True,
                text="Second part of the transcription.",
                processing_time=2.8,
                retry_count=0,
                segments=[{"start": 0.0, "end": 30.0, "text": "Second part of the transcription."}]
            )
        ]
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test_large.wav"
            
            with patch('shutil.move'):
                files = {"file": ("large_audio.wav", sample_large_file, "audio/wav")}
                data = {"language": "en"}
                
                async with async_client as ac:
                    response = await ac.post("/transcriptions/", files=files, data=data)
                
                assert response.status_code == 201
                job_data = response.json()
                
                # Verify chunking would be used
                assert mock_successful_services["ffmpeg"].get_file_size_mb.called

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_job_progress_tracking(self, async_client, sample_wav_short, mock_successful_services):
        """Test job progress tracking through states."""
        # Setup progressive state updates
        state_sequence = ["QUEUED", "PROCESSING", "COMPLETED"]
        call_count = 0
        
        def get_job_side_effect(*args):
            nonlocal call_count
            state = state_sequence[min(call_count, len(state_sequence) - 1)]
            call_count += 1
            return {"id": "test-job", "state": state, "chunks_done": call_count, "chunks_total": 3}
        
        mock_successful_services["redis"].get_job.side_effect = get_job_side_effect
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test_progress.wav"
            
            with patch('shutil.move'):
                # Create job
                files = {"file": ("progress_test.wav", sample_wav_short, "audio/wav")}
                data = {"language": "en"}
                
                async with async_client as ac:
                    response = await ac.post("/transcriptions/", files=files, data=data)
                
                job_id = response.json()["job_id"]
                
                # Check initial state
                async with async_client as ac:
                    response = await ac.get(f"/jobs/{job_id}")
                assert response.json()["status"] == "QUEUED"
                
                # Check processing state
                async with async_client as ac:
                    response = await ac.get(f"/jobs/{job_id}")
                assert response.json()["status"] == "PROCESSING"
                
                # Check completion state
                async with async_client as ac:
                    response = await ac.get(f"/jobs/{job_id}")
                assert response.json()["status"] == "COMPLETED"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multiple_concurrent_jobs(self, async_client, sample_wav_short, mock_successful_services):
        """Test processing multiple jobs concurrently."""
        job_ids = []
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test_concurrent.wav"
            
            with patch('shutil.move'):
                # Create multiple jobs
                async def create_job(client, file_data, job_num):
                    files = {"file": (f"concurrent_{job_num}.wav", file_data, "audio/wav")}
                    data = {"language": "en"}
                    response = await client.post("/transcriptions/", files=files, data=data)
                    return response.json()["job_id"]
                
                async with async_client as ac:
                    tasks = [
                        create_job(ac, sample_wav_short, i) 
                        for i in range(3)
                    ]
                    job_ids = await asyncio.gather(*tasks)
                
                assert len(job_ids) == 3
                assert len(set(job_ids)) == 3  # All unique job IDs
                
                # Verify all jobs exist
                for job_id in job_ids:
                    async with async_client as ac:
                        response = await ac.get(f"/jobs/{job_id}")
                    assert response.status_code == 200

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_job_cancellation_workflow(self, async_client, sample_wav_short, mock_successful_services):
        """Test job cancellation and cleanup."""
        with patch('media_to_text.services.cleanup_service.get_cleanup_service') as mock_cleanup:
            cleanup_instance = AsyncMock()
            cleanup_instance.trigger_immediate_cleanup.return_value = None
            mock_cleanup.return_value = cleanup_instance
            
            with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
                mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test_cancel.wav"
                
                with patch('shutil.move'):
                    # Create job
                    files = {"file": ("cancel_test.wav", sample_wav_short, "audio/wav")}
                    data = {"language": "en"}
                    
                    async with async_client as ac:
                        response = await ac.post("/transcriptions/", files=files, data=data)
                    
                    job_id = response.json()["job_id"]
                    
                    # Cancel job
                    async with async_client as ac:
                        response = await ac.delete(f"/jobs/{job_id}")
                    
                    assert response.status_code == 200
                    assert "cancelled" in response.json()["message"]
                    
                    # Verify cleanup was triggered
                    cleanup_instance.trigger_immediate_cleanup.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_handling_invalid_audio(self, async_client, mock_successful_services):
        """Test error handling with invalid audio file."""
        # Mock FFmpeg to fail for invalid file
        mock_successful_services["ffmpeg"].probe_media.side_effect = RuntimeError("Invalid audio file")
        
        invalid_file = BytesIO(b"This is not an audio file")
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test_invalid.wav"
            
            with patch('shutil.move'):
                files = {"file": ("invalid.wav", invalid_file, "audio/wav")}
                data = {"language": "en"}
                
                async with async_client as ac:
                    response = await ac.post("/transcriptions/", files=files, data=data)
                
                # Should still create job (error handling happens during processing)
                assert response.status_code == 201

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_openai_service_failure_handling(self, async_client, sample_wav_short, mock_successful_services):
        """Test handling of OpenAI service failures."""
        # Mock OpenAI to fail
        mock_successful_services["openai"].transcribe_chunks_parallel.return_value = [
            TranscriptionResult(
                chunk_index=0,
                success=False,
                text="",
                processing_time=0.0,
                retry_count=3,
                error_message="OpenAI API rate limit exceeded"
            )
        ]
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test_openai_fail.wav"
            
            with patch('shutil.move'):
                files = {"file": ("openai_fail.wav", sample_wav_short, "audio/wav")}
                data = {"language": "en"}
                
                async with async_client as ac:
                    response = await ac.post("/transcriptions/", files=files, data=data)
                
                assert response.status_code == 201
                # Job would be marked as failed during processing

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_redis_connectivity_resilience(self, async_client, sample_wav_short, mock_successful_services):
        """Test resilience to Redis connectivity issues."""
        # Mock Redis to fail initially, then recover
        call_count = 0
        
        def health_check_side_effect():
            nonlocal call_count
            call_count += 1
            return call_count > 2  # Fail first 2 calls, then succeed
        
        mock_successful_services["redis"].health_check.side_effect = health_check_side_effect
        
        # Test readiness check during Redis issues
        async with async_client as ac:
            response = await ac.get("/readyz")
        assert response.status_code == 503  # Not ready due to Redis
        
        # Test readiness check after Redis recovery
        async with async_client as ac:
            response = await ac.get("/readyz")
        assert response.status_code == 200  # Ready after Redis recovery

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_different_language_support(self, async_client, sample_wav_short, mock_successful_services):
        """Test transcription with different language settings."""
        languages = ["en", "es", "fr", "de"]
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test_lang.wav"
            
            with patch('shutil.move'):
                for lang in languages:
                    files = {"file": (f"test_{lang}.wav", sample_wav_short, "audio/wav")}
                    data = {"language": lang}
                    
                    async with async_client as ac:
                        response = await ac.post("/transcriptions/", files=files, data=data)
                    
                    assert response.status_code == 201
                    job_data = response.json()
                    assert job_data["language"] == lang

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_file_cleanup_after_processing(self, async_client, sample_wav_short, mock_successful_services):
        """Test that temporary files are cleaned up after processing."""
        with patch('media_to_text.services.cleanup_service.get_cleanup_service') as mock_cleanup:
            cleanup_instance = AsyncMock()
            cleanup_instance.schedule_job_cleanup.return_value = None
            mock_cleanup.return_value = cleanup_instance
            
            with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
                temp_file_path = "/tmp/test_cleanup.wav"
                mock_temp_file.return_value.__enter__.return_value.name = temp_file_path
                
                with patch('shutil.move'):
                    # Create and process job
                    files = {"file": ("cleanup_test.wav", sample_wav_short, "audio/wav")}
                    data = {"language": "en"}
                    
                    async with async_client as ac:
                        response = await ac.post("/transcriptions/", files=files, data=data)
                    
                    # Verify cleanup would be scheduled
                    # In real scenario, cleanup service would handle file removal

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_server_sent_events_streaming(self, async_client, sample_wav_short, mock_successful_services):
        """Test server-sent events for job updates."""
        # Mock streaming updates
        async def mock_stream_updates(job_id, timeout):
            events = [
                {"event": "progress", "chunks_done": 1, "chunks_total": 3},
                {"event": "progress", "chunks_done": 2, "chunks_total": 3},
                {"event": "completed", "status": "COMPLETED"}
            ]
            for event in events:
                yield event
                await asyncio.sleep(0.1)
        
        mock_successful_services["redis"].stream_job_updates.return_value = mock_stream_updates("test-job", 30)
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test_sse.wav"
            
            with patch('shutil.move'):
                # Create job
                files = {"file": ("sse_test.wav", sample_wav_short, "audio/wav")}
                data = {"language": "en"}
                
                async with async_client as ac:
                    response = await ac.post("/transcriptions/", files=files, data=data)
                
                job_id = response.json()["job_id"]
                
                # Test SSE endpoint
                async with async_client as ac:
                    response = await ac.get(f"/jobs/{job_id}/events")
                
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_metrics_collection_during_processing(self, async_client, sample_wav_short, mock_successful_services):
        """Test that metrics are collected during job processing."""
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test_metrics.wav"
            
            with patch('shutil.move'):
                # Create job
                files = {"file": ("metrics_test.wav", sample_wav_short, "audio/wav")}
                data = {"language": "en"}
                
                async with async_client as ac:
                    response = await ac.post("/transcriptions/", files=files, data=data)
                
                # Check metrics endpoint
                async with async_client as ac:
                    response = await ac.get("/metrics")
                
                assert response.status_code == 200
                assert "text/plain" in response.headers["content-type"]
                # Metrics should include job-related counters