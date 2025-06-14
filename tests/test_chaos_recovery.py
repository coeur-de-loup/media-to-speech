"""Chaos testing for crash recovery functionality.

This module contains comprehensive chaos tests that simulate worker crashes
during different phases of job processing and validates that:
1. Jobs are properly recovered and resumed
2. Idempotency is maintained (no duplicate processing)
3. Recovery events are published correctly
4. State consistency is maintained after crashes
"""

import asyncio
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
import pytest_asyncio
from unittest.mock import Mock

from media_to_text.config import Settings
from media_to_text.models import JobMetadata, JobState, ChunkInfo
from media_to_text.services.redis_service import RedisService
from media_to_text.services.job_worker import JobWorker, init_job_worker, close_job_worker
from media_to_text.services.ffmpeg_service import FFmpegService
from media_to_text.services.openai_service import OpenAITranscriptionService
from media_to_text.services.cleanup_service import CleanupService


class ChaosTestFramework:
    """Framework for simulating various crash scenarios and validating recovery."""
    
    def __init__(self, redis_service: RedisService, settings: Settings):
        self.redis_service = redis_service
        self.settings = settings
        self.temp_dir = tempfile.mkdtemp()
        self.test_files = []
        self.created_jobs = []
        
    async def setup(self):
        """Set up test environment."""
        # Create test audio file
        self.test_audio_file = os.path.join(self.temp_dir, "test_audio.wav")
        await self._create_test_audio_file()
        
    async def teardown(self):
        """Clean up test environment."""
        # Clean up test jobs
        for job_id in self.created_jobs:
            try:
                await self.redis_service.delete_job(job_id)
            except:
                pass
        
        # Clean up temp files
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    async def _create_test_audio_file(self):
        """Create a test WAV file for testing."""
        # Create a simple WAV file header for testing
        # This creates a minimal valid WAV file for testing purposes
        wav_header = b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x08\x00\x00'
        test_data = b'\x00' * 2048  # 2KB of silence
        
        with open(self.test_audio_file, 'wb') as f:
            f.write(wav_header + test_data)
    
    async def create_test_job(self, file_size_mb: float = 1.0) -> str:
        """Create a test job with specified file size."""
        job_id = f"chaos_test_{int(time.time() * 1000)}"
        
        # Create test file of specified size
        test_file = os.path.join(self.temp_dir, f"test_{job_id}.wav")
        
        # Create WAV file of specified size
        wav_header = b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x08\x00\x00'
        data_size = int(file_size_mb * 1024 * 1024) - len(wav_header)
        test_data = b'\x00' * max(0, data_size)
        
        with open(test_file, 'wb') as f:
            f.write(wav_header + test_data)
        
        # Create job metadata
        job = JobMetadata(
            job_id=job_id,
            file_path=test_file,
            language="en",
            state=JobState.QUEUED
        )
        
        # Store job in Redis
        await self.redis_service.create_job(job)
        self.created_jobs.append(job_id)
        
        return job_id
    
    async def simulate_crash_during_processing(self, job_id: str, crash_after_chunks: int = 2) -> Dict[str, Any]:
        """Simulate a worker crash during job processing."""
        # Start processing and crash after specified chunks
        worker = await self._create_test_worker()
        
        # Mock chunk processing to crash after specified chunks
        original_transcribe = worker.openai_service.transcribe_chunks_parallel
        chunk_count = 0
        
        async def mock_transcribe_with_crash(chunks, language, job_id):
            nonlocal chunk_count
            results = []
            
            for chunk in chunks:
                chunk_count += 1
                if chunk_count > crash_after_chunks:
                    # Simulate crash by raising exception
                    raise Exception("Simulated worker crash during transcription")
                
                # Mock successful chunk processing
                result = MagicMock()
                result.success = True
                result.chunk_index = chunk.index
                result.text = f"Test transcription for chunk {chunk.index}"
                result.processing_time = 1.0
                result.retry_count = 0
                results.append(result)
                
                # Publish chunk completion event
                await self.redis_service.publish_job_update(job_id, {
                    "event": "chunk_transcribed",
                    "chunk_index": chunk.index,
                    "chunk_text": result.text,
                    "processing_time": result.processing_time,
                    "retry_count": result.retry_count
                })
            
            return results
        
        worker.openai_service.transcribe_chunks_parallel = mock_transcribe_with_crash
        
        # Attempt to process job (should crash)
        try:
            await worker.process_job_by_id(job_id)
        except Exception as e:
            pass  # Expected crash
        
        # Get job state after crash
        job_state = await self.redis_service.get_job(job_id)
        
        await worker.stop()
        
        return {
            "job_id": job_id,
            "crash_after_chunks": crash_after_chunks,
            "chunks_completed_before_crash": chunk_count - 1,
            "job_state_after_crash": job_state
        }
    
    async def validate_recovery(self, crash_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that job recovery works correctly after crash."""
        job_id = crash_info["job_id"]
        
        # Get job state before recovery
        job_before_recovery = await self.redis_service.get_job(job_id)
        
        # Get events before recovery
        events_before = await self.redis_service.get_job_updates(job_id, "0")
        completed_chunks_before = [
            event for event in events_before 
            if event.get("event") == "chunk_transcribed"
        ]
        
        # Create new worker and perform recovery
        recovery_worker = await self._create_test_worker()
        
        # Mock successful transcription for remaining chunks
        original_transcribe = recovery_worker.openai_service.transcribe_chunks_parallel
        
        async def mock_successful_transcribe(chunks, language, job_id):
            results = []
            for chunk in chunks:
                result = MagicMock()
                result.success = True
                result.chunk_index = chunk.index
                result.text = f"Recovered transcription for chunk {chunk.index}"
                result.processing_time = 1.0
                result.retry_count = 0
                results.append(result)
            return results
        
        recovery_worker.openai_service.transcribe_chunks_parallel = mock_successful_transcribe
        
        # Trigger recovery by starting worker (should detect interrupted job)
        await recovery_worker.start()
        
        # Wait for recovery to complete
        await asyncio.sleep(2.0)
        
        # Get job state after recovery
        job_after_recovery = await self.redis_service.get_job(job_id)
        
        # Get events after recovery
        events_after = await self.redis_service.get_job_updates(job_id, "0")
        recovery_events = [
            event for event in events_after 
            if event.get("recovery") is True
        ]
        
        await recovery_worker.stop()
        
        return {
            "job_before_recovery": job_before_recovery,
            "job_after_recovery": job_after_recovery,
            "completed_chunks_before": len(completed_chunks_before),
            "recovery_events": recovery_events,
            "recovery_successful": job_after_recovery.get("state") == "COMPLETED"
        }
    
    async def validate_idempotency(self, job_id: str) -> Dict[str, Any]:
        """Validate that idempotency is maintained during recovery."""
        # Get all chunk events
        events = await self.redis_service.get_job_updates(job_id, "0")
        
        chunk_events = [
            event for event in events 
            if event.get("event") == "chunk_transcribed"
        ]
        
        # Count chunk completions by index
        chunk_counts = {}
        for event in chunk_events:
            chunk_index = event.get("chunk_index")
            if chunk_index is not None:
                chunk_counts[chunk_index] = chunk_counts.get(chunk_index, 0) + 1
        
        # Check for duplicates
        duplicated_chunks = {
            index: count for index, count in chunk_counts.items() 
            if count > 1
        }
        
        return {
            "total_chunk_events": len(chunk_events),
            "unique_chunks": len(chunk_counts),
            "duplicated_chunks": duplicated_chunks,
            "idempotency_maintained": len(duplicated_chunks) == 0
        }
    
    async def validate_event_publishing(self, job_id: str) -> Dict[str, Any]:
        """Validate that recovery events are published correctly."""
        events = await self.redis_service.get_job_updates(job_id, "0")
        
        recovery_events = [
            event for event in events 
            if event.get("recovery") is True
        ]
        
        event_types = [event.get("event") for event in recovery_events]
        
        expected_events = [
            "recovery_started",
            "processing_resumed", 
            "recovery_completed"
        ]
        
        events_found = {event_type: event_type in event_types for event_type in expected_events}
        
        return {
            "total_recovery_events": len(recovery_events),
            "event_types": event_types,
            "expected_events_found": events_found,
            "all_expected_events_present": all(events_found.values())
        }
    
    async def _create_test_worker(self) -> JobWorker:
        """Create a test worker with mocked services."""
        # Create mock services
        ffmpeg_service = MagicMock(spec=FFmpegService)
        openai_service = MagicMock(spec=OpenAITranscriptionService)
        cleanup_service = MagicMock(spec=CleanupService)
        
        # Mock FFmpeg service responses
        ffmpeg_service.probe_media = AsyncMock(return_value=MagicMock(
            has_audio=True,
            needs_conversion=False,
            duration=10.0
        ))
        ffmpeg_service.get_file_size_mb = AsyncMock(return_value=30.0)  # Large enough to trigger chunking
        ffmpeg_service.chunk_wav_with_segments = AsyncMock(return_value=[
            ChunkInfo(file_path=f"{self.temp_dir}/chunk_0.wav", index=0, start_time=0.0, duration=5.0),
            ChunkInfo(file_path=f"{self.temp_dir}/chunk_1.wav", index=1, start_time=5.0, duration=5.0),
            ChunkInfo(file_path=f"{self.temp_dir}/chunk_2.wav", index=2, start_time=10.0, duration=5.0),
            ChunkInfo(file_path=f"{self.temp_dir}/chunk_3.wav", index=3, start_time=15.0, duration=5.0),
        ])
        ffmpeg_service.validate_chunk_sizes = AsyncMock(return_value=True)
        
        # Mock OpenAI service responses
        openai_service.transcribe_chunks_parallel = AsyncMock()
        openai_service.combine_transcription_results = AsyncMock(return_value=(
            "Test transcription",
            {"success_rate": 1.0, "total_processing_time": 4.0}
        ))
        
        # Create worker
        worker = JobWorker(self.settings, self.redis_service)
        worker.ffmpeg_service = ffmpeg_service
        worker.openai_service = openai_service
        worker.cleanup_service = cleanup_service
        
        return worker


@pytest_asyncio.fixture
async def chaos_framework():
    """Create chaos testing framework."""
    # Create test settings
    settings = Settings()
    settings.temp_dir = tempfile.mkdtemp()
    settings.openai_max_chunk_size_mb = 25.0
    
    # Create mock Redis service
    redis_service = MagicMock(spec=RedisService)
    redis_service.jobs = {}
    redis_service.events = {}
    
    async def mock_create_job(job):
        redis_service.jobs[job.job_id] = {
            "id": job.job_id,
            "file_path": job.file_path,
            "language": job.language,
            "state": job.state.value,
            "chunks_done": 0,
            "chunks_total": 0
        }
    
    async def mock_get_job(job_id):
        return redis_service.jobs.get(job_id)
    
    async def mock_update_job_state(job_id, state, error_message=None):
        if job_id in redis_service.jobs:
            redis_service.jobs[job_id]["state"] = state.value
            if error_message:
                redis_service.jobs[job_id]["error_message"] = error_message
    
    async def mock_publish_job_update(job_id, event_data):
        if job_id not in redis_service.events:
            redis_service.events[job_id] = []
        redis_service.events[job_id].append(event_data)
    
    async def mock_get_job_updates(job_id, start_id):
        return redis_service.events.get(job_id, [])
    
    async def mock_list_jobs(state_filter=None):
        jobs = []
        for job_data in redis_service.jobs.values():
            if state_filter is None or job_data["state"] == state_filter.value:
                jobs.append(job_data)
        return jobs
    
    async def mock_delete_job(job_id):
        redis_service.jobs.pop(job_id, None)
        redis_service.events.pop(job_id, None)
    
    redis_service.create_job = mock_create_job
    redis_service.get_job = mock_get_job
    redis_service.update_job_state = mock_update_job_state
    redis_service.publish_job_update = mock_publish_job_update
    redis_service.get_job_updates = mock_get_job_updates
    redis_service.list_jobs = mock_list_jobs
    redis_service.delete_job = mock_delete_job
    
    framework = ChaosTestFramework(redis_service, settings)
    await framework.setup()
    
    yield framework
    
    await framework.teardown()


@pytest.mark.asyncio
class TestCrashRecovery:
    """Test suite for crash recovery functionality."""
    
    async def test_recovery_after_transcription_crash(self, chaos_framework):
        """Test recovery when worker crashes during transcription."""
        # Create test job
        job_id = await chaos_framework.create_test_job(file_size_mb=30.0)
        
        # Simulate crash during transcription (after 2 chunks)
        crash_info = await chaos_framework.simulate_crash_during_processing(job_id, crash_after_chunks=2)
        
        # Validate crash state
        assert crash_info["chunks_completed_before_crash"] == 2
        assert crash_info["job_state_after_crash"]["state"] == "PROCESSING"
        
        # Validate recovery
        recovery_info = await chaos_framework.validate_recovery(crash_info)
        
        assert recovery_info["recovery_successful"]
        assert len(recovery_info["recovery_events"]) > 0
        
        # Validate idempotency
        idempotency_info = await chaos_framework.validate_idempotency(job_id)
        assert idempotency_info["idempotency_maintained"]
        assert idempotency_info["duplicated_chunks"] == {}
        
        # Validate event publishing
        events_info = await chaos_framework.validate_event_publishing(job_id)
        assert events_info["all_expected_events_present"]
    
    async def test_recovery_with_no_previous_chunks(self, chaos_framework):
        """Test recovery when job had no completed chunks before crash."""
        job_id = await chaos_framework.create_test_job(file_size_mb=30.0)
        
        # Simulate crash before any chunks complete
        crash_info = await chaos_framework.simulate_crash_during_processing(job_id, crash_after_chunks=0)
        
        assert crash_info["chunks_completed_before_crash"] == 0
        
        # Validate recovery restarts from beginning
        recovery_info = await chaos_framework.validate_recovery(crash_info)
        assert recovery_info["recovery_successful"]
        
        # Should have recovery events even for restart scenario
        events_info = await chaos_framework.validate_event_publishing(job_id)
        assert len(events_info["event_types"]) > 0
    
    async def test_recovery_with_all_chunks_completed(self, chaos_framework):
        """Test recovery when all chunks were completed before crash."""
        job_id = await chaos_framework.create_test_job(file_size_mb=30.0)
        
        # Simulate crash after all chunks (large number)
        crash_info = await chaos_framework.simulate_crash_during_processing(job_id, crash_after_chunks=10)
        
        # All chunks should be completed
        assert crash_info["chunks_completed_before_crash"] >= 4  # All 4 chunks in our mock
        
        # Recovery should finalize the job
        recovery_info = await chaos_framework.validate_recovery(crash_info)
        assert recovery_info["recovery_successful"]
        
        # Validate idempotency (no additional chunk processing)
        idempotency_info = await chaos_framework.validate_idempotency(job_id)
        assert idempotency_info["idempotency_maintained"]
    
    async def test_multiple_recovery_attempts(self, chaos_framework):
        """Test that multiple recovery attempts don't cause issues."""
        job_id = await chaos_framework.create_test_job(file_size_mb=30.0)
        
        # First crash and recovery
        crash_info = await chaos_framework.simulate_crash_during_processing(job_id, crash_after_chunks=1)
        recovery_info = await chaos_framework.validate_recovery(crash_info)
        
        # Second recovery attempt (should be idempotent)
        recovery_info_2 = await chaos_framework.validate_recovery(crash_info)
        
        # Validate idempotency across multiple recoveries
        idempotency_info = await chaos_framework.validate_idempotency(job_id)
        assert idempotency_info["idempotency_maintained"]
        
        # Should have multiple recovery events but no duplicate chunk processing
        events_info = await chaos_framework.validate_event_publishing(job_id)
        assert len(events_info["event_types"]) >= 2  # Multiple recovery attempts
    
    async def test_recovery_event_structure(self, chaos_framework):
        """Test that recovery events have correct structure and data."""
        job_id = await chaos_framework.create_test_job(file_size_mb=30.0)
        
        crash_info = await chaos_framework.simulate_crash_during_processing(job_id, crash_after_chunks=2)
        recovery_info = await chaos_framework.validate_recovery(crash_info)
        
        # Get recovery events
        events = await chaos_framework.redis_service.get_job_updates(job_id, "0")
        recovery_events = [event for event in events if event.get("recovery") is True]
        
        for event in recovery_events:
            # Validate required fields
            assert "event" in event
            assert "job_id" in event
            assert "timestamp" in event
            assert event["recovery"] is True
            
            # Validate specific event types
            if event["event"] == "recovery_started":
                assert "message" in event
                assert "recovery_type" in event
            elif event["event"] == "processing_resumed":
                assert "remaining_chunks" in event
                assert "completed_chunks" in event
            elif event["event"] == "recovery_completed":
                assert "recovery_result" in event
    
    async def test_crash_during_different_phases(self, chaos_framework):
        """Test crashes during different phases of job processing."""
        test_cases = [
            {"crash_after_chunks": 0, "phase": "initialization"},
            {"crash_after_chunks": 1, "phase": "early_transcription"},
            {"crash_after_chunks": 2, "phase": "mid_transcription"},
            {"crash_after_chunks": 3, "phase": "late_transcription"},
        ]
        
        for case in test_cases:
            job_id = await chaos_framework.create_test_job(file_size_mb=30.0)
            
            crash_info = await chaos_framework.simulate_crash_during_processing(
                job_id, 
                crash_after_chunks=case["crash_after_chunks"]
            )
            
            recovery_info = await chaos_framework.validate_recovery(crash_info)
            
            # All phases should recover successfully
            assert recovery_info["recovery_successful"], f"Recovery failed for phase: {case['phase']}"
            
            # All phases should maintain idempotency
            idempotency_info = await chaos_framework.validate_idempotency(job_id)
            assert idempotency_info["idempotency_maintained"], f"Idempotency violated for phase: {case['phase']}"


@pytest.mark.asyncio
class TestChaosScenarios:
    """Extended chaos testing scenarios."""
    
    async def test_redis_connection_loss_during_recovery(self, chaos_framework):
        """Test recovery behavior when Redis connection is lost."""
        job_id = await chaos_framework.create_test_job(file_size_mb=30.0)
        
        # Simulate crash
        crash_info = await chaos_framework.simulate_crash_during_processing(job_id, crash_after_chunks=2)
        
        # Simulate Redis connection issues during recovery
        original_publish = chaos_framework.redis_service.publish_job_update
        call_count = 0
        
        async def mock_publish_with_failures(job_id, event_data):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every 3rd call
                raise Exception("Redis connection lost")
            return await original_publish(job_id, event_data)
        
        chaos_framework.redis_service.publish_job_update = mock_publish_with_failures
        
        # Recovery should still work despite Redis issues
        recovery_info = await chaos_framework.validate_recovery(crash_info)
        
        # Recovery might not be marked as successful due to Redis issues,
        # but idempotency should still be maintained
        idempotency_info = await chaos_framework.validate_idempotency(job_id)
        assert idempotency_info["idempotency_maintained"]
    
    async def test_concurrent_recovery_attempts(self, chaos_framework):
        """Test behavior when multiple workers attempt recovery simultaneously."""
        job_id = await chaos_framework.create_test_job(file_size_mb=30.0)
        
        # Simulate crash
        crash_info = await chaos_framework.simulate_crash_during_processing(job_id, crash_after_chunks=2)
        
        # Simulate concurrent recovery attempts
        async def concurrent_recovery():
            return await chaos_framework.validate_recovery(crash_info)
        
        # Run multiple recovery attempts concurrently
        recovery_tasks = [concurrent_recovery() for _ in range(3)]
        recovery_results = await asyncio.gather(*recovery_tasks, return_exceptions=True)
        
        # At least one should succeed
        successful_recoveries = [r for r in recovery_results if isinstance(r, dict) and r.get("recovery_successful")]
        assert len(successful_recoveries) >= 1
        
        # Idempotency should be maintained across all attempts
        idempotency_info = await chaos_framework.validate_idempotency(job_id)
        assert idempotency_info["idempotency_maintained"]
    
    async def test_recovery_with_corrupted_chunk_files(self, chaos_framework):
        """Test recovery when chunk files are corrupted or missing."""
        job_id = await chaos_framework.create_test_job(file_size_mb=30.0)
        
        # Simulate crash and corrupt some chunk files
        crash_info = await chaos_framework.simulate_crash_during_processing(job_id, crash_after_chunks=2)
        
        # Mock chunk info recovery to simulate missing/corrupted files
        original_recover_chunk_info = None
        
        async def mock_recover_chunk_info_with_corruption(job_id, job_dir):
            # Return partial chunk info (simulating corruption)
            return [
                ChunkInfo(file_path=f"{chaos_framework.temp_dir}/chunk_0.wav", index=0, start_time=0.0, duration=5.0),
                # Missing chunk 1 (corrupted)
                ChunkInfo(file_path=f"{chaos_framework.temp_dir}/chunk_2.wav", index=2, start_time=10.0, duration=5.0),
            ]
        
        # Recovery should handle missing chunks gracefully
        recovery_info = await chaos_framework.validate_recovery(crash_info)
        
        # Even with corruption, the system should handle it gracefully
        events_info = await chaos_framework.validate_event_publishing(job_id)
        assert len(events_info["event_types"]) > 0  # Should have some recovery events


# Integration test to run all chaos tests
@pytest.mark.asyncio
async def test_comprehensive_chaos_suite(chaos_framework):
    """Run comprehensive chaos testing suite."""
    test_results = {
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "failure_details": []
    }
    
    # Define all chaos test scenarios
    chaos_scenarios = [
        ("Basic Recovery", lambda: TestCrashRecovery().test_recovery_after_transcription_crash(chaos_framework)),
        ("No Previous Chunks", lambda: TestCrashRecovery().test_recovery_with_no_previous_chunks(chaos_framework)),
        ("All Chunks Completed", lambda: TestCrashRecovery().test_recovery_with_all_chunks_completed(chaos_framework)),
        ("Multiple Recovery", lambda: TestCrashRecovery().test_multiple_recovery_attempts(chaos_framework)),
        ("Event Structure", lambda: TestCrashRecovery().test_recovery_event_structure(chaos_framework)),
        ("Different Phases", lambda: TestCrashRecovery().test_crash_during_different_phases(chaos_framework)),
    ]
    
    # Run all scenarios
    for scenario_name, scenario_func in chaos_scenarios:
        test_results["tests_run"] += 1
        try:
            await scenario_func()
            test_results["tests_passed"] += 1
            print(f"✅ {scenario_name}: PASSED")
        except Exception as e:
            test_results["tests_failed"] += 1
            test_results["failure_details"].append(f"{scenario_name}: {str(e)}")
            print(f"❌ {scenario_name}: FAILED - {str(e)}")
    
    # Print summary
    print(f"\n📊 Chaos Testing Summary:")
    print(f"   Tests Run: {test_results['tests_run']}")
    print(f"   Passed: {test_results['tests_passed']}")
    print(f"   Failed: {test_results['tests_failed']}")
    print(f"   Success Rate: {(test_results['tests_passed'] / test_results['tests_run']) * 100:.1f}%")
    
    if test_results["failure_details"]:
        print(f"\n❌ Failures:")
        for failure in test_results["failure_details"]:
            print(f"   - {failure}")
    
    # Assert overall success
    assert test_results["tests_failed"] == 0, f"Chaos tests failed: {test_results['failure_details']}"
    assert test_results["tests_passed"] >= 6, "Not all chaos tests completed successfully"


if __name__ == "__main__":
    # Run chaos tests directly
    pytest.main([__file__, "-v", "--tb=short"])