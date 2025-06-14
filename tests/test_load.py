"""Load tests for concurrent job processing and performance monitoring."""

import asyncio
import psutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from unittest.mock import AsyncMock, Mock, patch

import pytest
from httpx import AsyncClient

from media_to_text.main import app
from media_to_text.services.ffmpeg_service import ChunkInfo, MediaInfo
from media_to_text.services.openai_service import TranscriptionResult


class TestLoadPerformance:
    """Load tests for system performance under stress."""

    @pytest.fixture
    def async_client(self):
        """Create async test client."""
        return AsyncClient(app=app, base_url="http://test")

    @pytest.fixture
    def sample_audio_files(self):
        """Create multiple sample audio files for load testing."""
        files = []
        for i in range(10):  # Create 10 different file types
            # Varying file sizes and types
            if i % 3 == 0:
                # Small WAV file
                wav_header = b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x02\x00\x44\xac\x00\x00\x10\xb1\x02\x00\x04\x00\x10\x00data\x00\x08\x00\x00'
                audio_data = b'\x00\x01' * (200000 + i * 50000)  # Varying sizes
                files.append(("small", BytesIO(wav_header + audio_data)))
            elif i % 3 == 1:
                # Medium MP3 file
                mp3_header = b'\xff\xfb\x90\x00'
                audio_data = b'\x00\x01' * (500000 + i * 100000)
                files.append(("medium", BytesIO(mp3_header + audio_data)))
            else:
                # Large file
                wav_header = b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x02\x00\x44\xac\x00\x00\x10\xb1\x02\x00\x04\x00\x10\x00data\x00\x08\x00\x00'
                audio_data = b'\x00\x01' * (1000000 + i * 200000)
                files.append(("large", BytesIO(wav_header + audio_data)))
        return files

    @pytest.fixture
    def mock_services_for_load_test(self):
        """Mock services optimized for load testing."""
        with patch('media_to_text.services.redis_service.get_redis_service') as mock_redis, \
             patch('media_to_text.services.ffmpeg_service.get_ffmpeg_service') as mock_ffmpeg, \
             patch('media_to_text.services.openai_service.get_openai_service') as mock_openai:
            
            # Redis service mock
            redis_instance = AsyncMock()
            redis_instance.create_job.return_value = True
            redis_instance.get_job.return_value = {"id": "load-test-job", "state": "QUEUED"}
            redis_instance.update_job_state.return_value = True
            redis_instance.update_job_progress.return_value = True
            redis_instance.publish_job_update.return_value = True
            redis_instance.health_check.return_value = True
            redis_instance.list_jobs.return_value = []
            mock_redis.return_value = redis_instance
            
            # FFmpeg service mock with realistic processing times
            ffmpeg_instance = Mock()
            def probe_media_side_effect(file_path):
                # Simulate varying processing based on file type
                if "small" in file_path:
                    return MediaInfo(has_audio=True, duration=30.0, format="wav", needs_conversion=False)
                elif "medium" in file_path:
                    return MediaInfo(has_audio=True, duration=120.0, format="mp3", needs_conversion=True)
                else:
                    return MediaInfo(has_audio=True, duration=300.0, format="wav", needs_conversion=False)
            
            ffmpeg_instance.probe_media.side_effect = probe_media_side_effect
            ffmpeg_instance.get_file_size_mb.return_value = 15.0  # Moderate size
            ffmpeg_instance.chunk_wav_with_segments.return_value = [
                ChunkInfo("/tmp/chunk_0.wav", 0, 0.0, 30.0, 15 * 1024 * 1024)
            ]
            ffmpeg_instance.convert_to_wav.return_value = None
            mock_ffmpeg.return_value = ffmpeg_instance
            
            # OpenAI service mock with realistic response times
            openai_instance = AsyncMock()
            async def transcribe_side_effect(chunks, language, job_id):
                # Simulate OpenAI processing time
                await asyncio.sleep(0.1)  # Small delay to simulate API call
                return [
                    TranscriptionResult(
                        chunk_index=i,
                        success=True,
                        text=f"Load test transcription for chunk {i}",
                        processing_time=2.0,
                        retry_count=0,
                        segments=[{"start": 0.0, "end": 30.0, "text": f"Load test transcription for chunk {i}"}]
                    )
                    for i, chunk in enumerate(chunks)
                ]
            
            openai_instance.transcribe_chunks_parallel.side_effect = transcribe_side_effect
            mock_openai.return_value = openai_instance
            
            yield {
                "redis": redis_instance,
                "ffmpeg": ffmpeg_instance,
                "openai": openai_instance
            }

    class PerformanceMonitor:
        """Monitor system performance during load tests."""
        
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.memory_samples = []
            self.cpu_samples = []
            self.response_times = []
            self.monitoring = False
            
        def start_monitoring(self):
            """Start performance monitoring."""
            self.start_time = time.time()
            self.monitoring = True
            self.memory_samples = []
            self.cpu_samples = []
            self.response_times = []
            
        def record_response_time(self, response_time):
            """Record API response time."""
            self.response_times.append(response_time)
            
        def sample_system_metrics(self):
            """Sample current system metrics."""
            if self.monitoring:
                process = psutil.Process()
                self.memory_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
                self.cpu_samples.append(process.cpu_percent())
                
        def stop_monitoring(self):
            """Stop monitoring and return results."""
            self.end_time = time.time()
            self.monitoring = False
            
            return {
                "duration": self.end_time - self.start_time,
                "memory_stats": {
                    "max_mb": max(self.memory_samples) if self.memory_samples else 0,
                    "min_mb": min(self.memory_samples) if self.memory_samples else 0,
                    "avg_mb": statistics.mean(self.memory_samples) if self.memory_samples else 0,
                    "samples": len(self.memory_samples)
                },
                "cpu_stats": {
                    "max_percent": max(self.cpu_samples) if self.cpu_samples else 0,
                    "avg_percent": statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
                    "samples": len(self.cpu_samples)
                },
                "response_time_stats": {
                    "max_ms": max(self.response_times) * 1000 if self.response_times else 0,
                    "min_ms": min(self.response_times) * 1000 if self.response_times else 0,
                    "avg_ms": statistics.mean(self.response_times) * 1000 if self.response_times else 0,
                    "p95_ms": statistics.quantiles(self.response_times, n=20)[18] * 1000 if len(self.response_times) > 20 else 0,
                    "total_requests": len(self.response_times)
                }
            }

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.load
    async def test_concurrent_job_creation_100_jobs(self, async_client, sample_audio_files, mock_services_for_load_test):
        """Test creating 100 concurrent transcription jobs."""
        monitor = self.PerformanceMonitor()
        monitor.start_monitoring()
        
        # Monitor system metrics in background
        async def monitor_system():
            while monitor.monitoring:
                monitor.sample_system_metrics()
                await asyncio.sleep(0.5)
        
        monitor_task = asyncio.create_task(monitor_system())
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            mock_temp_file.return_value.__enter__.return_value.name = "/tmp/load_test.wav"
            
            with patch('shutil.move'):
                # Create 100 concurrent jobs
                async def create_single_job(client, file_data, job_num):
                    start_time = time.time()
                    try:
                        file_type, file_content = file_data
                        files = {"file": (f"load_test_{job_num}.wav", file_content, "audio/wav")}
                        data = {"language": "en"}
                        
                        response = await client.post("/transcriptions/", files=files, data=data)
                        response_time = time.time() - start_time
                        monitor.record_response_time(response_time)
                        
                        return {
                            "job_num": job_num,
                            "status_code": response.status_code,
                            "response_time": response_time,
                            "job_id": response.json().get("job_id") if response.status_code == 201 else None
                        }
                    except Exception as e:
                        response_time = time.time() - start_time
                        monitor.record_response_time(response_time)
                        return {
                            "job_num": job_num,
                            "status_code": 500,
                            "response_time": response_time,
                            "error": str(e)
                        }
                
                # Create 100 jobs using different file types
                async with async_client as ac:
                    tasks = []
                    for i in range(100):
                        file_data = sample_audio_files[i % len(sample_audio_files)]
                        tasks.append(create_single_job(ac, file_data, i))
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Stop monitoring
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
                
                performance_stats = monitor.stop_monitoring()
                
                # Analyze results
                successful_jobs = [r for r in results if isinstance(r, dict) and r.get("status_code") == 201]
                failed_jobs = [r for r in results if isinstance(r, dict) and r.get("status_code") != 201]
                
                # Performance assertions
                assert len(successful_jobs) >= 95, f"Expected at least 95% success rate, got {len(successful_jobs)}/100"
                assert performance_stats["response_time_stats"]["avg_ms"] < 5000, f"Average response time too high: {performance_stats['response_time_stats']['avg_ms']:.2f}ms"
                assert performance_stats["memory_stats"]["max_mb"] < 2048, f"Memory usage too high: {performance_stats['memory_stats']['max_mb']:.2f}MB"
                
                # Log performance results
                print(f"\n=== Load Test Results (100 concurrent jobs) ===")
                print(f"Success Rate: {len(successful_jobs)}/100 ({len(successful_jobs)}%)")
                print(f"Failed Jobs: {len(failed_jobs)}")
                print(f"Total Duration: {performance_stats['duration']:.2f}s")
                print(f"Memory Usage: {performance_stats['memory_stats']['avg_mb']:.1f}MB avg, {performance_stats['memory_stats']['max_mb']:.1f}MB max")
                print(f"CPU Usage: {performance_stats['cpu_stats']['avg_percent']:.1f}% avg, {performance_stats['cpu_stats']['max_percent']:.1f}% max")
                print(f"Response Times: {performance_stats['response_time_stats']['avg_ms']:.1f}ms avg, {performance_stats['response_time_stats']['p95_ms']:.1f}ms p95")

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.load
    async def test_sustained_load_with_job_status_checks(self, async_client, sample_audio_files, mock_services_for_load_test):
        """Test sustained load with continuous job status checking."""
        monitor = self.PerformanceMonitor()
        monitor.start_monitoring()
        
        job_ids = []
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            mock_temp_file.return_value.__enter__.return_value.name = "/tmp/sustained_load.wav"
            
            with patch('shutil.move'):
                # Phase 1: Create 50 jobs
                async def create_job_batch(client, batch_size, batch_num):
                    batch_job_ids = []
                    for i in range(batch_size):
                        file_data = sample_audio_files[i % len(sample_audio_files)]
                        file_type, file_content = file_data
                        
                        files = {"file": (f"sustained_{batch_num}_{i}.wav", file_content, "audio/wav")}
                        data = {"language": "en"}
                        
                        start_time = time.time()
                        response = await client.post("/transcriptions/", files=files, data=data)
                        monitor.record_response_time(time.time() - start_time)
                        
                        if response.status_code == 201:
                            batch_job_ids.append(response.json()["job_id"])
                    
                    return batch_job_ids
                
                # Phase 2: Continuous status checking while creating more jobs
                async def check_job_statuses(client, job_ids_to_check):
                    for job_id in job_ids_to_check[:20]:  # Check first 20 jobs
                        start_time = time.time()
                        response = await client.get(f"/jobs/{job_id}")
                        monitor.record_response_time(time.time() - start_time)
                        monitor.sample_system_metrics()
                
                async with async_client as ac:
                    # Create initial batch
                    initial_jobs = await create_job_batch(ac, 25, 0)
                    job_ids.extend(initial_jobs)
                    
                    # Concurrent operations: create more jobs while checking status
                    tasks = []
                    
                    # Create more job batches
                    for batch_num in range(1, 4):  # 3 more batches of 25
                        tasks.append(create_job_batch(ac, 25, batch_num))
                    
                    # Status checking tasks
                    for _ in range(5):  # 5 parallel status checking tasks
                        tasks.append(check_job_statuses(ac, job_ids))
                    
                    # Execute all tasks concurrently
                    results = await asyncio.gather(*tasks[:3])  # Job creation tasks
                    for batch_job_ids in results:
                        job_ids.extend(batch_job_ids)
                    
                    # Final status check round
                    await asyncio.gather(*tasks[3:])  # Status check tasks
                
                performance_stats = monitor.stop_monitoring()
                
                # Performance assertions for sustained load
                assert len(job_ids) >= 90, f"Expected at least 90 jobs created, got {len(job_ids)}"
                assert performance_stats["response_time_stats"]["p95_ms"] < 10000, f"P95 response time too high: {performance_stats['response_time_stats']['p95_ms']:.2f}ms"
                assert performance_stats["memory_stats"]["max_mb"] < 1536, f"Memory usage too high during sustained load: {performance_stats['memory_stats']['max_mb']:.2f}MB"
                
                print(f"\n=== Sustained Load Test Results ===")
                print(f"Jobs Created: {len(job_ids)}")
                print(f"Total Requests: {performance_stats['response_time_stats']['total_requests']}")
                print(f"Duration: {performance_stats['duration']:.2f}s")
                print(f"Throughput: {performance_stats['response_time_stats']['total_requests'] / performance_stats['duration']:.1f} req/s")

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.load
    async def test_memory_leak_detection(self, async_client, sample_audio_files, mock_services_for_load_test):
        """Test for memory leaks during continuous operation."""
        monitor = self.PerformanceMonitor()
        monitor.start_monitoring()
        
        initial_memory = None
        memory_readings = []
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            mock_temp_file.return_value.__enter__.return_value.name = "/tmp/memory_test.wav"
            
            with patch('shutil.move'):
                # Run multiple rounds of job creation and deletion
                for round_num in range(10):  # 10 rounds
                    round_job_ids = []
                    
                    # Create 10 jobs per round
                    async with async_client as ac:
                        for i in range(10):
                            file_data = sample_audio_files[i % len(sample_audio_files)]
                            file_type, file_content = file_data
                            
                            files = {"file": (f"memory_test_{round_num}_{i}.wav", file_content, "audio/wav")}
                            data = {"language": "en"}
                            
                            response = await ac.post("/transcriptions/", files=files, data=data)
                            if response.status_code == 201:
                                round_job_ids.append(response.json()["job_id"])
                    
                    # Sample memory after job creation
                    monitor.sample_system_metrics()
                    current_memory = monitor.memory_samples[-1]
                    memory_readings.append(current_memory)
                    
                    if initial_memory is None:
                        initial_memory = current_memory
                    
                    # Simulate job deletion/cleanup
                    mock_services_for_load_test["redis"].delete_job.return_value = True
                    
                    # Check a few job statuses
                    async with async_client as ac:
                        for job_id in round_job_ids[:3]:
                            await ac.get(f"/jobs/{job_id}")
                    
                    # Allow some time for garbage collection
                    await asyncio.sleep(0.1)
                
                performance_stats = monitor.stop_monitoring()
                
                # Memory leak detection
                final_memory = memory_readings[-1]
                memory_growth = final_memory - initial_memory
                memory_growth_percent = (memory_growth / initial_memory) * 100
                
                # Check for excessive memory growth
                assert memory_growth_percent < 50, f"Potential memory leak detected: {memory_growth_percent:.1f}% growth"
                assert final_memory < 1024, f"Final memory usage too high: {final_memory:.1f}MB"
                
                print(f"\n=== Memory Leak Test Results ===")
                print(f"Initial Memory: {initial_memory:.1f}MB")
                print(f"Final Memory: {final_memory:.1f}MB")
                print(f"Memory Growth: {memory_growth:.1f}MB ({memory_growth_percent:.1f}%)")
                print(f"Max Memory: {performance_stats['memory_stats']['max_mb']:.1f}MB")

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.load
    async def test_redis_connection_pool_under_load(self, async_client, sample_audio_files, mock_services_for_load_test):
        """Test Redis connection pool performance under high load."""
        monitor = self.PerformanceMonitor()
        monitor.start_monitoring()
        
        # Track Redis operation times
        redis_operation_times = []
        
        def mock_redis_with_timing(original_method):
            async def timed_method(*args, **kwargs):
                start_time = time.time()
                result = await original_method(*args, **kwargs)
                redis_operation_times.append(time.time() - start_time)
                return result
            return timed_method
        
        # Wrap Redis methods with timing
        redis_instance = mock_services_for_load_test["redis"]
        original_create_job = redis_instance.create_job
        original_get_job = redis_instance.get_job
        
        redis_instance.create_job = mock_redis_with_timing(original_create_job)
        redis_instance.get_job = mock_redis_with_timing(original_get_job)
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            mock_temp_file.return_value.__enter__.return_value.name = "/tmp/redis_load.wav"
            
            with patch('shutil.move'):
                # High-frequency Redis operations
                async def redis_intensive_operations(client, operation_count):
                    job_ids = []
                    
                    # Create jobs (write operations)
                    for i in range(operation_count):
                        file_data = sample_audio_files[i % len(sample_audio_files)]
                        file_type, file_content = file_data
                        
                        files = {"file": (f"redis_test_{i}.wav", file_content, "audio/wav")}
                        data = {"language": "en"}
                        
                        response = await client.post("/transcriptions/", files=files, data=data)
                        if response.status_code == 201:
                            job_ids.append(response.json()["job_id"])
                        
                        monitor.sample_system_metrics()
                    
                    # Read operations
                    for job_id in job_ids:
                        await client.get(f"/jobs/{job_id}")
                    
                    return len(job_ids)
                
                async with async_client as ac:
                    # Run multiple concurrent Redis-intensive operations
                    tasks = [
                        redis_intensive_operations(ac, 20) 
                        for _ in range(5)  # 5 concurrent workers
                    ]
                    
                    results = await asyncio.gather(*tasks)
                
                performance_stats = monitor.stop_monitoring()
                total_jobs_created = sum(results)
                
                # Redis performance analysis
                if redis_operation_times:
                    redis_avg_time = statistics.mean(redis_operation_times) * 1000  # ms
                    redis_max_time = max(redis_operation_times) * 1000  # ms
                else:
                    redis_avg_time = redis_max_time = 0
                
                # Performance assertions
                assert total_jobs_created >= 90, f"Expected at least 90 jobs, got {total_jobs_created}"
                assert redis_avg_time < 100, f"Average Redis operation time too high: {redis_avg_time:.2f}ms"
                assert redis_max_time < 1000, f"Max Redis operation time too high: {redis_max_time:.2f}ms"
                
                print(f"\n=== Redis Load Test Results ===")
                print(f"Total Jobs Created: {total_jobs_created}")
                print(f"Redis Operations: {len(redis_operation_times)}")
                print(f"Avg Redis Time: {redis_avg_time:.2f}ms")
                print(f"Max Redis Time: {redis_max_time:.2f}ms")
                print(f"Redis Throughput: {len(redis_operation_times) / performance_stats['duration']:.1f} ops/s")

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.load
    async def test_api_rate_limiting_under_load(self, async_client, sample_audio_files, mock_services_for_load_test):
        """Test API behavior under rate limiting conditions."""
        monitor = self.PerformanceMonitor()
        monitor.start_monitoring()
        
        status_code_counts = {}
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            mock_temp_file.return_value.__enter__.return_value.name = "/tmp/rate_limit_test.wav"
            
            with patch('shutil.move'):
                # Rapid-fire requests to test rate limiting
                async def rapid_requests(client, request_count):
                    results = []
                    for i in range(request_count):
                        file_data = sample_audio_files[0]  # Use same file type
                        file_type, file_content = file_data
                        
                        files = {"file": (f"rate_test_{i}.wav", file_content, "audio/wav")}
                        data = {"language": "en"}
                        
                        start_time = time.time()
                        try:
                            response = await client.post("/transcriptions/", files=files, data=data)
                            response_time = time.time() - start_time
                            
                            status_code = response.status_code
                            status_code_counts[status_code] = status_code_counts.get(status_code, 0) + 1
                            
                            results.append({
                                "status_code": status_code,
                                "response_time": response_time
                            })
                            
                            monitor.record_response_time(response_time)
                        except Exception as e:
                            response_time = time.time() - start_time
                            monitor.record_response_time(response_time)
                            results.append({
                                "status_code": 500,
                                "response_time": response_time,
                                "error": str(e)
                            })
                    
                    return results
                
                # Multiple clients making rapid requests
                async with async_client as ac:
                    tasks = [
                        rapid_requests(ac, 25) 
                        for _ in range(4)  # 4 clients, 25 requests each
                    ]
                    
                    all_results = await asyncio.gather(*tasks)
                
                performance_stats = monitor.stop_monitoring()
                
                # Flatten results
                flat_results = [item for sublist in all_results for item in sublist]
                
                # Analyze rate limiting behavior
                total_requests = len(flat_results)
                successful_requests = len([r for r in flat_results if r["status_code"] == 201])
                success_rate = (successful_requests / total_requests) * 100
                
                # Performance assertions
                assert success_rate >= 80, f"Success rate too low under load: {success_rate:.1f}%"
                assert performance_stats["response_time_stats"]["avg_ms"] < 10000, f"Average response time too high: {performance_stats['response_time_stats']['avg_ms']:.2f}ms"
                
                print(f"\n=== Rate Limiting Test Results ===")
                print(f"Total Requests: {total_requests}")
                print(f"Success Rate: {success_rate:.1f}%")
                print(f"Status Code Distribution: {status_code_counts}")
                print(f"Avg Response Time: {performance_stats['response_time_stats']['avg_ms']:.1f}ms")
                print(f"Request Rate: {total_requests / performance_stats['duration']:.1f} req/s")