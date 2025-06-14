"""Job worker for processing transcription jobs."""

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from media_to_text.config import Settings
from media_to_text.models import JobMetadata, JobState
from media_to_text.services.redis_service import RedisService
from media_to_text.services.ffmpeg_service import FFmpegService, get_ffmpeg_service, ChunkInfo
from media_to_text.services.openai_service import get_openai_service, OpenAITranscriptionService
from media_to_text.services.transcript_service import get_transcript_processor, TranscriptProcessor
from media_to_text.services.cleanup_service import get_cleanup_service, CleanupService


class JobWorker:
    """Worker for processing transcription jobs."""
    
    def __init__(self, settings: Settings, redis_service: RedisService):
        self.settings = settings
        self.redis_service = redis_service
        self.ffmpeg_service = get_ffmpeg_service()
        self.openai_service = get_openai_service()
        self.transcript_processor = get_transcript_processor()
        self.cleanup_service = None  # Will be set during initialization
        self.running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the job worker."""
        if self.running:
            return
        
        self.running = True
        self._task = asyncio.create_task(self._worker_loop())
        print("🚀 Job worker started")
    
    async def stop(self) -> None:
        """Stop the job worker."""
        if not self.running:
            return
        
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        print("🛑 Job worker stopped")
    
    async def set_cleanup_service(self, cleanup_service: CleanupService) -> None:
        """Set the cleanup service for the job worker."""
        self.cleanup_service = cleanup_service
    
    async def _worker_loop(self) -> None:
        """Main worker loop to process jobs."""
        while self.running:
            try:
                # Get queued jobs
                queued_jobs = await self.redis_service.list_jobs(state=JobState.QUEUED)
                
                if queued_jobs:
                    # Process the oldest job
                    job = queued_jobs[0]
                    await self._process_job(job)
                else:
                    # No jobs available, wait a bit
                    await asyncio.sleep(1.0)
                    
            except Exception as e:
                print(f"❌ Error in worker loop: {e}")
                await asyncio.sleep(5.0)  # Wait before retrying
    
    async def _process_job(self, job: JobMetadata) -> None:
        """Process a single transcription job."""
        job_dir = None
        chunk_infos = []
        
        try:
            print(f"📝 Processing job {job.job_id}: {job.file_path}")
            
            # Update job state to PROCESSING
            await self.redis_service.update_job_state(job.job_id, JobState.PROCESSING)
            
            # Create temporary directory for this job
            job_dir = os.path.join(self.settings.temp_dir, f"job_{job.job_id}")
            os.makedirs(job_dir, exist_ok=True)
            
            # Step 1: Probe media file
            print(f"🔍 Probing media file: {job.file_path}")
            media_info = await self.ffmpeg_service.probe_media(job.file_path)
            
            if not media_info.has_audio:
                raise RuntimeError("Input file has no audio streams")
            
            # Step 2: Determine if conversion is needed
            converted_file = job.file_path
            
            if media_info.needs_conversion:
                print(f"🔄 Converting {job.file_path} to WAV format")
                converted_file = os.path.join(job_dir, "converted.wav")
                
                await self.ffmpeg_service.convert_to_wav(job.file_path, converted_file)
                print(f"✅ Conversion complete: {converted_file}")
            else:
                print(f"✅ File is already in WAV format, no conversion needed")
            
            # Step 3: Check file size and determine chunking strategy
            file_size_mb = await self.ffmpeg_service.get_file_size_mb(converted_file)
            max_chunk_size_mb = self.settings.openai_max_chunk_size_mb
            
            if file_size_mb > max_chunk_size_mb:
                print(f"📊 File size ({file_size_mb:.1f}MB) exceeds limit ({max_chunk_size_mb}MB), chunking with segments...")
                
                # Use enhanced segment-based chunking
                chunks_dir = os.path.join(job_dir, "chunks")
                chunk_infos = await self.ffmpeg_service.chunk_wav_with_segments(
                    converted_file, 
                    chunks_dir, 
                    max_chunk_size_mb
                )
                
                print(f"✅ Created {len(chunk_infos)} chunks using FFmpeg segments")
                
                # Validate chunk sizes
                is_valid = await self.ffmpeg_service.validate_chunk_sizes(chunk_infos, max_chunk_size_mb)
                if not is_valid:
                    print("⚠️  Some chunks exceed size limit, but proceeding...")
                
                # Log chunk details
                for chunk in chunk_infos:
                    print(f"   📦 Chunk {chunk.index}: {chunk.size_mb:.1f}MB, {chunk.duration:.1f}s - {chunk.file_path}")
                
            else:
                print(f"✅ File size ({file_size_mb:.1f}MB) is within limit, processing as single file")
                
                # Create a single chunk info for the entire file
                chunk_infos = [ChunkInfo(
                    file_path=converted_file,
                    index=0,
                    start_time=0.0,
                    duration=media_info.duration
                )]
                chunk_infos[0].size_bytes = int(file_size_mb * 1024 * 1024)
            
            # Step 4: Store chunk information in Redis for tracking
            chunk_data = [chunk.to_dict() for chunk in chunk_infos]
            await self.redis_service.publish_job_update(job.job_id, {
                "event": "chunks_created",
                "chunk_count": len(chunk_infos),
                "chunks": chunk_data
            })
            
            # Step 5: Update job with total chunks count
            await self.redis_service.update_job_progress(
                job.job_id, 
                chunks_done=0, 
                chunks_total=len(chunk_infos)
            )
            
            # Step 6: Perform parallel transcription with OpenAI
            print(f"🎯 Starting OpenAI transcription for {len(chunk_infos)} chunks")
            
            await self.redis_service.publish_job_update(job.job_id, {
                "event": "transcription_started",
                "message": f"Starting OpenAI transcription for {len(chunk_infos)} chunks"
            })
            
            # Use OpenAI service for parallel transcription
            transcription_results = await self.openai_service.transcribe_chunks_parallel(
                chunks=chunk_infos,
                language=job.language,
                job_id=job.job_id
            )
            
            # Step 7: Process results and update progress
            successful_results = []
            failed_results = []
            
            for result in transcription_results:
                if result.success:
                    successful_results.append(result)
                    
                    # Store individual chunk result in Redis
                    await self.redis_service.publish_job_update(job.job_id, {
                        "event": "chunk_transcribed",
                        "chunk_index": result.chunk_index,
                        "chunk_text": result.text,
                        "processing_time": result.processing_time,
                        "retry_count": result.retry_count
                    })
                    
                else:
                    failed_results.append(result)
                    
                    # Log failed chunk
                    await self.redis_service.publish_job_update(job.job_id, {
                        "event": "chunk_failed",
                        "chunk_index": result.chunk_index,
                        "error_message": result.error_message,
                        "retry_count": result.retry_count
                    })
                
                # Update progress after each chunk
                await self.redis_service.update_job_progress(
                    job.job_id,
                    chunks_done=len(successful_results) + len(failed_results),
                    chunks_total=len(chunk_infos)
                )
            
            # Step 8: Enhanced transcript processing with timestamp normalization
            if successful_results:
                # Get basic metadata from OpenAI service
                combined_text, basic_metadata = await self.openai_service.combine_transcription_results(transcription_results)
                
                # Prepare transcript file path
                transcript_file_path = os.path.join(job_dir, "transcript.json")
                
                # Use enhanced transcript processor for normalization and formatting
                print("🔄 Processing transcript with timestamp normalization...")
                
                await self.redis_service.publish_job_update(job.job_id, {
                    "event": "transcript_processing_started",
                    "message": "Normalizing timestamps and aggregating transcript"
                })
                
                # Process complete transcript with normalization
                formatted_transcript = self.transcript_processor.process_complete_transcript(
                    transcription_results=transcription_results,
                    chunk_infos=chunk_infos,
                    original_metadata=basic_metadata,
                    output_path=transcript_file_path
                )
                
                # Enhanced final result with normalized transcript
                final_result = {
                    "transcript": formatted_transcript["text"],
                    "normalized_transcript": formatted_transcript,
                    "job_id": job.job_id,
                    "original_file": job.file_path,
                    "language": job.language,
                    "transcript_file": transcript_file_path,
                    "processing_summary": {
                        "total_chunks": len(chunk_infos),
                        "successful_chunks": len(successful_results),
                        "failed_chunks": len(failed_results),
                        "success_rate": basic_metadata["success_rate"],
                        "total_processing_time": basic_metadata["total_processing_time"],
                        "total_duration": formatted_transcript["metadata"]["total_duration"]
                    }
                }
                
                # Store enhanced result in Redis
                result_key = f"transcript:{job.job_id}"
                await self.redis_service.redis.set(
                    result_key, 
                    json.dumps(final_result),
                    ex=self.redis_service.ttl_seconds
                )
                
                # Publish final result with normalized transcript
                await self.redis_service.publish_job_update(job.job_id, {
                    "event": "transcription_completed",
                    "transcript": formatted_transcript["text"],
                    "normalized_transcript": formatted_transcript,
                    "metadata": formatted_transcript["metadata"],
                    "result_key": result_key,
                    "transcript_file": transcript_file_path
                })
                
                print(f"✅ Enhanced transcript complete: {len(formatted_transcript['text'])} characters")
                print(f"📊 Success rate: {basic_metadata['success_rate']:.1%} ({basic_metadata['successful_chunks']}/{basic_metadata['total_chunks']} chunks)")
                print(f"⏱️  Total duration: {formatted_transcript['metadata']['total_duration']:.1f}s")
                
                # Mark job as completed if we have at least some successful transcriptions
                if basic_metadata["success_rate"] >= 0.5:  # At least 50% success rate
                    await self.redis_service.update_job_state(job.job_id, JobState.COMPLETED)
                    
                    # Schedule enhanced cleanup using cleanup service
                    if self.cleanup_service:
                        await self.cleanup_service.schedule_job_cleanup(job.job_id, delay_seconds=300)
                    
                    print(f"🎉 Job {job.job_id} completed successfully with normalized transcript")
                else:
                    # Low success rate - mark as failed
                    await self.redis_service.update_job_state(
                        job.job_id, 
                        JobState.FAILED, 
                        error_message=f"Low success rate: {basic_metadata['success_rate']:.1%}"
                    )
                    
                    # Trigger immediate cleanup for failed job
                    if self.cleanup_service:
                        await self.cleanup_service.trigger_immediate_cleanup(job.job_id, JobState.FAILED)
                    
                    print(f"❌ Job {job.job_id} failed due to low success rate")
            else:
                # No successful transcriptions
                await self.redis_service.update_job_state(
                    job.job_id, 
                    JobState.FAILED, 
                    error_message="All transcription chunks failed"
                )
                
                # Trigger immediate cleanup for failed job
                if self.cleanup_service:
                    await self.cleanup_service.trigger_immediate_cleanup(job.job_id, JobState.FAILED)
                
                print(f"❌ Job {job.job_id} failed - no successful transcriptions")
            
        except Exception as e:
            print(f"❌ Job {job.job_id} failed: {e}")
            await self.redis_service.update_job_state(
                job.job_id, 
                JobState.FAILED, 
                error_message=str(e)
            )
            
            # Trigger immediate cleanup for failed job
            if self.cleanup_service:
                await self.cleanup_service.trigger_immediate_cleanup(job.job_id, JobState.FAILED)
            
            # Publish error event
            await self.redis_service.publish_job_update(job.job_id, {
                "event": "job_failed",
                "error": str(e)
            })
        
        finally:
            # Note: Enhanced cleanup is now handled by the CleanupService
            # Basic fallback cleanup only if cleanup service is not available
            if job_dir and self.settings.cleanup_temp_files and not self.cleanup_service:
                try:
                    # Keep transcript.json for a while, cleanup other files
                    transcript_file = os.path.join(job_dir, "transcript.json")
                    if os.path.exists(transcript_file):
                        print(f"📄 Preserved transcript file: {transcript_file}")
                    
                    # Remove chunks and converted files, but keep transcript
                    for item in os.listdir(job_dir):
                        item_path = os.path.join(job_dir, item)
                        if item != "transcript.json":
                            if os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                            else:
                                os.remove(item_path)
                    
                    print(f"🧹 Basic cleanup completed for: {job_dir}")
                except Exception as e:
                    print(f"⚠️  Failed basic cleanup {job_dir}: {e}")
    
    async def process_job_by_id(self, job_id: str) -> bool:
        """Process a specific job by ID (for testing/manual processing)."""
        job = await self.redis_service.get_job(job_id)
        if not job:
            print(f"❌ Job {job_id} not found")
            return False
        
        if job.state != JobState.QUEUED:
            print(f"❌ Job {job_id} is not in QUEUED state (current: {job.state})")
            return False
        
        await self._process_job(job)
        return True


# Global job worker instance
job_worker: Optional[JobWorker] = None


async def get_job_worker() -> JobWorker:
    """Get job worker instance."""
    global job_worker
    if job_worker is None:
        raise RuntimeError("Job worker not initialized")
    return job_worker


async def init_job_worker(settings: Settings, redis_service: RedisService) -> JobWorker:
    """Initialize job worker."""
    global job_worker
    job_worker = JobWorker(settings, redis_service)
    await job_worker.start()
    return job_worker


async def close_job_worker() -> None:
    """Close job worker."""
    global job_worker
    if job_worker:
        await job_worker.stop()
        job_worker = None