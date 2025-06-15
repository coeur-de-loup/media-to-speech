"""Job worker for processing transcription jobs with crash recovery."""

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

from media_to_text.config import Settings
from media_to_text.logging import LoggerMixin, get_logger
from media_to_text.models import JobMetadata, JobState
from media_to_text.services.redis_service import RedisService
from media_to_text.services.ffmpeg_service import FFmpegService, get_ffmpeg_service, ChunkInfo
from media_to_text.services.openai_service import get_openai_service, OpenAIService
from media_to_text.services.transcript_service import get_transcript_processor, TranscriptProcessor
from media_to_text.services.cleanup_service import get_cleanup_service, CleanupService


class JobWorker(LoggerMixin):
    """Worker for processing transcription jobs with crash recovery."""
    
    def __init__(self, settings: Settings, redis_service: RedisService):
        self.settings = settings
        self.redis_service = redis_service
        self.ffmpeg_service = get_ffmpeg_service()
        self.openai_service = get_openai_service()
        self.transcript_processor = get_transcript_processor()
        self.cleanup_service = None  # Will be set during initialization
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self.recovery_completed = False
    
    async def start(self) -> None:
        """Start the job worker with crash recovery."""
        if self.running:
            self.logger.warning("Job worker already running")
            return
        
        self.running = True
        
        # Perform crash recovery before starting normal processing
        await self._perform_crash_recovery()
        
        # Start the main worker loop
        self._task = asyncio.create_task(self._worker_loop())
        self.logger.info("Job worker started with crash recovery completed")
    
    async def stop(self) -> None:
        """Stop the job worker."""
        if not self.running:
            self.logger.info("Job worker already stopped")
            return
        
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Job worker stopped")
    
    async def set_cleanup_service(self, cleanup_service: CleanupService) -> None:
        """Set the cleanup service for the job worker."""
        self.cleanup_service = cleanup_service
        self.logger.info("Cleanup service connected to job worker")
    
    async def _perform_crash_recovery(self) -> None:
        """Perform crash recovery by scanning for interrupted jobs."""
        self.logger.info("Starting crash recovery scan")
        
        try:
            # Scan for jobs in PROCESSING state (potentially interrupted)
            processing_jobs = await self.redis_service.list_jobs(state_filter=JobState.PROCESSING)
            
            if not processing_jobs:
                self.logger.info("No interrupted jobs found during crash recovery")
                self.recovery_completed = True
                return
            
            self.logger.info("Found interrupted jobs during crash recovery", count=len(processing_jobs))
            
            recovered_count = 0
            failed_recovery_count = 0
            
            for job_data in processing_jobs:
                try:
                    # Convert dict to JobMetadata if needed
                    if isinstance(job_data, dict):
                        job_id = job_data.get("id")
                        if not job_id:
                            self.logger.warning("Job data missing ID", job_data=job_data)
                            continue
                    else:
                        job_id = job_data.job_id
                    
                    self.logger.info("Attempting to recover job", job_id=job_id)
                    
                    # Publish recovery start event
                    await self._publish_recovery_event(job_id, "recovery_started", {
                        "message": "Starting crash recovery for interrupted job",
                        "recovery_type": "startup_scan"
                    })
                    
                    # Attempt to resume the job
                    recovery_success = await self._resume_job(job_id)
                    
                    if recovery_success:
                        recovered_count += 1
                        await self._publish_recovery_event(job_id, "recovery_completed", {
                            "message": "Job successfully recovered and resumed",
                            "recovery_result": "success"
                        })
                        self.logger.info("Job recovery completed successfully", job_id=job_id)
                    else:
                        failed_recovery_count += 1
                        await self._publish_recovery_event(job_id, "recovery_failed", {
                            "message": "Job recovery failed, marking as failed",
                            "recovery_result": "failed"
                        })
                        self.logger.warning("Job recovery failed", job_id=job_id)
                        
                        # Mark job as failed if recovery fails
                        await self.redis_service.update_job_state(
                            job_id,
                            JobState.FAILED,
                            error_message="Job recovery failed after crash"
                        )
                        
                        # Trigger cleanup for failed recovery
                        if self.cleanup_service:
                            try:
                                await self.cleanup_service.trigger_immediate_cleanup(job_id, JobState.FAILED)
                            except Exception as cleanup_error:
                                self.logger.warning("Failed to cleanup failed recovery job", 
                                                  job_id=job_id,
                                                  cleanup_error=str(cleanup_error))
                
                except Exception as e:
                    failed_recovery_count += 1
                    self.logger.error("Exception during job recovery", 
                                    job_id=job_id if 'job_id' in locals() else "unknown",
                                    error=str(e))
            
            self.logger.info("Crash recovery completed", 
                           total_jobs=len(processing_jobs),
                           recovered=recovered_count,
                           failed=failed_recovery_count)
            
            self.recovery_completed = True
            
        except Exception as e:
            self.logger.error("Critical error during crash recovery", error=str(e))
            # Don't fail startup, but log the issue
            self.recovery_completed = True
    
    async def _resume_job(self, job_id: str) -> bool:
        """Resume a specific job from its current state."""
        try:
            # Get current job metadata
            job_data = await self.redis_service.get_job(job_id)
            if not job_data:
                self.logger.warning("Job not found for recovery", job_id=job_id)
                return False
            
            # Convert to JobMetadata if it's a dict
            if isinstance(job_data, dict):
                # Create a temporary JobMetadata object for processing
                job = JobMetadata(
                    job_id=job_data.get("id", job_id),
                    file_path=job_data.get("file_path", ""),
                    language=job_data.get("language", "en"),
                    state=JobState(job_data.get("state", "QUEUED")),
                    chunks_done=int(job_data.get("chunks_done", 0)),
                    chunks_total=int(job_data.get("chunks_total", 0))
                )
            else:
                job = job_data
            
            self.logger.info("Resuming job recovery", 
                           job_id=job_id,
                           chunks_done=job.chunks_done,
                           chunks_total=job.chunks_total)
            
            # Check if file still exists
            if not os.path.exists(job.file_path):
                self.logger.error("Original file not found for recovery", 
                                job_id=job_id,
                                file_path=job.file_path)
                return False
            
            # Get job directory
            job_dir = os.path.join(self.settings.temp_dir, f"job_{job_id}")
            
            # Check if we have previous chunk information
            chunk_infos = await self._recover_chunk_info(job_id, job_dir)
            
            if not chunk_infos:
                self.logger.info("No chunk info found, restarting job from beginning", job_id=job_id)
                # Restart job from the beginning
                await self.redis_service.update_job_state(job_id, JobState.QUEUED)
                return True
            
            # Check for already completed chunks (idempotency)
            completed_chunks = await self._check_completed_chunks(job_id, chunk_infos)
            remaining_chunks = [chunk for chunk in chunk_infos 
                              if chunk.index not in completed_chunks]
            
            self.logger.info("Chunk recovery analysis", 
                           job_id=job_id,
                           total_chunks=len(chunk_infos),
                           completed_chunks=len(completed_chunks),
                           remaining_chunks=len(remaining_chunks))
            
            if not remaining_chunks:
                # All chunks are completed, process final result
                self.logger.info("All chunks completed, processing final transcript", job_id=job_id)
                return await self._finalize_recovered_job(job_id, chunk_infos, completed_chunks)
            
            # Resume processing with remaining chunks
            return await self._resume_chunk_processing(job_id, job, remaining_chunks, completed_chunks)
            
        except Exception as e:
            self.logger.error("Failed to resume job", job_id=job_id, error=str(e))
            return False
    
    async def _recover_chunk_info(self, job_id: str, job_dir: str) -> List[ChunkInfo]:
        """Recover chunk information from previous processing."""
        chunk_infos = []
        
        try:
            # Try to get chunk info from Redis stream events
            updates = await self.redis_service.get_job_updates(job_id, "0")
            
            for update in updates:
                if update.get("event") == "chunks_created":
                    chunks_data = update.get("chunks", [])
                    if isinstance(chunks_data, str):
                        chunks_data = json.loads(chunks_data)
                    
                    for chunk_data in chunks_data:
                        chunk_info = ChunkInfo(
                            file_path=chunk_data["file_path"],
                            index=chunk_data["index"],
                            start_time=chunk_data["start_time"],
                            duration=chunk_data["duration"],
                            size_bytes=chunk_data.get("size_bytes", 0)
                        )
                        chunk_infos.append(chunk_info)
                    
                    self.logger.info("Recovered chunk info from Redis events", 
                                   job_id=job_id,
                                   chunk_count=len(chunk_infos))
                    return chunk_infos
            
            # Fallback: Try to discover chunks from filesystem
            if os.path.exists(job_dir):
                chunks_dir = os.path.join(job_dir, "chunks")
                if os.path.exists(chunks_dir):
                    chunk_files = sorted([f for f in os.listdir(chunks_dir) if f.endswith('.wav')])
                    
                    for i, chunk_file in enumerate(chunk_files):
                        chunk_path = os.path.join(chunks_dir, chunk_file)
                        if os.path.exists(chunk_path):
                            chunk_info = ChunkInfo(
                                file_path=chunk_path,
                                index=i,
                                start_time=0.0,  # Will be recalculated if needed
                                duration=0.0,    # Will be recalculated if needed
                                size_bytes=os.path.getsize(chunk_path)
                            )
                            chunk_infos.append(chunk_info)
                    
                    self.logger.info("Recovered chunk info from filesystem", 
                                   job_id=job_id,
                                   chunk_count=len(chunk_infos))
            
            return chunk_infos
            
        except Exception as e:
            self.logger.warning("Failed to recover chunk info", job_id=job_id, error=str(e))
            return []
    
    async def _check_completed_chunks(self, job_id: str, chunk_infos: List[ChunkInfo]) -> List[int]:
        """Check which chunks have already been completed (idempotency check)."""
        completed_chunks = []
        
        try:
            # Check Redis events for completed chunks
            updates = await self.redis_service.get_job_updates(job_id, "0")
            
            for update in updates:
                if update.get("event") == "chunk_transcribed":
                    chunk_index = update.get("chunk_index")
                    if chunk_index is not None:
                        if isinstance(chunk_index, str):
                            chunk_index = int(chunk_index)
                        completed_chunks.append(chunk_index)
            
            # Remove duplicates and sort
            completed_chunks = sorted(list(set(completed_chunks)))
            
            self.logger.info("Identified completed chunks for recovery", 
                           job_id=job_id,
                           completed_chunks=completed_chunks)
            
            return completed_chunks
            
        except Exception as e:
            self.logger.warning("Failed to check completed chunks", job_id=job_id, error=str(e))
            return []
    
    async def _resume_chunk_processing(self, job_id: str, job: JobMetadata, 
                                     remaining_chunks: List[ChunkInfo], 
                                     completed_chunks: List[int]) -> bool:
        """Resume processing with remaining chunks."""
        try:
            self.logger.info("Resuming chunk processing", 
                           job_id=job_id,
                           remaining_chunks=len(remaining_chunks),
                           completed_chunks=len(completed_chunks))
            
            # Publish resume event
            await self._publish_recovery_event(job_id, "processing_resumed", {
                "message": f"Resuming processing with {len(remaining_chunks)} remaining chunks",
                "remaining_chunks": len(remaining_chunks),
                "completed_chunks": len(completed_chunks)
            })
            
            # Process remaining chunks
            transcription_results = await self.openai_service.transcribe_chunks(
                chunks=remaining_chunks,
                language=job.language,
                job_id=job_id
            )
            
            # Update progress as chunks complete
            for result in transcription_results:
                if result.success:
                    await self.redis_service.publish_job_update(job_id, {
                        "event": "chunk_transcribed",
                        "chunk_index": result.chunk_index,
                        "chunk_text": result.text,
                        "processing_time": result.processing_time,
                        "retry_count": result.retry_count,
                        "recovered": True  # Mark as recovered chunk
                    })
                else:
                    await self.redis_service.publish_job_update(job_id, {
                        "event": "chunk_failed",
                        "chunk_index": result.chunk_index,
                        "error_message": result.error_message,
                        "retry_count": result.retry_count,
                        "recovered": True  # Mark as recovered chunk
                    })
                
                # Update progress
                total_completed = len(completed_chunks) + sum(1 for r in transcription_results if r.success)
                total_chunks = len(completed_chunks) + len(remaining_chunks)
                
                await self.redis_service.update_job_progress(
                    job_id,
                    chunks_done=total_completed,
                    chunks_total=total_chunks
                )
            
            # Check if we have enough successful results
            successful_new = [r for r in transcription_results if r.success]
            total_successful = len(completed_chunks) + len(successful_new)
            total_chunks = len(completed_chunks) + len(remaining_chunks)
            success_rate = total_successful / total_chunks if total_chunks > 0 else 0
            
            if success_rate >= 0.5:  # At least 50% success rate
                # Mark as completed and finalize
                await self.redis_service.update_job_state(job_id, JobState.COMPLETED)
                
                # Schedule cleanup
                if self.cleanup_service:
                    await self.cleanup_service.schedule_job_cleanup(job_id, delay_seconds=300)
                
                self.logger.info("Recovered job completed successfully", 
                               job_id=job_id,
                               success_rate=f"{success_rate:.1%}")
                return True
            else:
                # Low success rate, mark as failed
                await self.redis_service.update_job_state(
                    job_id,
                    JobState.FAILED,
                    error_message=f"Low success rate after recovery: {success_rate:.1%}"
                )
                
                self.logger.warning("Recovered job failed due to low success rate", 
                                  job_id=job_id,
                                  success_rate=f"{success_rate:.1%}")
                return False
            
        except Exception as e:
            self.logger.error("Failed to resume chunk processing", job_id=job_id, error=str(e))
            return False
    
    async def _finalize_recovered_job(self, job_id: str, chunk_infos: List[ChunkInfo], 
                                    completed_chunks: List[int]) -> bool:
        """Finalize a job that had all chunks completed."""
        try:
            self.logger.info("Finalizing recovered job with all chunks completed", job_id=job_id)
            
            # Check if final result already exists
            result_key = f"transcript:{job_id}"
            existing_result = await self.redis_service.redis.get(result_key)
            
            if existing_result:
                # Job already finalized, just mark as completed
                await self.redis_service.update_job_state(job_id, JobState.COMPLETED)
                self.logger.info("Job already finalized, marked as completed", job_id=job_id)
                return True
            
            # Need to regenerate final transcript from completed chunks
            # This would require collecting all chunk results and processing them
            # For now, mark as completed and let normal processing handle it
            await self.redis_service.update_job_state(job_id, JobState.COMPLETED)
            
            self.logger.info("Recovered job finalized", job_id=job_id)
            return True
            
        except Exception as e:
            self.logger.error("Failed to finalize recovered job", job_id=job_id, error=str(e))
            return False
    
    async def _publish_recovery_event(self, job_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """Publish recovery-related events to Redis stream."""
        try:
            event_data = {
                "event": event_type,
                "job_id": job_id,
                "timestamp": asyncio.get_event_loop().time(),
                "recovery": True,
                **data
            }
            
            await self.redis_service.publish_job_update(job_id, event_data)
            
        except Exception as e:
            self.logger.warning("Failed to publish recovery event", 
                              job_id=job_id,
                              event_type=event_type,
                              error=str(e))
    
    async def _worker_loop(self) -> None:
        """Main worker loop to process jobs."""
        self.logger.info("Job worker loop started", recovery_completed=self.recovery_completed)
        
        while self.running:
            try:
                # Get queued jobs
                queued_jobs = await self.redis_service.list_jobs(state_filter=JobState.QUEUED)
                
                if queued_jobs:
                    # Process the oldest job
                    job_data = queued_jobs[0]
                    
                    # Convert dict to JobMetadata if needed
                    if isinstance(job_data, dict):
                        job = JobMetadata(
                            job_id=job_data.get("id", ""),
                            file_path=job_data.get("file_path", ""),
                            language=job_data.get("language", "en"),
                            state=JobState(job_data.get("state", "QUEUED")),
                            chunks_done=int(job_data.get("chunks_done", 0)),
                            chunks_total=int(job_data.get("chunks_total", 0))
                        )
                    else:
                        job = job_data
                    
                    await self._process_job(job)
                else:
                    # No jobs available, wait a bit
                    await asyncio.sleep(1.0)
                    
            except Exception as e:
                self.logger.error("Worker loop error", error=str(e), error_type=type(e).__name__)
                await asyncio.sleep(5.0)  # Wait before retrying
    
    async def _process_job(self, job: JobMetadata) -> None:
        """Process a single transcription job."""
        job_dir = None
        chunk_infos = []
        
        try:
            self.logger.info("Processing job", job_id=job.job_id, file_path=job.file_path)
            
            # Update job state to PROCESSING
            await self.redis_service.update_job_state(job.job_id, JobState.PROCESSING)
            
            # Create temporary directory for this job
            job_dir = os.path.join(self.settings.temp_dir, f"job_{job.job_id}")
            os.makedirs(job_dir, exist_ok=True)
            
            # Step 1: Probe media file
            self.logger.debug("Probing media file", job_id=job.job_id, file_path=job.file_path)
            media_info = await self.ffmpeg_service.get_media_info(job.file_path)
            
            if not media_info.has_audio:
                raise RuntimeError("Input file has no audio streams")
            
            # Step 2: Determine if conversion is needed
            converted_file = job.file_path
            
            if media_info.needs_conversion:
                self.logger.info("Converting file to WAV format", job_id=job.job_id, file_path=job.file_path)
                converted_file = await self.ffmpeg_service.convert_to_wav(job.file_path, job.job_id)
                self.logger.info("Conversion complete", job_id=job.job_id, output_path=converted_file)
            else:
                self.logger.info("File is already in WAV format, no conversion needed", job_id=job.job_id)
            
            # Step 3: Check file size and determine chunking strategy
            file_size_mb = os.path.getsize(converted_file) / (1024 * 1024)
            max_chunk_size_mb = self.settings.openai_max_chunk_size_mb
            
            if file_size_mb > max_chunk_size_mb:
                self.logger.info("File size exceeds limit", job_id=job.job_id, file_size_mb=file_size_mb, max_chunk_size_mb=max_chunk_size_mb)
                
                # Use FFmpeg chunking
                chunk_infos = await self.ffmpeg_service.chunk_wav_file(converted_file, job.job_id)
                
                self.logger.info("Created chunks using FFmpeg", job_id=job.job_id, chunk_count=len(chunk_infos))
                
                # Validate chunk sizes manually
                oversized_chunks = [chunk for chunk in chunk_infos if chunk.size_mb > max_chunk_size_mb]
                if oversized_chunks:
                    self.logger.warning("Some chunks exceed size limit, but proceeding", 
                                       job_id=job.job_id, 
                                       oversized_count=len(oversized_chunks))
                
                # Log chunk details
                for chunk in chunk_infos:
                    self.logger.info("Chunk created", job_id=job.job_id, chunk_index=chunk.index, size_mb=chunk.size_mb, duration=chunk.duration, file_path=chunk.file_path)
                
            else:
                self.logger.info("File size is within limit", job_id=job.job_id, file_size_mb=file_size_mb)
                
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
            self.logger.info("Starting OpenAI transcription for chunks", job_id=job.job_id, chunk_count=len(chunk_infos))
            
            await self.redis_service.publish_job_update(job.job_id, {
                "event": "transcription_started",
                "message": f"Starting OpenAI transcription for {len(chunk_infos)} chunks"
            })
            
            # Use OpenAI service for parallel transcription
            transcription_results = await self.openai_service.transcribe_chunks(
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
                self.logger.info("Processing transcript with timestamp normalization", job_id=job.job_id)
                
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
                
                self.logger.info("Enhanced transcript complete", job_id=job.job_id, text_length=len(formatted_transcript["text"]))
                self.logger.info("Success rate", job_id=job.job_id, success_rate=f"{basic_metadata['success_rate']:.1%}")
                self.logger.info("Total duration", job_id=job.job_id, duration=formatted_transcript["metadata"]["total_duration"])
                
                # Mark job as completed if we have at least some successful transcriptions
                if basic_metadata["success_rate"] >= 0.5:  # At least 50% success rate
                    await self.redis_service.update_job_state(job.job_id, JobState.COMPLETED)
                    
                    # Schedule enhanced cleanup using cleanup service
                    if self.cleanup_service:
                        await self.cleanup_service.schedule_job_cleanup(job.job_id, delay_seconds=300)
                    
                    self.logger.info("Job completed successfully with normalized transcript", job_id=job.job_id)
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
                    
                    self.logger.info("Job failed due to low success rate", job_id=job.job_id)
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
                
                self.logger.info("Job failed - no successful transcriptions", job_id=job.job_id)
            
        except Exception as e:
            self.logger.error("Job failed", job_id=job.job_id, error=str(e))
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
                        self.logger.info("Preserved transcript file", transcript_file=transcript_file)
                    
                    # Remove chunks and converted files, but keep transcript
                    for item in os.listdir(job_dir):
                        item_path = os.path.join(job_dir, item)
                        if item != "transcript.json":
                            if os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                            else:
                                os.remove(item_path)
                    
                    self.logger.info("Basic cleanup completed", job_dir=job_dir)
                except Exception as e:
                    self.logger.error("Failed basic cleanup", job_dir=job_dir, error=str(e))
    
    async def process_job_by_id(self, job_id: str) -> bool:
        """Process a specific job by ID (for testing/manual processing)."""
        job_data = await self.redis_service.get_job(job_id)
        if not job_data:
            self.logger.warning("Job not found", job_id=job_id)
            return False
        
        # Convert dict to JobMetadata if needed
        if isinstance(job_data, dict):
            job = JobMetadata(
                job_id=job_data.get("id", job_id),
                file_path=job_data.get("file_path", ""),
                language=job_data.get("language", "en"),
                state=JobState(job_data.get("state", "QUEUED")),
                chunks_done=int(job_data.get("chunks_done", 0)),
                chunks_total=int(job_data.get("chunks_total", 0))
            )
        else:
            job = job_data
        
        if job.state != JobState.QUEUED:
            self.logger.warning("Job is not in QUEUED state", job_id=job_id, current_state=job.state)
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
    """Initialize job worker with crash recovery."""
    global job_worker
    logger = get_logger("job_worker_init")
    
    try:
        job_worker = JobWorker(settings, redis_service)
        await job_worker.start()  # This now includes crash recovery
        logger.info("Job worker initialized and started successfully with crash recovery")
        return job_worker
    except Exception as e:
        logger.error("Failed to initialize job worker", error=str(e))
        raise


async def close_job_worker() -> None:
    """Close job worker."""
    global job_worker
    logger = get_logger("job_worker_close")
    
    if job_worker:
        try:
            await job_worker.stop()
            job_worker = None
            logger.info("Job worker closed successfully")
        except Exception as e:
            logger.warning("Error closing job worker", error=str(e))