"""OpenAI transcription service with parallel processing and rate limiting."""

import asyncio
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import openai
from openai import AsyncOpenAI

from media_to_text.config import Settings
from media_to_text.logging import LoggerMixin, get_logger
from media_to_text.services.ffmpeg_service import ChunkInfo


@dataclass
class TranscriptionResult:
    """Result of a transcription request."""
    chunk_index: int
    text: str
    chunk_info: ChunkInfo
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    retry_count: int = 0


class RateLimiter(LoggerMixin):
    """Token bucket rate limiter for API requests."""
    
    def __init__(self, requests_per_second: float = 10.0, burst_size: int = 20):
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self._lock = asyncio.Lock()
        
        self.logger.debug("Rate limiter initialized", 
                         requests_per_second=requests_per_second,
                         burst_size=burst_size)
    
    async def acquire(self) -> None:
        """Acquire a token for making a request."""
        async with self._lock:
            now = time.time()
            
            # Add tokens based on elapsed time
            elapsed = now - self.last_update
            tokens_to_add = elapsed * self.requests_per_second
            self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
            self.last_update = now
            
            # If no tokens available, wait
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.requests_per_second
                self.logger.debug("Rate limit exceeded, waiting", 
                                wait_time=wait_time,
                                current_tokens=self.tokens)
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1
                self.logger.debug("Token acquired", remaining_tokens=self.tokens)


class OpenAIService(LoggerMixin):
    """Service for parallel OpenAI transcription with rate limiting."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.rate_limiter = RateLimiter(
            requests_per_second=float(settings.openai_max_parallel_requests) / 2,  # Conservative rate limiting
            burst_size=settings.openai_max_parallel_requests
        )
        self.semaphore = asyncio.Semaphore(settings.openai_max_parallel_requests)
        
        self.logger.info("OpenAI service initialized", 
                        max_parallel_requests=settings.openai_max_parallel_requests,
                        max_chunk_size_mb=settings.openai_max_chunk_size_mb)
        
    async def transcribe_chunks(
        self, 
        chunks: List[ChunkInfo], 
        job_id: str,
        language: str = "en",
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[TranscriptionResult]:
        """
        Transcribe multiple chunks in parallel with rate limiting and progress tracking.
        
        Args:
            chunks: List of audio chunks to transcribe
            job_id: Job ID for logging and idempotency
            language: Language code for transcription
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of TranscriptionResult objects in chunk order
        """
        self.logger.info("Starting parallel transcription", 
                        job_id=job_id,
                        chunk_count=len(chunks),
                        language=language,
                        max_parallel=self.settings.openai_max_parallel_requests)
        
        # Create tasks for parallel processing
        tasks = []
        for chunk in chunks:
            task = asyncio.create_task(
                self._transcribe_chunk_with_retry(chunk, language, job_id)
            )
            tasks.append(task)
        
        # Process tasks as they complete for progress tracking
        transcription_results = []
        completed_count = 0
        
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                transcription_results.append(result)
                completed_count += 1
                
                # Call progress callback if provided
                if progress_callback:
                    try:
                        progress_callback(completed_count, len(chunks))
                    except Exception as e:
                        self.logger.warning("Progress callback failed", 
                                          job_id=job_id,
                                          error=str(e))
                
                if result.success:
                    self.logger.debug("Chunk transcription successful", 
                                    job_id=job_id,
                                    chunk_index=result.chunk_index,
                                    processing_time=result.processing_time,
                                    text_length=len(result.text))
                else:
                    self.logger.warning("Chunk transcription failed", 
                                      job_id=job_id,
                                      chunk_index=result.chunk_index,
                                      error=result.error_message,
                                      retry_count=result.retry_count)
                    
            except Exception as e:
                self.logger.error("Task processing failed", 
                                job_id=job_id,
                                error=str(e))
                # Create error result for failed task
                error_result = TranscriptionResult(
                    chunk_index=len(transcription_results),  # Best guess at index
                    text="",
                    chunk_info=chunks[len(transcription_results)] if len(transcription_results) < len(chunks) else chunks[0],
                    processing_time=0.0,
                    success=False,
                    error_message=str(e)
                )
                transcription_results.append(error_result)
                completed_count += 1
        
        # Sort results by chunk index to maintain order
        transcription_results.sort(key=lambda x: x.chunk_index)
        
        # Log summary
        successful = sum(1 for r in transcription_results if r.success)
        failed = len(transcription_results) - successful
        total_processing_time = sum(r.processing_time for r in transcription_results if r.success)
        success_rate = successful / len(transcription_results) if transcription_results else 0
        
        self.logger.info("Parallel transcription complete", 
                        job_id=job_id,
                        total_chunks=len(transcription_results),
                        successful_chunks=successful,
                        failed_chunks=failed,
                        success_rate=f"{success_rate:.1%}",
                        total_processing_time=total_processing_time)
        
        return transcription_results
    
    async def _transcribe_chunk_with_retry(
        self, 
        chunk: ChunkInfo, 
        language: str,
        job_id: str,
        max_retries: int = 3
    ) -> TranscriptionResult:
        """
        Transcribe a single chunk with retry logic.
        
        Args:
            chunk: Audio chunk to transcribe
            language: Language code
            job_id: Job ID for idempotency
            max_retries: Maximum number of retries
        
        Returns:
            TranscriptionResult with success/failure information
        """
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                # Acquire semaphore to limit concurrency
                async with self.semaphore:
                    # Rate limiting
                    await self.rate_limiter.acquire()
                    
                    start_time = time.time()
                    
                    # Make transcription request
                    result = await self._make_transcription_request(chunk, language, job_id, retry_count)
                    
                    processing_time = time.time() - start_time
                    
                    self.logger.info("Chunk transcribed successfully", 
                                   job_id=job_id,
                                   chunk_index=chunk.index,
                                   processing_time=processing_time,
                                   text_length=len(result),
                                   retry_count=retry_count)
                    
                    return TranscriptionResult(
                        chunk_index=chunk.index,
                        text=result,
                        chunk_info=chunk,
                        processing_time=processing_time,
                        success=True,
                        retry_count=retry_count
                    )
                    
            except openai.RateLimitError as e:
                self.logger.warning("Rate limit hit, retrying", 
                                  job_id=job_id,
                                  chunk_index=chunk.index,
                                  retry_count=retry_count + 1,
                                  max_retries=max_retries,
                                  error=str(e))
                wait_time = min(2 ** retry_count, 60)  # Exponential backoff, max 60s
                await asyncio.sleep(wait_time)
                retry_count += 1
                last_error = e
                
            except openai.APIError as e:
                if e.status_code >= 500:  # Server errors are retryable
                    self.logger.warning("Server error, retrying", 
                                      job_id=job_id,
                                      chunk_index=chunk.index,
                                      retry_count=retry_count + 1,
                                      status_code=e.status_code,
                                      error=str(e))
                    wait_time = min(2 ** retry_count, 30)
                    await asyncio.sleep(wait_time)
                    retry_count += 1
                    last_error = e
                else:
                    # Client errors (4xx) are not retryable
                    self.logger.error("Client error, not retrying", 
                                    job_id=job_id,
                                    chunk_index=chunk.index,
                                    status_code=e.status_code,
                                    error=str(e))
                    last_error = e
                    break
                    
            except Exception as e:
                self.logger.error("Unexpected error during transcription", 
                                job_id=job_id,
                                chunk_index=chunk.index,
                                error=str(e),
                                error_type=type(e).__name__)
                last_error = e
                break
        
        # All retries failed
        self.logger.error("Chunk transcription failed after retries", 
                        job_id=job_id,
                        chunk_index=chunk.index,
                        retry_count=retry_count,
                        final_error=str(last_error) if last_error else "Unknown error")
        
        return TranscriptionResult(
            chunk_index=chunk.index,
            text="",
            chunk_info=chunk,
            processing_time=0.0,
            success=False,
            error_message=str(last_error) if last_error else "Unknown error",
            retry_count=retry_count
        )
    
    async def _make_transcription_request(
        self, 
        chunk: ChunkInfo, 
        language: str,
        job_id: str,
        retry_count: int = 0
    ) -> str:
        """
        Make the actual transcription request to OpenAI.
        
        Args:
            chunk: Audio chunk to transcribe
            language: Language code
            job_id: Job ID for idempotency
            retry_count: Current retry attempt
        
        Returns:
            Transcribed text
        
        Raises:
            openai.OpenAIError: If the API request fails
        """
        # Create idempotency key
        idempotency_key = f"{job_id}_{chunk.index}_{retry_count}"
        
        # Open and read the audio file
        audio_file_path = Path(chunk.file_path)
        if not audio_file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {chunk.file_path}")
        
        self.logger.debug("Making transcription request", 
                         job_id=job_id,
                         chunk_index=chunk.index,
                         file_path=str(audio_file_path),
                         file_size_mb=chunk.size_mb,
                         duration=chunk.duration,
                         idempotency_key=idempotency_key)
        
        with open(audio_file_path, "rb") as audio_file:
            # Make the transcription request
            response = await self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format="text",
                # Add metadata in prompt if helpful
                prompt=f"Audio chunk {chunk.index + 1}, duration: {chunk.duration:.1f}s"
            )
        
        # OpenAI returns the text directly when response_format="text"
        transcribed_text = response.strip() if isinstance(response, str) else str(response).strip()
        
        if not transcribed_text:
            self.logger.warning("Empty transcription result", 
                              job_id=job_id,
                              chunk_index=chunk.index)
            return ""
        
        self.logger.debug("Transcription request successful", 
                         job_id=job_id,
                         chunk_index=chunk.index,
                         text_length=len(transcribed_text))
        
        return transcribed_text
    
    async def combine_transcription_results(self, results: List[TranscriptionResult]) -> Tuple[str, Dict]:
        """
        Combine transcription results into final text and metadata.
        
        Args:
            results: List of transcription results
        
        Returns:
            Tuple of (combined_text, metadata)
        """
        self.logger.debug("Combining transcription results", 
                         total_results=len(results))
        
        # Filter successful results and sort by chunk index
        successful_results = [r for r in results if r.success]
        successful_results.sort(key=lambda x: x.chunk_index)
        
        # Combine text
        combined_text = " ".join(r.text for r in successful_results if r.text.strip())
        
        # Calculate metadata
        total_chunks = len(results)
        successful_chunks = len(successful_results)
        failed_chunks = total_chunks - successful_chunks
        total_processing_time = sum(r.processing_time for r in successful_results)
        total_retries = sum(r.retry_count for r in results)
        
        metadata = {
            "total_chunks": total_chunks,
            "successful_chunks": successful_chunks,
            "failed_chunks": failed_chunks,
            "success_rate": successful_chunks / total_chunks if total_chunks > 0 else 0.0,
            "total_processing_time": total_processing_time,
            "average_processing_time": total_processing_time / successful_chunks if successful_chunks > 0 else 0.0,
            "total_retries": total_retries,
            "failed_chunk_indices": [r.chunk_index for r in results if not r.success],
            "combined_text_length": len(combined_text)
        }
        
        self.logger.info("Transcription results combined", 
                        total_chunks=total_chunks,
                        successful_chunks=successful_chunks,
                        failed_chunks=failed_chunks,
                        success_rate=f"{metadata['success_rate']:.1%}",
                        final_text_length=len(combined_text),
                        total_processing_time=total_processing_time)
        
        return combined_text, metadata


# Global OpenAI service instance
_openai_service: Optional[OpenAIService] = None


def init_openai_service(settings: Settings) -> OpenAIService:
    """Initialize OpenAI service."""
    global _openai_service
    logger = get_logger("openai_init")
    
    try:
        _openai_service = OpenAIService(settings)
        logger.info("OpenAI service initialized successfully")
        return _openai_service
    except Exception as e:
        logger.error("Failed to initialize OpenAI service", error=str(e))
        raise


def get_openai_service() -> OpenAIService:
    """Get the global OpenAI service instance."""
    if _openai_service is None:
        raise RuntimeError("OpenAI service not initialized")
    return _openai_service