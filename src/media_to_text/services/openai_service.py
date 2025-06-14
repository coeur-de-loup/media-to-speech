"""OpenAI transcription service with parallel processing and rate limiting."""

import asyncio
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openai
from openai import AsyncOpenAI

from media_to_text.config import Settings
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


class RateLimiter:
    """Token bucket rate limiter for API requests."""
    
    def __init__(self, requests_per_second: float = 10.0, burst_size: int = 20):
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
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
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class OpenAITranscriptionService:
    """Service for parallel OpenAI transcription with rate limiting."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.rate_limiter = RateLimiter(
            requests_per_second=float(settings.openai_max_parallel_requests) / 2,  # Conservative rate limiting
            burst_size=settings.openai_max_parallel_requests
        )
        self.semaphore = asyncio.Semaphore(settings.openai_max_parallel_requests)
        
    async def transcribe_chunks_parallel(
        self, 
        chunks: List[ChunkInfo], 
        language: str = "en",
        job_id: Optional[str] = None
    ) -> List[TranscriptionResult]:
        """
        Transcribe multiple chunks in parallel with rate limiting.
        
        Args:
            chunks: List of audio chunks to transcribe
            language: Language code for transcription
            job_id: Optional job ID for logging and idempotency
        
        Returns:
            List of TranscriptionResult objects in chunk order
        """
        print(f"🚀 Starting parallel transcription of {len(chunks)} chunks")
        
        # Create tasks for parallel processing
        tasks = []
        for chunk in chunks:
            task = asyncio.create_task(
                self._transcribe_chunk_with_retry(chunk, language, job_id)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        transcription_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle exceptions from failed tasks
                error_result = TranscriptionResult(
                    chunk_index=i,
                    text="",
                    chunk_info=chunks[i],
                    processing_time=0.0,
                    success=False,
                    error_message=str(result)
                )
                transcription_results.append(error_result)
            else:
                transcription_results.append(result)
        
        # Sort results by chunk index to maintain order
        transcription_results.sort(key=lambda x: x.chunk_index)
        
        # Log summary
        successful = sum(1 for r in transcription_results if r.success)
        failed = len(transcription_results) - successful
        print(f"✅ Transcription complete: {successful} successful, {failed} failed")
        
        return transcription_results
    
    async def _transcribe_chunk_with_retry(
        self, 
        chunk: ChunkInfo, 
        language: str,
        job_id: Optional[str] = None,
        max_retries: int = 3
    ) -> TranscriptionResult:
        """
        Transcribe a single chunk with retry logic.
        
        Args:
            chunk: Audio chunk to transcribe
            language: Language code
            job_id: Optional job ID
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
                    
                    print(f"✅ Chunk {chunk.index} transcribed successfully in {processing_time:.2f}s")
                    
                    return TranscriptionResult(
                        chunk_index=chunk.index,
                        text=result,
                        chunk_info=chunk,
                        processing_time=processing_time,
                        success=True,
                        retry_count=retry_count
                    )
                    
            except openai.RateLimitError as e:
                print(f"⏳ Rate limit hit for chunk {chunk.index}, retry {retry_count + 1}")
                wait_time = min(2 ** retry_count, 60)  # Exponential backoff, max 60s
                await asyncio.sleep(wait_time)
                retry_count += 1
                last_error = e
                
            except openai.APIError as e:
                if e.status_code >= 500:  # Server errors are retryable
                    print(f"🔄 Server error for chunk {chunk.index}, retry {retry_count + 1}: {e}")
                    wait_time = min(2 ** retry_count, 30)
                    await asyncio.sleep(wait_time)
                    retry_count += 1
                    last_error = e
                else:
                    # Client errors (4xx) are not retryable
                    print(f"❌ Client error for chunk {chunk.index}: {e}")
                    break
                    
            except Exception as e:
                print(f"❌ Unexpected error for chunk {chunk.index}: {e}")
                last_error = e
                break
        
        # All retries failed
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
        job_id: Optional[str] = None,
        retry_count: int = 0
    ) -> str:
        """
        Make the actual transcription request to OpenAI.
        
        Args:
            chunk: Audio chunk to transcribe
            language: Language code
            job_id: Optional job ID for idempotency
            retry_count: Current retry attempt
        
        Returns:
            Transcribed text
        
        Raises:
            openai.OpenAIError: If the API request fails
        """
        # Create idempotency key
        idempotency_key = f"{job_id}_{chunk.index}_{retry_count}" if job_id else str(uuid.uuid4())
        
        # Open and read the audio file
        audio_file_path = Path(chunk.file_path)
        if not audio_file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {chunk.file_path}")
        
        print(f"🎯 Transcribing chunk {chunk.index}: {audio_file_path.name} ({chunk.size_mb:.1f}MB)")
        
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
            print(f"⚠️  Empty transcription for chunk {chunk.index}")
            return ""
        
        return transcribed_text
    
    async def combine_transcription_results(self, results: List[TranscriptionResult]) -> Tuple[str, Dict]:
        """
        Combine transcription results into final text and metadata.
        
        Args:
            results: List of transcription results
        
        Returns:
            Tuple of (combined_text, metadata)
        """
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
            "failed_chunk_indices": [r.chunk_index for r in results if not r.success]
        }
        
        return combined_text, metadata


# Global OpenAI service instance
openai_service: Optional[OpenAITranscriptionService] = None


def get_openai_service() -> OpenAITranscriptionService:
    """Get OpenAI service instance."""
    global openai_service
    if openai_service is None:
        from media_to_text.config import Settings
        settings = Settings()
        openai_service = OpenAITranscriptionService(settings)
    return openai_service