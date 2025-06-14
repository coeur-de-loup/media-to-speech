"""FFmpeg service for media processing and conversion."""

import asyncio
import json
import os
import re
import subprocess
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

from media_to_text.config import Settings
from media_to_text.logging import LoggerMixin, get_logger


@dataclass
class AudioStreamInfo:
    """Audio stream information."""
    index: int
    codec: str
    sample_rate: int
    channels: int
    duration: float
    bitrate: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class MediaInfo:
    """Media file information."""
    format: str
    duration: float
    size: int
    audio_streams: List[AudioStreamInfo]
    
    @property
    def has_audio(self) -> bool:
        """Check if file has audio streams."""
        return len(self.audio_streams) > 0
    
    @property
    def needs_conversion(self) -> bool:
        """Check if file needs conversion to WAV."""
        if not self.has_audio:
            return False
        
        # Check if it's already WAV format with compatible settings
        if self.format.lower() == "wav":
            primary_audio = self.audio_streams[0]
            # Check for 16-bit PCM WAV (commonly supported)
            return not (primary_audio.sample_rate >= 16000 and primary_audio.channels <= 2)
        
        return True
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "format": self.format,
            "duration": self.duration,
            "size": self.size,
            "audio_streams": [stream.to_dict() for stream in self.audio_streams],
            "has_audio": self.has_audio,
            "needs_conversion": self.needs_conversion
        }


@dataclass
class ChunkInfo:
    """Information about a media chunk."""
    file_path: str
    index: int
    start_time: float
    duration: float
    size_bytes: int = 0
    
    @property
    def end_time(self) -> float:
        """Calculate end time."""
        return self.start_time + self.duration
    
    @property
    def size_mb(self) -> float:
        """Get size in MB."""
        return self.size_bytes / (1024 * 1024)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class FFmpegService(LoggerMixin):
    """Service for FFmpeg operations via Docker container."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
    async def get_media_info(self, file_path: str) -> MediaInfo:
        """Get comprehensive media file information using ffprobe."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.logger.debug("Analyzing media file", file_path=file_path)
        
        try:
            # Use ffprobe to get detailed media information
            cmd = [
                "docker", "exec", "ffmpeg",
                "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams",
                f"/workspace/{os.path.basename(file_path)}"
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown ffprobe error"
                self.logger.error("ffprobe failed", file_path=file_path, error=error_msg)
                raise RuntimeError(f"ffprobe failed: {error_msg}")
            
            # Parse ffprobe output
            probe_data = json.loads(stdout.decode())
            
            # Extract format information
            format_info = probe_data.get("format", {})
            duration = float(format_info.get("duration", 0))
            size = int(format_info.get("size", 0))
            format_name = format_info.get("format_name", "unknown")
            
            # Extract audio streams
            audio_streams = []
            for stream in probe_data.get("streams", []):
                if stream.get("codec_type") == "audio":
                    audio_stream = AudioStreamInfo(
                        index=stream.get("index", 0),
                        codec=stream.get("codec_name", "unknown"),
                        sample_rate=int(stream.get("sample_rate", 0)),
                        channels=stream.get("channels", 0),
                        duration=float(stream.get("duration", duration)),
                        bitrate=int(stream.get("bit_rate", 0)) if stream.get("bit_rate") else None
                    )
                    audio_streams.append(audio_stream)
            
            media_info = MediaInfo(
                format=format_name,
                duration=duration,
                size=size,
                audio_streams=audio_streams
            )
            
            self.logger.info("Media analysis complete", 
                           file_path=file_path, 
                           duration=duration,
                           format=format_name,
                           audio_streams=len(audio_streams))
            
            return media_info
            
        except json.JSONDecodeError as e:
            self.logger.error("Failed to parse ffprobe output", file_path=file_path, error=str(e))
            raise RuntimeError(f"Failed to parse ffprobe output: {e}")
        except Exception as e:
            self.logger.error("Media analysis failed", file_path=file_path, error=str(e))
            raise
    
    async def convert_to_wav(self, input_path: str, job_id: str) -> str:
        """Convert media file to 16-bit PCM WAV format."""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        output_path = f"/tmp/{job_id}/converted.wav"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.logger.info("Converting to WAV format", 
                        input_path=input_path, 
                        output_path=output_path,
                        job_id=job_id)
        
        try:
            # FFmpeg command for conversion to 16-bit PCM WAV
            cmd = [
                "docker", "exec", "ffmpeg",
                "ffmpeg", "-i", f"/workspace/{os.path.basename(input_path)}",
                "-acodec", "pcm_s16le",  # 16-bit PCM
                "-ar", "16000",          # 16kHz sample rate
                "-ac", "1",              # Mono
                "-y",                    # Overwrite output
                f"/tmp/{job_id}/converted.wav"
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown ffmpeg error"
                self.logger.error("WAV conversion failed", 
                                input_path=input_path, 
                                output_path=output_path,
                                error=error_msg)
                raise RuntimeError(f"FFmpeg conversion failed: {error_msg}")
            
            # Verify output file exists
            if not os.path.exists(output_path):
                raise RuntimeError(f"Conversion completed but output file not found: {output_path}")
            
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            self.logger.info("WAV conversion complete", 
                           input_path=input_path,
                           output_path=output_path,
                           size_mb=file_size_mb)
            
            return output_path
            
        except Exception as e:
            self.logger.error("WAV conversion failed", 
                            input_path=input_path,
                            error=str(e))
            raise
    
    async def chunk_wav_file(self, wav_path: str, job_id: str) -> List[ChunkInfo]:
        """Chunk WAV file using FFmpeg segments for files larger than 25MB."""
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"WAV file not found: {wav_path}")
        
        # Get file info
        media_info = await self.get_media_info(wav_path)
        file_size_mb = os.path.getsize(wav_path) / (1024 * 1024)
        max_chunk_size_mb = self.settings.openai_max_chunk_size_mb
        
        self.logger.info("Chunking WAV file", 
                        wav_path=wav_path,
                        file_size_mb=file_size_mb,
                        max_chunk_size_mb=max_chunk_size_mb,
                        duration=media_info.duration,
                        job_id=job_id)
        
        # Calculate optimal chunk duration based on file size and bitrate
        estimated_bitrate = (file_size_mb * 8 * 1024 * 1024) / media_info.duration  # bits per second
        target_chunk_size_bytes = max_chunk_size_mb * 1024 * 1024
        optimal_chunk_duration = (target_chunk_size_bytes * 8) / estimated_bitrate
        
        # Ensure reasonable chunk duration (between 30 seconds and 10 minutes)
        chunk_duration = max(30, min(600, optimal_chunk_duration))
        
        self.logger.debug("Calculated chunk parameters", 
                         job_id=job_id,
                         estimated_bitrate=estimated_bitrate,
                         optimal_chunk_duration=optimal_chunk_duration,
                         final_chunk_duration=chunk_duration)
        
        output_dir = f"/tmp/{job_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Use FFmpeg segment feature for precise chunking
            cmd = [
                "docker", "exec", "ffmpeg",
                "ffmpeg", "-i", wav_path,
                "-f", "segment",
                "-segment_time", str(chunk_duration),
                "-fs", f"{max_chunk_size_mb}M",  # File size limit
                "-c", "copy",  # Copy without re-encoding
                "-y",  # Overwrite
                f"/tmp/{job_id}/chunk_%03d.wav"
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown ffmpeg error"
                self.logger.error("WAV chunking failed", 
                                wav_path=wav_path, 
                                error=error_msg,
                                job_id=job_id)
                raise RuntimeError(f"FFmpeg chunking failed: {error_msg}")
            
            # Collect chunk information
            chunks = []
            chunk_index = 0
            current_start_time = 0.0
            
            while True:
                chunk_path = f"/tmp/{job_id}/chunk_{chunk_index:03d}.wav"
                if not os.path.exists(chunk_path):
                    break
                
                # Get chunk duration using ffprobe
                chunk_info_cmd = [
                    "docker", "exec", "ffmpeg",
                    "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                    "-of", "csv=p=0", chunk_path
                ]
                
                info_result = await asyncio.create_subprocess_exec(
                    *chunk_info_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                info_stdout, _ = await info_result.communicate()
                chunk_duration_actual = float(info_stdout.decode().strip()) if info_stdout else chunk_duration
                
                chunk_info = ChunkInfo(
                    file_path=chunk_path,
                    index=chunk_index,
                    start_time=current_start_time,
                    duration=chunk_duration_actual,
                    size_bytes=os.path.getsize(chunk_path)
                )
                
                chunks.append(chunk_info)
                current_start_time += chunk_duration_actual
                chunk_index += 1
                
                self.logger.debug("Chunk created", 
                                job_id=job_id,
                                chunk_index=chunk_index,
                                chunk_path=chunk_path,
                                duration=chunk_duration_actual,
                                size_mb=chunk_info.size_mb)
            
            self.logger.info("WAV chunking complete", 
                           job_id=job_id,
                           chunk_count=len(chunks),
                           total_duration=sum(c.duration for c in chunks),
                           total_size_mb=sum(c.size_mb for c in chunks))
            
            return chunks
            
        except Exception as e:
            self.logger.error("WAV chunking failed", 
                            wav_path=wav_path,
                            job_id=job_id,
                            error=str(e))
            raise


# Global FFmpeg service instance
_ffmpeg_service: Optional[FFmpegService] = None


def init_ffmpeg_service(settings: Settings) -> FFmpegService:
    """Initialize FFmpeg service."""
    global _ffmpeg_service
    logger = get_logger("ffmpeg_init")
    
    try:
        _ffmpeg_service = FFmpegService(settings)
        logger.info("FFmpeg service initialized successfully")
        return _ffmpeg_service
    except Exception as e:
        logger.error("Failed to initialize FFmpeg service", error=str(e))
        raise


def get_ffmpeg_service() -> FFmpegService:
    """Get the global FFmpeg service instance."""
    if _ffmpeg_service is None:
        raise RuntimeError("FFmpeg service not initialized")
    return _ffmpeg_service