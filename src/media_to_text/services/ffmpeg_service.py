"""FFmpeg service for media processing via Docker container."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from media_to_text.config import Settings


class MediaInfo:
    """Media file information extracted by FFprobe."""
    
    def __init__(self, data: Dict):
        self.data = data
        self.streams = data.get("streams", [])
        self.format_info = data.get("format", {})
    
    @property
    def duration(self) -> float:
        """Get media duration in seconds."""
        duration = self.format_info.get("duration")
        if duration:
            return float(duration)
        
        # Try to get from first stream if format doesn't have it
        for stream in self.streams:
            if stream.get("duration"):
                return float(stream["duration"])
        
        return 0.0
    
    @property
    def has_audio(self) -> bool:
        """Check if media has audio streams."""
        return any(stream.get("codec_type") == "audio" for stream in self.streams)
    
    @property
    def audio_codec(self) -> Optional[str]:
        """Get audio codec name."""
        for stream in self.streams:
            if stream.get("codec_type") == "audio":
                return stream.get("codec_name")
        return None
    
    @property
    def sample_rate(self) -> Optional[int]:
        """Get audio sample rate."""
        for stream in self.streams:
            if stream.get("codec_type") == "audio":
                rate = stream.get("sample_rate")
                return int(rate) if rate else None
        return None
    
    @property
    def bit_depth(self) -> Optional[int]:
        """Get audio bit depth."""
        for stream in self.streams:
            if stream.get("codec_type") == "audio":
                # Try to extract from sample_fmt
                sample_fmt = stream.get("sample_fmt", "")
                if "s16" in sample_fmt:
                    return 16
                elif "s24" in sample_fmt:
                    return 24
                elif "s32" in sample_fmt:
                    return 32
        return None
    
    @property
    def needs_conversion(self) -> bool:
        """Check if media needs conversion to 16-bit PCM WAV."""
        # Check if it's already WAV format
        format_name = self.format_info.get("format_name", "").lower()
        if "wav" not in format_name:
            return True
        
        # Check if audio codec is PCM and 16-bit
        codec = self.audio_codec
        bit_depth = self.bit_depth
        
        if codec != "pcm_s16le" or bit_depth != 16:
            return True
        
        return False


class ChunkInfo:
    """Information about a media chunk."""
    
    def __init__(self, file_path: str, index: int, start_time: float = 0.0, duration: float = 0.0):
        self.file_path = file_path
        self.index = index
        self.start_time = start_time
        self.duration = duration
        self.size_bytes = 0
    
    @property
    def size_mb(self) -> float:
        """Get chunk size in MB."""
        if self.size_bytes == 0 and os.path.exists(self.file_path):
            self.size_bytes = os.path.getsize(self.file_path)
        return self.size_bytes / (1024 * 1024)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "file_path": self.file_path,
            "index": self.index,
            "start_time": self.start_time,
            "duration": self.duration,
            "size_mb": self.size_mb
        }


class FFmpegService:
    """Service for FFmpeg operations via Docker container."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.container_name = "media-to-text-ffmpeg"
    
    async def _run_docker_exec(self, command: list[str], cwd: str = "/tmp") -> Tuple[str, str, int]:
        """Execute command in FFmpeg container."""
        docker_cmd = [
            "docker", "exec", "-w", cwd, self.container_name
        ] + command
        
        try:
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            return (
                stdout.decode('utf-8', errors='ignore'),
                stderr.decode('utf-8', errors='ignore'),
                process.returncode or 0
            )
        except Exception as e:
            return "", f"Docker exec failed: {str(e)}", 1
    
    async def probe_media(self, file_path: str) -> MediaInfo:
        """Probe media file using FFprobe."""
        # FFprobe command to get JSON output
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            file_path
        ]
        
        stdout, stderr, returncode = await self._run_docker_exec(cmd)
        
        if returncode != 0:
            raise RuntimeError(f"FFprobe failed: {stderr}")
        
        try:
            probe_data = json.loads(stdout)
            return MediaInfo(probe_data)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse FFprobe output: {e}")
    
    async def convert_to_wav(self, input_path: str, output_path: str) -> None:
        """Convert media file to 16-bit PCM WAV."""
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # FFmpeg command for conversion
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar", "16000",          # 16kHz sample rate (good for speech)
            "-ac", "1",              # Mono
            "-y",                    # Overwrite output file
            output_path
        ]
        
        stdout, stderr, returncode = await self._run_docker_exec(cmd)
        
        if returncode != 0:
            raise RuntimeError(f"FFmpeg conversion failed: {stderr}")
        
        # Verify output file was created
        if not os.path.exists(output_path):
            raise RuntimeError(f"Output file was not created: {output_path}")
    
    async def chunk_wav_with_segments(
        self, 
        input_path: str, 
        output_dir: str, 
        max_size_mb: float = 25.0
    ) -> List[ChunkInfo]:
        """
        Chunk WAV file using FFmpeg's segment feature.
        
        Uses the command: ffmpeg -i input.wav -f segment -segment_time [calculated] -fs 25M /tmp/{job-id}/chunk_%03d.wav
        
        Args:
            input_path: Path to input WAV file
            output_dir: Directory to store chunks
            max_size_mb: Maximum chunk size in MB
        
        Returns:
            List of ChunkInfo objects for created chunks
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # First, probe the media to get duration and estimate bitrate
        media_info = await self.probe_media(input_path)
        duration = media_info.duration
        
        if duration <= 0:
            raise RuntimeError("Could not determine media duration")
        
        # Get file size to estimate bitrate
        file_size_mb = await self.get_file_size_mb(input_path)
        
        # Calculate segment time to stay under size limit
        # Bitrate estimate: file_size_mb / duration_minutes
        duration_minutes = duration / 60
        estimated_bitrate_mb_per_min = file_size_mb / duration_minutes if duration_minutes > 0 else 1.9
        
        # Calculate segment time to stay under max_size_mb
        segment_time_minutes = max_size_mb / estimated_bitrate_mb_per_min
        segment_time_seconds = max(60, segment_time_minutes * 60)  # Minimum 1 minute chunks
        
        # Ensure we don't create too many tiny chunks for short files
        if duration < segment_time_seconds:
            segment_time_seconds = duration
        
        # Output pattern for chunks
        chunk_pattern = os.path.join(output_dir, "chunk_%03d.wav")
        
        # FFmpeg segment command
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-f", "segment",
            "-segment_time", str(segment_time_seconds),
            "-fs", f"{int(max_size_mb)}M",  # File size limit
            "-acodec", "pcm_s16le",        # Maintain 16-bit PCM
            "-ar", "16000",                # Maintain sample rate
            "-ac", "1",                    # Maintain mono
            "-y",                          # Overwrite files
            chunk_pattern
        ]
        
        stdout, stderr, returncode = await self._run_docker_exec(cmd)
        
        if returncode != 0:
            raise RuntimeError(f"FFmpeg segmentation failed: {stderr}")
        
        # Discover created chunks
        chunks = []
        chunk_index = 0
        
        while True:
            chunk_file = os.path.join(output_dir, f"chunk_{chunk_index:03d}.wav")
            if not os.path.exists(chunk_file):
                break
            
            # Create chunk info
            start_time = chunk_index * segment_time_seconds
            chunk_info = ChunkInfo(
                file_path=chunk_file,
                index=chunk_index,
                start_time=start_time,
                duration=min(segment_time_seconds, duration - start_time)
            )
            
            chunks.append(chunk_info)
            chunk_index += 1
        
        if not chunks:
            raise RuntimeError("No chunks were created")
        
        return chunks
    
    async def extract_audio_chunks(
        self, 
        input_path: str, 
        output_dir: str, 
        chunk_duration: float = 300.0  # 5 minutes
    ) -> list[str]:
        """
        Extract audio into chunks for processing (legacy method).
        
        Note: This method is kept for backward compatibility.
        For new implementations, use chunk_wav_with_segments().
        """
        # First, probe the media to get duration
        media_info = await self.probe_media(input_path)
        total_duration = media_info.duration
        
        if total_duration <= 0:
            raise RuntimeError("Could not determine media duration")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate number of chunks
        num_chunks = int((total_duration + chunk_duration - 1) // chunk_duration)
        chunk_files = []
        
        # Extract each chunk
        for i in range(num_chunks):
            start_time = i * chunk_duration
            chunk_file = os.path.join(output_dir, f"chunk_{i:04d}.wav")
            
            cmd = [
                "ffmpeg",
                "-i", input_path,
                "-ss", str(start_time),
                "-t", str(chunk_duration),
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                "-y",
                chunk_file
            ]
            
            stdout, stderr, returncode = await self._run_docker_exec(cmd)
            
            if returncode != 0:
                raise RuntimeError(f"Failed to extract chunk {i}: {stderr}")
            
            if os.path.exists(chunk_file):
                chunk_files.append(chunk_file)
        
        return chunk_files
    
    async def validate_chunk_sizes(self, chunks: List[ChunkInfo], max_size_mb: float = 25.0) -> bool:
        """Validate that all chunks are within the size limit."""
        for chunk in chunks:
            if chunk.size_mb > max_size_mb:
                return False
        return True
    
    async def get_chunk_info(self, chunk_files: List[str]) -> List[ChunkInfo]:
        """Get detailed information about chunks."""
        chunks = []
        for i, chunk_file in enumerate(chunk_files):
            chunk_info = ChunkInfo(
                file_path=chunk_file,
                index=i
            )
            
            # Get file size
            if os.path.exists(chunk_file):
                chunk_info.size_bytes = os.path.getsize(chunk_file)
            
            chunks.append(chunk_info)
        
        return chunks
    
    async def get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB."""
        try:
            size_bytes = os.path.getsize(file_path)
            return size_bytes / (1024 * 1024)
        except OSError:
            return 0.0


# Global FFmpeg service instance
ffmpeg_service: Optional[FFmpegService] = None


def get_ffmpeg_service() -> FFmpegService:
    """Get FFmpeg service instance."""
    global ffmpeg_service
    if ffmpeg_service is None:
        from media_to_text.config import Settings
        settings = Settings()
        ffmpeg_service = FFmpegService(settings)
    return ffmpeg_service