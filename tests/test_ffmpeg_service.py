"""Unit tests for FFmpeg service."""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from media_to_text.services.ffmpeg_service import FFmpegService, MediaInfo, ChunkInfo


class TestFFmpegService:
    """Test suite for FFmpeg service."""

    @pytest.fixture
    def ffmpeg_service(self):
        """Create FFmpeg service instance."""
        return FFmpegService()

    @pytest.fixture
    def sample_media_info(self):
        """Sample media info for testing."""
        return MediaInfo(
            has_audio=True,
            duration=120.5,
            format="mp3",
            needs_conversion=True,
            bit_rate=128000,
            sample_rate=44100,
            channels=2
        )

    @pytest.fixture
    def sample_chunk_info(self):
        """Sample chunk info for testing."""
        return ChunkInfo(
            file_path="/tmp/chunk_0.wav",
            index=0,
            start_time=0.0,
            duration=30.0,
            size_bytes=1024000
        )

    @pytest.fixture
    def temp_files(self):
        """Create temporary files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "input.mp3")
            output_file = os.path.join(temp_dir, "output.wav")
            
            # Create dummy files
            Path(input_file).touch()
            
            yield {
                "temp_dir": temp_dir,
                "input_file": input_file,
                "output_file": output_file
            }

    @pytest.mark.asyncio
    async def test_probe_media_success(self, ffmpeg_service):
        """Test successful media probing."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = '''
        {
            "streams": [
                {
                    "codec_type": "audio",
                    "duration": "120.5",
                    "bit_rate": "128000",
                    "sample_rate": "44100",
                    "channels": 2
                }
            ],
            "format": {
                "format_name": "mp3"
            }
        }
        '''
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_result):
            with patch.object(mock_result, 'communicate', return_value=(mock_result.stdout, "")):
                media_info = await ffmpeg_service.probe_media("/path/to/file.mp3")
                
                assert media_info.has_audio is True
                assert media_info.duration == 120.5
                assert media_info.format == "mp3"
                assert media_info.needs_conversion is True
                assert media_info.bit_rate == 128000
                assert media_info.sample_rate == 44100
                assert media_info.channels == 2

    @pytest.mark.asyncio
    async def test_probe_media_no_audio(self, ffmpeg_service):
        """Test probing media with no audio streams."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = '''
        {
            "streams": [
                {
                    "codec_type": "video",
                    "duration": "120.5"
                }
            ],
            "format": {
                "format_name": "mp4"
            }
        }
        '''
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_result):
            with patch.object(mock_result, 'communicate', return_value=(mock_result.stdout, "")):
                media_info = await ffmpeg_service.probe_media("/path/to/file.mp4")
                
                assert media_info.has_audio is False
                assert media_info.duration == 120.5
                assert media_info.format == "mp4"

    @pytest.mark.asyncio
    async def test_probe_media_already_wav(self, ffmpeg_service):
        """Test probing WAV file that doesn't need conversion."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = '''
        {
            "streams": [
                {
                    "codec_type": "audio",
                    "duration": "60.0",
                    "sample_rate": "44100",
                    "channels": 2
                }
            ],
            "format": {
                "format_name": "wav"
            }
        }
        '''
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_result):
            with patch.object(mock_result, 'communicate', return_value=(mock_result.stdout, "")):
                media_info = await ffmpeg_service.probe_media("/path/to/file.wav")
                
                assert media_info.has_audio is True
                assert media_info.needs_conversion is False
                assert media_info.format == "wav"

    @pytest.mark.asyncio
    async def test_probe_media_ffmpeg_error(self, ffmpeg_service):
        """Test handling FFmpeg probe errors."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "FFmpeg error: Invalid file format"
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_result):
            with patch.object(mock_result, 'communicate', return_value=("", mock_result.stderr)):
                with pytest.raises(RuntimeError, match="FFmpeg probe failed"):
                    await ffmpeg_service.probe_media("/path/to/invalid.file")

    @pytest.mark.asyncio
    async def test_convert_to_wav_success(self, ffmpeg_service, temp_files):
        """Test successful WAV conversion."""
        mock_result = Mock()
        mock_result.returncode = 0
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_result):
            with patch.object(mock_result, 'communicate', return_value=("", "")):
                await ffmpeg_service.convert_to_wav(
                    temp_files["input_file"],
                    temp_files["output_file"]
                )
                
                # Should not raise an exception

    @pytest.mark.asyncio
    async def test_convert_to_wav_error(self, ffmpeg_service, temp_files):
        """Test handling WAV conversion errors."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Conversion failed"
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_result):
            with patch.object(mock_result, 'communicate', return_value=("", mock_result.stderr)):
                with pytest.raises(RuntimeError, match="FFmpeg conversion failed"):
                    await ffmpeg_service.convert_to_wav(
                        temp_files["input_file"],
                        temp_files["output_file"]
                    )

    @pytest.mark.asyncio
    async def test_get_file_size_mb(self, ffmpeg_service, temp_files):
        """Test file size calculation."""
        # Create a file with known size
        test_content = b"x" * 2048000  # 2MB
        with open(temp_files["input_file"], "wb") as f:
            f.write(test_content)
        
        size_mb = await ffmpeg_service.get_file_size_mb(temp_files["input_file"])
        
        # Should be approximately 2MB (allowing for small differences)
        assert 1.9 <= size_mb <= 2.1

    @pytest.mark.asyncio
    async def test_chunk_wav_with_segments_small_file(self, ffmpeg_service, temp_files):
        """Test chunking with small file (no chunking needed)."""
        # Mock small file
        with patch.object(ffmpeg_service, 'get_file_size_mb', return_value=5.0):
            chunks = await ffmpeg_service.chunk_wav_with_segments(
                temp_files["input_file"],
                temp_files["temp_dir"],
                max_chunk_size_mb=25.0
            )
            
            assert len(chunks) == 1
            assert chunks[0].index == 0
            assert chunks[0].file_path == temp_files["input_file"]

    @pytest.mark.asyncio
    async def test_chunk_wav_with_segments_large_file(self, ffmpeg_service, temp_files):
        """Test chunking with large file."""
        mock_result = Mock()
        mock_result.returncode = 0
        
        # Mock large file that needs chunking
        with patch.object(ffmpeg_service, 'get_file_size_mb', return_value=50.0):
            with patch('asyncio.create_subprocess_exec', return_value=mock_result):
                with patch.object(mock_result, 'communicate', return_value=("", "")):
                    with patch('os.path.exists', return_value=True):
                        with patch('os.path.getsize', return_value=10240000):  # 10MB per chunk
                            chunks = await ffmpeg_service.chunk_wav_with_segments(
                                temp_files["input_file"],
                                temp_files["temp_dir"],
                                max_chunk_size_mb=25.0
                            )
                            
                            # Should create chunks based on our mocked setup
                            assert len(chunks) >= 1
                            assert all(chunk.file_path.endswith('.wav') for chunk in chunks)

    @pytest.mark.asyncio
    async def test_validate_chunk_sizes_all_valid(self, ffmpeg_service):
        """Test chunk size validation with all valid chunks."""
        chunks = [
            ChunkInfo("/path/chunk1.wav", 0, 0.0, 30.0, 20 * 1024 * 1024),  # 20MB
            ChunkInfo("/path/chunk2.wav", 1, 30.0, 30.0, 15 * 1024 * 1024),  # 15MB
        ]
        
        is_valid = await ffmpeg_service.validate_chunk_sizes(chunks, 25.0)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_chunk_sizes_some_invalid(self, ffmpeg_service):
        """Test chunk size validation with some invalid chunks."""
        chunks = [
            ChunkInfo("/path/chunk1.wav", 0, 0.0, 30.0, 20 * 1024 * 1024),  # 20MB
            ChunkInfo("/path/chunk2.wav", 1, 30.0, 30.0, 30 * 1024 * 1024),  # 30MB - too big
        ]
        
        is_valid = await ffmpeg_service.validate_chunk_sizes(chunks, 25.0)
        assert is_valid is False

    def test_chunk_info_to_dict(self, sample_chunk_info):
        """Test ChunkInfo to_dict method."""
        chunk_dict = sample_chunk_info.to_dict()
        
        expected = {
            "file_path": "/tmp/chunk_0.wav",
            "index": 0,
            "start_time": 0.0,
            "duration": 30.0,
            "size_bytes": 1024000,
            "size_mb": 1024000 / (1024 * 1024)
        }
        
        assert chunk_dict == expected

    def test_chunk_info_size_mb_property(self, sample_chunk_info):
        """Test ChunkInfo size_mb property."""
        assert sample_chunk_info.size_mb == 1024000 / (1024 * 1024)

    def test_media_info_initialization(self):
        """Test MediaInfo initialization with defaults."""
        media_info = MediaInfo(
            has_audio=True,
            duration=60.0,
            format="mp3"
        )
        
        assert media_info.has_audio is True
        assert media_info.duration == 60.0
        assert media_info.format == "mp3"
        assert media_info.needs_conversion is True  # Default for non-WAV
        assert media_info.bit_rate == 0  # Default
        assert media_info.sample_rate == 0  # Default
        assert media_info.channels == 0  # Default

    def test_media_info_wav_no_conversion(self):
        """Test MediaInfo for WAV files that don't need conversion."""
        media_info = MediaInfo(
            has_audio=True,
            duration=60.0,
            format="wav"
        )
        
        assert media_info.needs_conversion is False

    @pytest.mark.asyncio
    async def test_ffmpeg_command_execution_timeout(self, ffmpeg_service):
        """Test FFmpeg command timeout handling."""
        # Mock a process that hangs
        mock_process = Mock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError()):
                with pytest.raises(RuntimeError, match="FFmpeg operation timed out"):
                    await ffmpeg_service.probe_media("/path/to/file.mp3")

    @pytest.mark.asyncio
    async def test_chunk_creation_with_different_segment_sizes(self, ffmpeg_service, temp_files):
        """Test chunk creation with different segment sizes."""
        mock_result = Mock()
        mock_result.returncode = 0
        
        # Test with 10MB max chunk size
        with patch.object(ffmpeg_service, 'get_file_size_mb', return_value=30.0):
            with patch('asyncio.create_subprocess_exec', return_value=mock_result):
                with patch.object(mock_result, 'communicate', return_value=("", "")):
                    with patch('os.path.exists', return_value=True):
                        with patch('os.path.getsize', return_value=5 * 1024 * 1024):  # 5MB per chunk
                            chunks = await ffmpeg_service.chunk_wav_with_segments(
                                temp_files["input_file"],
                                temp_files["temp_dir"],
                                max_chunk_size_mb=10.0
                            )
                            
                            # Should handle smaller chunk size appropriately
                            assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_error_handling_invalid_json_probe(self, ffmpeg_service):
        """Test handling of invalid JSON from FFmpeg probe."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Invalid JSON output"
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_result):
            with patch.object(mock_result, 'communicate', return_value=(mock_result.stdout, "")):
                with pytest.raises(RuntimeError, match="Failed to parse FFmpeg probe output"):
                    await ffmpeg_service.probe_media("/path/to/file.mp3")

    @pytest.mark.asyncio
    async def test_chunk_directory_creation(self, ffmpeg_service, temp_files):
        """Test that chunk directory is created if it doesn't exist."""
        chunks_dir = os.path.join(temp_files["temp_dir"], "chunks")
        
        # Ensure directory doesn't exist initially
        assert not os.path.exists(chunks_dir)
        
        mock_result = Mock()
        mock_result.returncode = 0
        
        with patch.object(ffmpeg_service, 'get_file_size_mb', return_value=30.0):
            with patch('asyncio.create_subprocess_exec', return_value=mock_result):
                with patch.object(mock_result, 'communicate', return_value=("", "")):
                    with patch('os.path.exists', side_effect=lambda x: x != chunks_dir):
                        with patch('os.makedirs') as mock_makedirs:
                            await ffmpeg_service.chunk_wav_with_segments(
                                temp_files["input_file"],
                                chunks_dir,
                                max_chunk_size_mb=25.0
                            )
                            
                            # Should create the directory
                            mock_makedirs.assert_called_once_with(chunks_dir, exist_ok=True)