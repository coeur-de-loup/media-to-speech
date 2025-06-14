"""Unit tests for transcript service."""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from media_to_text.services.transcript_service import TranscriptProcessor
from media_to_text.services.ffmpeg_service import ChunkInfo
from media_to_text.services.openai_service import TranscriptionResult


class TestTranscriptProcessor:
    """Test suite for transcript processor."""

    @pytest.fixture
    def transcript_processor(self):
        """Create transcript processor instance."""
        return TranscriptProcessor()

    @pytest.fixture
    def sample_chunk_infos(self):
        """Sample chunk information for testing."""
        return [
            ChunkInfo("/tmp/chunk_0.wav", 0, 0.0, 30.0, 1024000),
            ChunkInfo("/tmp/chunk_1.wav", 1, 30.0, 30.0, 1024000),
            ChunkInfo("/tmp/chunk_2.wav", 2, 60.0, 25.5, 1024000)
        ]

    @pytest.fixture
    def sample_transcription_results(self):
        """Sample transcription results for testing."""
        return [
            TranscriptionResult(
                chunk_index=0,
                success=True,
                text="Hello, this is the first chunk of audio.",
                processing_time=2.5,
                retry_count=0,
                segments=[
                    {"start": 0.0, "end": 2.0, "text": "Hello, this is"},
                    {"start": 2.0, "end": 4.5, "text": "the first chunk of audio."}
                ]
            ),
            TranscriptionResult(
                chunk_index=1,
                success=True,
                text="This is the second chunk with more content.",
                processing_time=3.1,
                retry_count=0,
                segments=[
                    {"start": 0.0, "end": 3.0, "text": "This is the second chunk"},
                    {"start": 3.0, "end": 5.5, "text": "with more content."}
                ]
            ),
            TranscriptionResult(
                chunk_index=2,
                success=True,
                text="Final chunk of the transcription.",
                processing_time=2.8,
                retry_count=1,
                segments=[
                    {"start": 0.0, "end": 4.0, "text": "Final chunk of the transcription."}
                ]
            )
        ]

    @pytest.fixture
    def sample_basic_metadata(self):
        """Sample basic metadata for testing."""
        return {
            "total_processing_time": 8.4,
            "success_rate": 1.0,
            "total_chunks": 3,
            "successful_chunks": 3,
            "failed_chunks": 0
        }

    @pytest.fixture
    def temp_output_file(self):
        """Create temporary output file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            yield f.name
        # Cleanup
        if os.path.exists(f.name):
            os.unlink(f.name)

    def test_process_complete_transcript_success(
        self, 
        transcript_processor, 
        sample_transcription_results, 
        sample_chunk_infos,
        sample_basic_metadata,
        temp_output_file
    ):
        """Test successful complete transcript processing."""
        result = transcript_processor.process_complete_transcript(
            transcription_results=sample_transcription_results,
            chunk_infos=sample_chunk_infos,
            original_metadata=sample_basic_metadata,
            output_path=temp_output_file
        )
        
        # Verify structure
        assert "text" in result
        assert "segments" in result
        assert "metadata" in result
        
        # Verify combined text
        expected_text = (
            "Hello, this is the first chunk of audio. "
            "This is the second chunk with more content. "
            "Final chunk of the transcription."
        )
        assert result["text"] == expected_text
        
        # Verify segments with normalized timestamps
        segments = result["segments"]
        assert len(segments) == 6  # 2 + 2 + 1 + 1 segments
        
        # First chunk segments (no offset)
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 2.0
        assert segments[1]["start"] == 2.0
        assert segments[1]["end"] == 4.5
        
        # Second chunk segments (30s offset)
        assert segments[2]["start"] == 30.0
        assert segments[2]["end"] == 33.0
        assert segments[3]["start"] == 33.0
        assert segments[3]["end"] == 35.5
        
        # Third chunk segments (60s offset)
        assert segments[4]["start"] == 60.0
        assert segments[4]["end"] == 64.0
        
        # Verify metadata
        metadata = result["metadata"]
        assert metadata["total_duration"] == 85.5  # 60 + 25.5
        assert metadata["total_chunks"] == 3
        assert metadata["processing_stats"]["total_processing_time"] == 8.4
        
        # Verify file was written
        assert os.path.exists(temp_output_file)

    def test_process_complete_transcript_with_failed_chunks(
        self,
        transcript_processor,
        sample_chunk_infos,
        sample_basic_metadata,
        temp_output_file
    ):
        """Test transcript processing with some failed chunks."""
        # Mix of successful and failed results
        mixed_results = [
            TranscriptionResult(
                chunk_index=0,
                success=True,
                text="First chunk succeeded.",
                processing_time=2.0,
                retry_count=0,
                segments=[{"start": 0.0, "end": 3.0, "text": "First chunk succeeded."}]
            ),
            TranscriptionResult(
                chunk_index=1,
                success=False,
                text="",
                processing_time=0.0,
                retry_count=3,
                error_message="OpenAI API error",
                segments=[]
            ),
            TranscriptionResult(
                chunk_index=2,
                success=True,
                text="Third chunk also succeeded.",
                processing_time=2.5,
                retry_count=1,
                segments=[{"start": 0.0, "end": 4.0, "text": "Third chunk also succeeded."}]
            )
        ]
        
        result = transcript_processor.process_complete_transcript(
            transcription_results=mixed_results,
            chunk_infos=sample_chunk_infos,
            original_metadata=sample_basic_metadata,
            output_path=temp_output_file
        )
        
        # Should combine only successful chunks
        expected_text = "First chunk succeeded. Third chunk also succeeded."
        assert result["text"] == expected_text
        
        # Should have segments from successful chunks only
        segments = result["segments"]
        assert len(segments) == 2
        
        # First chunk (index 0)
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 3.0
        
        # Third chunk (index 2) with 60s offset
        assert segments[1]["start"] == 60.0
        assert segments[1]["end"] == 64.0
        
        # Metadata should reflect mixed results
        metadata = result["metadata"]
        assert metadata["chunk_details"]["successful_chunks"] == 2
        assert metadata["chunk_details"]["failed_chunks"] == 1

    def test_process_complete_transcript_no_segments(
        self,
        transcript_processor,
        sample_chunk_infos,
        sample_basic_metadata,
        temp_output_file
    ):
        """Test transcript processing when OpenAI doesn't return segments."""
        # Results without detailed segments
        results_no_segments = [
            TranscriptionResult(
                chunk_index=0,
                success=True,
                text="Simple transcription without segments.",
                processing_time=2.0,
                retry_count=0,
                segments=None  # No segments provided
            ),
            TranscriptionResult(
                chunk_index=1,
                success=True,
                text="Another chunk without segments.",
                processing_time=2.5,
                retry_count=0,
                segments=[]  # Empty segments
            )
        ]
        
        result = transcript_processor.process_complete_transcript(
            transcription_results=results_no_segments,
            chunk_infos=sample_chunk_infos[:2],  # Only first 2 chunks
            original_metadata=sample_basic_metadata,
            output_path=temp_output_file
        )
        
        # Should still combine text
        expected_text = "Simple transcription without segments. Another chunk without segments."
        assert result["text"] == expected_text
        
        # Should create basic segments from chunk boundaries
        segments = result["segments"]
        assert len(segments) == 2
        
        # Should use chunk-level timing
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 30.0  # Duration of first chunk
        assert segments[0]["text"] == "Simple transcription without segments."
        
        assert segments[1]["start"] == 30.0  # Start of second chunk
        assert segments[1]["end"] == 60.0   # End of second chunk
        assert segments[1]["text"] == "Another chunk without segments."

    def test_normalize_timestamps(self, transcript_processor, sample_chunk_infos):
        """Test timestamp normalization logic."""
        # Sample segments from different chunks
        segments_chunk_0 = [
            {"start": 0.0, "end": 2.0, "text": "First segment"},
            {"start": 2.0, "end": 4.0, "text": "Second segment"}
        ]
        
        segments_chunk_1 = [
            {"start": 0.0, "end": 1.5, "text": "Third segment"},
            {"start": 1.5, "end": 3.0, "text": "Fourth segment"}
        ]
        
        # Normalize with chunk offset
        normalized_0 = transcript_processor._normalize_timestamps(
            segments_chunk_0, sample_chunk_infos[0]
        )
        normalized_1 = transcript_processor._normalize_timestamps(
            segments_chunk_1, sample_chunk_infos[1]
        )
        
        # First chunk should have no offset (starts at 0.0)
        assert normalized_0[0]["start"] == 0.0
        assert normalized_0[0]["end"] == 2.0
        assert normalized_0[1]["start"] == 2.0
        assert normalized_0[1]["end"] == 4.0
        
        # Second chunk should have 30s offset
        assert normalized_1[0]["start"] == 30.0
        assert normalized_1[0]["end"] == 31.5
        assert normalized_1[1]["start"] == 31.5
        assert normalized_1[1]["end"] == 33.0

    def test_combine_text_from_results(self, transcript_processor, sample_transcription_results):
        """Test text combination from transcription results."""
        text = transcript_processor._combine_text_from_results(sample_transcription_results)
        
        expected = (
            "Hello, this is the first chunk of audio. "
            "This is the second chunk with more content. "
            "Final chunk of the transcription."
        )
        
        assert text == expected

    def test_combine_text_with_failed_results(self, transcript_processor):
        """Test text combination with failed results."""
        mixed_results = [
            TranscriptionResult(0, True, "Success 1", 1.0, 0),
            TranscriptionResult(1, False, "", 0.0, 3, error_message="Failed"),
            TranscriptionResult(2, True, "Success 2", 1.5, 1)
        ]
        
        text = transcript_processor._combine_text_from_results(mixed_results)
        
        # Should only include successful results
        assert text == "Success 1 Success 2"

    def test_create_metadata(self, transcript_processor, sample_chunk_infos, sample_basic_metadata):
        """Test metadata creation."""
        metadata = transcript_processor._create_metadata(
            chunk_infos=sample_chunk_infos,
            original_metadata=sample_basic_metadata,
            successful_chunks=3,
            failed_chunks=0
        )
        
        assert metadata["total_duration"] == 85.5  # Sum of chunk durations
        assert metadata["total_chunks"] == 3
        assert metadata["chunk_details"]["successful_chunks"] == 3
        assert metadata["chunk_details"]["failed_chunks"] == 0
        assert metadata["processing_stats"]["total_processing_time"] == 8.4
        assert "created_at" in metadata

    def test_save_transcript_file(self, transcript_processor, temp_output_file):
        """Test saving transcript to file."""
        sample_transcript = {
            "text": "Test transcript",
            "segments": [{"start": 0.0, "end": 2.0, "text": "Test"}],
            "metadata": {"total_duration": 2.0}
        }
        
        transcript_processor._save_transcript_file(sample_transcript, temp_output_file)
        
        # Verify file was created and contains correct data
        assert os.path.exists(temp_output_file)
        
        with open(temp_output_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data == sample_transcript

    def test_save_transcript_file_error_handling(self, transcript_processor):
        """Test error handling when saving transcript file fails."""
        sample_transcript = {"text": "Test"}
        invalid_path = "/invalid/path/transcript.json"
        
        # Should not raise exception but log error
        transcript_processor._save_transcript_file(sample_transcript, invalid_path)
        
        # File should not exist
        assert not os.path.exists(invalid_path)

    def test_empty_transcription_results(
        self, 
        transcript_processor, 
        sample_chunk_infos, 
        sample_basic_metadata,
        temp_output_file
    ):
        """Test processing with empty transcription results."""
        result = transcript_processor.process_complete_transcript(
            transcription_results=[],
            chunk_infos=sample_chunk_infos,
            original_metadata=sample_basic_metadata,
            output_path=temp_output_file
        )
        
        assert result["text"] == ""
        assert result["segments"] == []
        assert result["metadata"]["chunk_details"]["successful_chunks"] == 0
        assert result["metadata"]["chunk_details"]["failed_chunks"] == 0

    def test_mismatched_chunks_and_results(
        self,
        transcript_processor,
        sample_chunk_infos,
        sample_basic_metadata,
        temp_output_file
    ):
        """Test handling when chunk count doesn't match result count."""
        # Only 2 results for 3 chunks
        partial_results = [
            TranscriptionResult(0, True, "First chunk", 1.0, 0, segments=[
                {"start": 0.0, "end": 2.0, "text": "First chunk"}
            ]),
            TranscriptionResult(1, True, "Second chunk", 1.0, 0, segments=[
                {"start": 0.0, "end": 2.0, "text": "Second chunk"}
            ])
        ]
        
        result = transcript_processor.process_complete_transcript(
            transcription_results=partial_results,
            chunk_infos=sample_chunk_infos,
            original_metadata=sample_basic_metadata,
            output_path=temp_output_file
        )
        
        # Should handle gracefully
        assert "First chunk Second chunk" in result["text"]
        assert len(result["segments"]) == 2

    def test_segment_overlap_handling(self, transcript_processor):
        """Test handling of overlapping segments."""
        chunk_info = ChunkInfo("/tmp/chunk.wav", 0, 0.0, 30.0, 1024000)
        
        # Segments with slight overlap
        overlapping_segments = [
            {"start": 0.0, "end": 2.5, "text": "First segment"},
            {"start": 2.0, "end": 4.0, "text": "Overlapping segment"},  # Overlaps with first
            {"start": 4.0, "end": 6.0, "text": "Third segment"}
        ]
        
        normalized = transcript_processor._normalize_timestamps(overlapping_segments, chunk_info)
        
        # Should preserve original timestamps (no automatic overlap resolution)
        assert normalized[0]["end"] == 2.5
        assert normalized[1]["start"] == 2.0
        assert normalized[1]["end"] == 4.0

    def test_large_transcript_processing(
        self,
        transcript_processor,
        temp_output_file
    ):
        """Test processing of large transcript with many chunks."""
        # Create large number of chunks and results
        num_chunks = 100
        
        chunk_infos = [
            ChunkInfo(f"/tmp/chunk_{i}.wav", i, i * 30.0, 30.0, 1024000)
            for i in range(num_chunks)
        ]
        
        transcription_results = [
            TranscriptionResult(
                chunk_index=i,
                success=True,
                text=f"Content for chunk {i}.",
                processing_time=1.0,
                retry_count=0,
                segments=[{
                    "start": 0.0,
                    "end": 2.0,
                    "text": f"Content for chunk {i}."
                }]
            )
            for i in range(num_chunks)
        ]
        
        basic_metadata = {
            "total_processing_time": 100.0,
            "success_rate": 1.0,
            "total_chunks": num_chunks,
            "successful_chunks": num_chunks,
            "failed_chunks": 0
        }
        
        result = transcript_processor.process_complete_transcript(
            transcription_results=transcription_results,
            chunk_infos=chunk_infos,
            original_metadata=basic_metadata,
            output_path=temp_output_file
        )
        
        # Should handle large transcript
        assert len(result["segments"]) == num_chunks
        assert result["metadata"]["total_chunks"] == num_chunks
        assert result["metadata"]["total_duration"] == 3000.0  # 100 * 30s
        
        # Verify timestamp progression
        for i in range(min(10, num_chunks)):  # Check first 10
            expected_start = i * 30.0
            assert result["segments"][i]["start"] == expected_start

    def test_unicode_text_handling(
        self,
        transcript_processor,
        sample_chunk_infos,
        sample_basic_metadata,
        temp_output_file
    ):
        """Test handling of Unicode text in transcriptions."""
        unicode_results = [
            TranscriptionResult(
                chunk_index=0,
                success=True,
                text="Hello 你好 こんにちは مرحبا",
                processing_time=2.0,
                retry_count=0,
                segments=[{
                    "start": 0.0,
                    "end": 3.0,
                    "text": "Hello 你好 こんにちは مرحبا"
                }]
            )
        ]
        
        result = transcript_processor.process_complete_transcript(
            transcription_results=unicode_results,
            chunk_infos=sample_chunk_infos[:1],
            original_metadata=sample_basic_metadata,
            output_path=temp_output_file
        )
        
        # Should handle Unicode correctly
        assert "你好" in result["text"]
        assert "こんにちは" in result["text"]
        assert "مرحبا" in result["text"]
        
        # File should be saved with Unicode support
        assert os.path.exists(temp_output_file)
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
            assert "你好" in saved_data["text"]