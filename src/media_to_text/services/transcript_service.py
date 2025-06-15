"""Transcript processing service for timestamp normalization and aggregation."""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from media_to_text.services.ffmpeg_service import ChunkInfo
from media_to_text.services.openai_service import TranscriptionResult


@dataclass
class TranscriptChunk:
    """Normalized transcript chunk with corrected timestamps."""
    index: int
    text: str
    start_time: float
    end_time: float
    duration: float
    chunk_file: str
    processing_time: float
    retry_count: int


@dataclass 
class NormalizedTranscript:
    """Complete transcript with normalized timestamps."""
    chunks: List[TranscriptChunk]
    full_text: str
    total_duration: float
    metadata: Dict


class TranscriptProcessor:
    """Service for processing and normalizing transcription results."""
    
    def __init__(self):
        pass
    
    def normalize_timestamps(
        self, 
        transcription_results: List[TranscriptionResult],
        chunk_infos: List[ChunkInfo]
    ) -> List[TranscriptChunk]:
        """
        Normalize timestamps across chunks to create gap-free, increasing timestamps.
        
        Args:
            transcription_results: Results from OpenAI transcription
            chunk_infos: Original chunk information with timing data
        
        Returns:
            List of TranscriptChunk objects with normalized timestamps
        """
        # Filter successful results and sort by chunk index
        successful_results = [r for r in transcription_results if r.success]
        successful_results.sort(key=lambda x: x.chunk_index)
        
        # Create mapping of chunk index to chunk info
        chunk_info_map = {chunk.index: chunk for chunk in chunk_infos}
        
        normalized_chunks = []
        cumulative_offset = 0.0
        
        for result in successful_results:
            chunk_info = chunk_info_map.get(result.chunk_index)
            if not chunk_info:
                continue
            
            # Calculate normalized timestamps
            # Start time is the cumulative offset
            start_time = cumulative_offset
            
            # End time is start + actual chunk duration
            end_time = start_time + chunk_info.duration
            
            # Create normalized chunk
            normalized_chunk = TranscriptChunk(
                index=result.chunk_index,
                text=result.text.strip(),
                start_time=start_time,
                end_time=end_time,
                duration=chunk_info.duration,
                chunk_file=chunk_info.file_path,
                processing_time=result.processing_time,
                retry_count=result.retry_count
            )
            
            normalized_chunks.append(normalized_chunk)
            
            # Update cumulative offset for next chunk
            cumulative_offset = end_time
        
        return normalized_chunks
    
    def aggregate_transcript(
        self,
        normalized_chunks: List[TranscriptChunk],
        original_metadata: Dict
    ) -> NormalizedTranscript:
        """
        Aggregate normalized chunks into a complete transcript.
        
        Args:
            normalized_chunks: List of normalized transcript chunks
            original_metadata: Original metadata from transcription process
        
        Returns:
            NormalizedTranscript object with complete transcript
        """
        # Combine all text with proper spacing
        full_text_parts = []
        for chunk in normalized_chunks:
            if chunk.text.strip():
                full_text_parts.append(chunk.text.strip())
        
        full_text = " ".join(full_text_parts)
        
        # Calculate total duration
        total_duration = max(chunk.end_time for chunk in normalized_chunks) if normalized_chunks else 0.0
        
        # Enhanced metadata
        enhanced_metadata = {
            **original_metadata,
            "normalization": {
                "total_chunks_normalized": len(normalized_chunks),
                "total_duration": total_duration,
                "average_chunk_duration": total_duration / len(normalized_chunks) if normalized_chunks else 0.0,
                "text_length": len(full_text),
                "words_estimated": len(full_text.split()) if full_text else 0
            },
            "quality_metrics": {
                "empty_chunks": sum(1 for chunk in normalized_chunks if not chunk.text.strip()),
                "non_empty_chunks": sum(1 for chunk in normalized_chunks if chunk.text.strip()),
                "average_processing_time": sum(chunk.processing_time for chunk in normalized_chunks) / len(normalized_chunks) if normalized_chunks else 0.0,
                "total_retries": sum(chunk.retry_count for chunk in normalized_chunks)
            }
        }
        
        return NormalizedTranscript(
            chunks=normalized_chunks,
            full_text=full_text,
            total_duration=total_duration,
            metadata=enhanced_metadata
        )
    
    def format_transcript_json(self, normalized_transcript: NormalizedTranscript) -> Dict:
        """
        Format normalized transcript into the specified JSON structure.
        
        Args:
            normalized_transcript: Complete normalized transcript
        
        Returns:
            Dictionary in the specified format: {"chunks": [...], "text": "..."}
        """
        # Format chunks for output
        formatted_chunks = []
        for chunk in normalized_transcript.chunks:
            formatted_chunk = {
                "index": chunk.index,
                "text": chunk.text,
                "start_time": round(chunk.start_time, 2),
                "end_time": round(chunk.end_time, 2),
                "duration": round(chunk.duration, 2),
                "processing_time": round(chunk.processing_time, 2),
                "retry_count": chunk.retry_count
            }
            formatted_chunks.append(formatted_chunk)
        
        # Create final output structure
        transcript_json = {
            "chunks": formatted_chunks,
            "text": normalized_transcript.full_text,
            "metadata": {
                "total_duration": round(normalized_transcript.total_duration, 2),
                "total_chunks": len(normalized_transcript.chunks),
                "processing_summary": normalized_transcript.metadata.get("processing_summary", {}),
                "normalization": normalized_transcript.metadata.get("normalization", {}),
                "quality_metrics": normalized_transcript.metadata.get("quality_metrics", {})
            }
        }
        
        return transcript_json
    
    def save_transcript_file(
        self, 
        transcript_json: Dict, 
        output_path: str
    ) -> bool:
        """
        Save formatted transcript to a JSON file.
        
        Args:
            transcript_json: Formatted transcript dictionary
            output_path: File path to save the transcript
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Write JSON file with proper formatting
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_json, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save transcript to {output_path}: {e}")
            return False
    
    def process_complete_transcript(
        self,
        transcription_results: List[TranscriptionResult],
        chunk_infos: List[ChunkInfo],
        original_metadata: Dict,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Complete processing pipeline: normalize, aggregate, format, and optionally save.
        
        Args:
            transcription_results: Results from OpenAI transcription
            chunk_infos: Original chunk information
            original_metadata: Original processing metadata
            output_path: Optional path to save transcript file
        
        Returns:
            Formatted transcript JSON dictionary
        """
        print("🔄 Processing transcript: normalizing timestamps...")
        
        # Step 1: Normalize timestamps
        normalized_chunks = self.normalize_timestamps(transcription_results, chunk_infos)
        print(f"✅ Normalized {len(normalized_chunks)} chunks")
        
        # Step 2: Aggregate transcript
        print("🔄 Aggregating transcript segments...")
        normalized_transcript = self.aggregate_transcript(normalized_chunks, original_metadata)
        print(f"✅ Created transcript with {len(normalized_transcript.full_text)} characters")
        
        # Step 3: Format for output
        print("🔄 Formatting transcript JSON...")
        transcript_json = self.format_transcript_json(normalized_transcript)
        print(f"✅ Formatted transcript with {len(transcript_json['chunks'])} chunks")
        
        # Step 4: Save to file if requested
        if output_path:
            print(f"🔄 Saving transcript to {output_path}...")
            success = self.save_transcript_file(transcript_json, output_path)
            if success:
                print(f"✅ Transcript saved successfully")
            else:
                print(f"❌ Failed to save transcript")
        
        # Log summary
        metadata = transcript_json["metadata"]
        print(f"📊 Transcript Summary:")
        print(f"   📝 Total text: {len(transcript_json['text'])} characters")
        print(f"   ⏱️  Total duration: {metadata['total_duration']:.1f} seconds")
        print(f"   📦 Total chunks: {metadata['total_chunks']}")
        print(f"   🎯 Success rate: {original_metadata.get('success_rate', 0):.1%}")
        
        return transcript_json


# Global transcript processor instance
transcript_processor: Optional[TranscriptProcessor] = None


def get_transcript_processor() -> TranscriptProcessor:
    """Get transcript processor instance."""
    global transcript_processor
    if transcript_processor is None:
        transcript_processor = TranscriptProcessor()
    return transcript_processor


def init_transcript_processor() -> TranscriptProcessor:
    """Initialize transcript processor service."""
    global transcript_processor
    if transcript_processor is None:
        transcript_processor = TranscriptProcessor()
    return transcript_processor