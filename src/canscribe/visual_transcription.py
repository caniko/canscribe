from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .api import transcribe_file
from .config import MOONSHINE_MODEL, PYANNOTE_MODEL
from .formatting import save_transcript as _save_transcript
from .types import TranscriptSegment


@dataclass(frozen=True)
class VisualSegment:
    """Compatibility type for callers that imported the old visual segment class."""

    start: float
    end: float
    speaker: str
    text: str
    face_id: str | None = None
    emotion: str | None = None
    visual_description: str | None = None

    def to_transcript_segment(self) -> TranscriptSegment:
        return TranscriptSegment(
            start=self.start,
            end=self.end,
            speaker=self.speaker,
            text=self.text,
            face_id=self.face_id,
            emotion=self.emotion,
            visual_description=self.visual_description,
        )


def save_visual_transcript(
    segments,
    audio_file: str,
    debug: bool = False,
) -> Path:
    typed_segments = [
        segment.to_transcript_segment()
        if isinstance(segment, VisualSegment)
        else segment
        for segment in segments
    ]
    return _save_transcript(typed_segments, audio_file, debug=debug)


def transcribe_with_vision(
    video_file: str,
    *,
    moonshine_model: str = MOONSHINE_MODEL,
    diarization_model: str = PYANNOTE_MODEL,
    speaker_count: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    device: str | None = None,
    debug: bool = False,
) -> None:
    """Compatibility wrapper for the previous visual transcription function."""
    result = transcribe_file(
        video_file,
        asr_backend="moonshine",
        asr_model=moonshine_model,
        diarization_model=diarization_model,
        speaker_count=speaker_count,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        device=device or "auto",
        visual=True,
        debug=debug,
    )
    print(f"\nEnriched transcript saved to: {result.output_path}")
