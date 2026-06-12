from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .api import transcribe_file
from .config import MOONSHINE_MODEL, PYANNOTE_MODEL
from .formatting import clean_repetitive_text, save_transcript as _save_transcript
from .types import TranscriptSegment


def save_transcript(
    segments: Iterable[TranscriptSegment] | Iterator[dict[str, Any]],
    audio_file: str,
    debug: bool = False,
) -> Path:
    """Compatibility wrapper for the pre-library transcript writer."""
    typed_segments: list[TranscriptSegment] = []
    for segment in segments:
        if isinstance(segment, TranscriptSegment):
            typed_segments.append(segment)
        else:
            typed_segments.append(
                TranscriptSegment(
                    start=float(segment["start"]),
                    end=float(segment["end"]),
                    speaker=str(segment["speaker"]),
                    text=str(segment["text"]),
                )
            )
    return _save_transcript(typed_segments, audio_file, debug=debug)


def transcribe_exclusive_speakers(
    audio_file: str,
    *,
    moonshine_model: str = MOONSHINE_MODEL,
    diarization_model: str = PYANNOTE_MODEL,
    speaker_count: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    device: str | None = None,
    debug: bool = False,
) -> None:
    """Compatibility wrapper for the previous Moonshine transcription function."""
    result = transcribe_file(
        audio_file,
        asr_backend="moonshine",
        asr_model=moonshine_model,
        diarization_model=diarization_model,
        speaker_count=speaker_count,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        device=device or "auto",
        debug=debug,
    )
    print(f"\nTranscript saved to: {result.output_path}")
