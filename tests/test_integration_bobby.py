"""Integration tests using bobby.mp4 clip (7:31-9:24).

These tests validate the transcription pipeline with a real video clip
containing 3 speakers with background laughing.
"""

import re
import shutil
from pathlib import Path

import pytest

from tests.conftest import requires_gpu_and_token, requires_pyannote_audio_decoder

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@requires_gpu_and_token
@requires_pyannote_audio_decoder
def test_transcribe_bobby_clip_produces_transcript(
    bobby_clip: Path, temp_output_dir: Path
) -> None:
    """
    Test that transcription produces a transcript file.

    Verifies:
    - Transcription completes without error
    - Transcript file is created
    - Transcript file contains content
    """
    from canscribe.transcription import transcribe_exclusive_speakers

    # Copy clip to temp directory so transcript is created there
    test_clip = temp_output_dir / "bobby_clip.mp4"
    shutil.copy(bobby_clip, test_clip)

    # Run transcription
    transcribe_exclusive_speakers(
        audio_file=str(test_clip),
        speaker_count=3,
        device="auto",
        debug=False,
    )

    # Verify transcript file was created
    transcript_path = temp_output_dir / "transcript-bobby_clip.txt"
    assert transcript_path.exists(), f"Transcript file not created at {transcript_path}"

    # Verify transcript has content
    content = transcript_path.read_text()
    assert len(content) > 0, "Transcript file is empty"

    # Verify transcript format (timestamps and speaker labels)
    lines = content.strip().split("\n")
    assert len(lines) > 0, "No transcript lines found"

    # Check at least one line has expected format: [X.XXs - Y.YYs] SPEAKER_XX: text
    timestamp_pattern = re.compile(r"\[\d+\.\d+s - \d+\.\d+s\] SPEAKER_\d+: .+")
    valid_lines = [line for line in lines if timestamp_pattern.match(line)]
    assert len(valid_lines) > 0, (
        f"No valid transcript lines found. Sample line: {lines[0]}"
    )


@requires_gpu_and_token
@requires_pyannote_audio_decoder
def test_transcribe_bobby_identifies_three_speakers(
    bobby_clip: Path, temp_output_dir: Path
) -> None:
    """
    Test that diarization honors the known 3-speaker count.
    """
    from canscribe.transcription import transcribe_exclusive_speakers

    # Copy clip to temp directory
    test_clip = temp_output_dir / "bobby_clip.mp4"
    shutil.copy(bobby_clip, test_clip)

    # Run transcription
    transcribe_exclusive_speakers(
        audio_file=str(test_clip),
        speaker_count=3,
        device="auto",
        debug=False,
    )

    # Parse transcript to extract unique speakers
    transcript_path = temp_output_dir / "transcript-bobby_clip.txt"
    content = transcript_path.read_text()

    # Extract speaker labels using regex
    speaker_pattern = re.compile(r"SPEAKER_(\d+)")
    speakers = set(speaker_pattern.findall(content))
    unique_speaker_count = len(speakers)

    assert unique_speaker_count == 3, f"Expected 3 speakers, found {speakers}"


@requires_gpu_and_token
@requires_pyannote_audio_decoder
def test_transcribe_bobby_segment_coverage(
    bobby_clip: Path, temp_output_dir: Path
) -> None:
    """
    Test that transcript segments cover the majority of the clip duration.

    Verifies that the transcription doesn't miss large portions of speech
    by checking that the total transcribed duration is reasonable relative
    to the clip length (113 seconds).
    """
    from canscribe.transcription import transcribe_exclusive_speakers

    # Copy clip to temp directory
    test_clip = temp_output_dir / "bobby_clip.mp4"
    shutil.copy(bobby_clip, test_clip)

    # Run transcription
    transcribe_exclusive_speakers(
        audio_file=str(test_clip),
        speaker_count=3,
        device="auto",
        debug=False,
    )

    # Parse transcript to calculate coverage
    transcript_path = temp_output_dir / "transcript-bobby_clip.txt"
    content = transcript_path.read_text()

    # Extract timestamps: [start - end]
    timestamp_pattern = re.compile(r"\[(\d+\.\d+)s - (\d+\.\d+)s\]")
    matches = timestamp_pattern.findall(content)

    assert len(matches) > 0, "No timestamp matches found in transcript"

    # Calculate total transcribed duration
    total_duration = sum(float(end) - float(start) for start, end in matches)

    # Clip is 113 seconds - expect at least 30% coverage (allowing for silence, pauses)
    # and segments should span across the clip (not just the beginning)
    min_expected_duration = 113 * 0.30  # 30% of clip
    assert total_duration >= min_expected_duration, (
        f"Transcribed duration {total_duration:.1f}s is less than "
        f"expected minimum {min_expected_duration:.1f}s (30% of 113s clip)"
    )

    # Verify segments span the clip (last segment should be after 50% of clip duration)
    if matches:
        last_end = max(float(end) for _, end in matches)
        assert last_end > 50, (
            f"Last segment ends at {last_end:.1f}s, expected segments to span "
            f"beyond 50s of the 113s clip"
        )
