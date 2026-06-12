from __future__ import annotations

from pathlib import Path

from canscribe.formatting import (
    clean_repetitive_text,
    default_transcript_path,
    save_transcript,
)
from canscribe.types import TranscriptSegment


def test_transcript_segment_renders_visual_context() -> None:
    segment = TranscriptSegment(
        start=1.234,
        end=5.678,
        speaker="SPEAKER_00",
        text="hello there",
        face_id="Person A",
        emotion="composed",
        visual_description="slides with a title",
    )

    assert segment.to_text_lines() == [
        "[1.23s - 5.68s] SPEAKER_00 (Person A, composed): hello there",
        "    [Visual: slides with a title]",
    ]


def test_clean_repetitive_text_reduces_common_asr_artifacts() -> None:
    assert clean_repetitive_text("yes yes yes yes yes") == "yes yes yes"
    assert (
        clean_repetitive_text("hello, hello, hello, hello, hello, next")
        == "hello, hello, hello, next"
    )


def test_default_transcript_path_uses_input_stem() -> None:
    assert default_transcript_path(Path("/tmp/meeting.mp4")) == Path(
        "/tmp/transcript-meeting.txt"
    )


def test_save_transcript_writes_all_segment_lines(tmp_path: Path) -> None:
    input_path = tmp_path / "meeting.wav"
    output_path = tmp_path / "custom.txt"
    segments = [
        TranscriptSegment(0, 1, "SPEAKER_00", "hello"),
        TranscriptSegment(
            1,
            2,
            "SPEAKER_01",
            "look",
            visual_description="whiteboard",
        ),
    ]

    written_path = save_transcript(segments, input_path, output_path=output_path)

    assert written_path == output_path
    assert output_path.read_text(encoding="utf-8") == (
        "[0.00s - 1.00s] SPEAKER_00: hello\n"
        "[1.00s - 2.00s] SPEAKER_01: look\n"
        "    [Visual: whiteboard]\n"
    )
