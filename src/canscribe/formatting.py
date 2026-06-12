from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from .types import TranscriptSegment


def clean_repetitive_text(text: str, max_repeats: int = 3) -> str:
    """Reduce common repeated-word artifacts from ASR output."""
    if not text:
        return text

    def reduce_comma_repeats(match: re.Match[str]) -> str:
        unit = match.group(1)
        return ", ".join([unit] * max_repeats)

    comma_pattern = r"\b(\w+)(?:,\s*\1){" + str(max_repeats) + r",}\b"
    text = re.sub(comma_pattern, reduce_comma_repeats, text, flags=re.IGNORECASE)

    def reduce_space_repeats(match: re.Match[str]) -> str:
        unit = match.group(1)
        return " ".join([unit.strip()] * max_repeats)

    space_pattern = r"\b(\w+)(?:\s+\1){" + str(max_repeats) + r",}\b"
    text = re.sub(space_pattern, reduce_space_repeats, text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def default_transcript_path(input_path: Path) -> Path:
    return input_path.parent / f"transcript-{input_path.stem}.txt"


def save_transcript(
    segments: Iterable[TranscriptSegment],
    input_path: Path | str,
    *,
    output_path: Path | str | None = None,
    debug: bool = False,
) -> Path:
    """Write transcript text and return the output path."""
    resolved_input = Path(input_path)
    resolved_output = (
        Path(output_path)
        if output_path is not None
        else default_transcript_path(resolved_input)
    )

    with resolved_output.open("w", encoding="utf-8") as handle:
        for segment in segments:
            for line in segment.to_text_lines():
                rendered = f"{line}\n"
                handle.write(rendered)
                if debug:
                    print(rendered, end="")

    return resolved_output
