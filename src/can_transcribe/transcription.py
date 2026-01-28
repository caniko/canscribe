import os
import re
from collections.abc import Generator
from pathlib import Path
from typing import Iterator

import torch
import torchaudio  # type: ignore[import-untyped]
from pyannote.audio import Pipeline
from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.pipelines.utils.hook import ProgressHook
from transformers import AutoProcessor, MoonshineForConditionalGeneration

from .audio import get_audio_path
from .config import MOONSHINE_MODEL, PYANNOTE_MODEL, supports_flash_attention

# Allow pyannote classes in torch.load for weights_only=True (PyTorch 2.6+)
torch.serialization.add_safe_globals([Problem, Resolution, Specifications])


def clean_repetitive_text(text: str, max_repeats: int = 3) -> str:
    """
    Clean up repetitive words/phrases that often appear in transcription artifacts.

    Args:
        text: The transcribed text to clean.
        max_repeats: Maximum allowed consecutive repetitions of a word/phrase.

    Returns:
        Cleaned text with excessive repetitions reduced.
    """
    if not text:
        return text

    # Pattern to match a word/phrase repeated more than max_repeats times
    # Matches: "word, word, word, word" or "word word word word" etc.
    # Captures the repeated unit and replaces with max_repeats occurrences

    # Handle comma-separated repetitions like "uh, uh, uh, uh, uh"
    def reduce_comma_repeats(match: re.Match) -> str:
        unit = match.group(1)
        return ", ".join([unit.strip()] * max_repeats)

    # Pattern for comma-separated repetitions (e.g., "uh, uh, uh, uh, uh")
    comma_pattern = r'((?:\w+(?:\s+\w+)*)\s*,\s*)(?:\1){' + str(max_repeats) + r',}'
    text = re.sub(comma_pattern, reduce_comma_repeats, text, flags=re.IGNORECASE)

    # Handle space-separated repetitions like "I I I I I I"
    def reduce_space_repeats(match: re.Match) -> str:
        unit = match.group(1)
        return " ".join([unit.strip()] * max_repeats)

    # Pattern for space-separated single word repetitions
    space_pattern = r'\b(\w+)(?:\s+\1){' + str(max_repeats) + r',}\b'
    text = re.sub(space_pattern, reduce_space_repeats, text, flags=re.IGNORECASE)

    # Clean up any resulting double spaces
    text = re.sub(r'\s{2,}', ' ', text)

    return text.strip()


def save_transcript(
    segments: Iterator[dict],
    audio_file: str,
    debug: bool = False,
) -> Path:
    """Save the transcript segments to a text file, streaming as they arrive."""
    audio_path = Path(audio_file)
    output_path = audio_path.parent / f"transcript-{audio_path.stem}.txt"

    with open(output_path, "w") as f:
        for segment in segments:
            line = f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['speaker']}: {segment['text']}\n"
            f.write(line)
            if debug:
                print(line, end="")

    return output_path


def transcribe_exclusive_speakers(
    audio_file: str,
    *,
    moonshine_model: str = MOONSHINE_MODEL,
    diarization_model: str = PYANNOTE_MODEL,
    device: str | None = None,
    debug: bool = False,
) -> None:
    """
    Transcribe an audio or video file with speaker diarization.

    Args:
        audio_file: Path to the audio or video file.
        moonshine_model: Moonshine model to use for transcription.
        diarization_model: Pyannote diarization model.
        device: Device to run on ("cuda" or "cpu"). Auto-detected if None.
        debug: Enable debug output.
    """
    discovered_device = device or "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\U0001f680 Running on {discovered_device.upper()} with {diarization_model}...")

    # Extract audio if input is a video file
    audio_path, temp_audio_file = get_audio_path(audio_file)

    try:
        # 1. Load Diarization Pipeline (Community-1)
        diarization_pipeline = Pipeline.from_pretrained(
            diarization_model, token=os.environ["HF_TOKEN"]
        )
        if diarization_pipeline is None:
            raise RuntimeError(f"Failed to load diarization model: {diarization_model}")
        diarization_pipeline = diarization_pipeline.to(torch.device(discovered_device))

        # 2. Load Transcription Model (Moonshine) with flash attention auto-detection
        use_flash = supports_flash_attention()
        if debug:
            print(f"🔧 Flash Attention 2 enabled: {use_flash}")

        processor = AutoProcessor.from_pretrained(moonshine_model)
        model = MoonshineForConditionalGeneration.from_pretrained(
            moonshine_model,
            attn_implementation="flash_attention_2" if use_flash else "sdpa",
            dtype=torch.float16 if use_flash else torch.float32,
        ).to(discovered_device)

        print("\U0001f50e Diarizing (Finding Dominant Speakers)...")
        with ProgressHook() as hook:
            diarization_result = diarization_pipeline(audio_path, hook=hook)

        # 3. Load Audio for Slicing
        waveform, sample_rate = torchaudio.load(audio_path)
        # Resample to 16kHz for Moonshine
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )
            waveform = resampler(waveform)
            sample_rate = 16000

        print("\U0001f399\ufe0f Transcribing Segments...")

        # --- Use Exclusive Speaker Diarization ---
        # Community-1 provides this specific property to flatten overlaps automatically.
        try:
            timeline = diarization_result.exclusive_speaker_diarization
        except AttributeError:
            print(
                "\u26a0\ufe0f Warning: 'exclusive_speaker_diarization' not found. Fallback to standard."
            )
            timeline = diarization_result

        def segment_generator() -> Generator[dict, None, None]:
            """Generate transcript segments one at a time."""
            for segment, _, speaker in timeline.itertracks(yield_label=True):
                start = segment.start
                end = segment.end
                duration = end - start

                # Skip tiny blips (< 0.2s) that are usually noise
                if duration < 0.2:
                    continue

                # Slice Audio
                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)
                audio_chunk = waveform[0, start_sample:end_sample]

                # Transcribe with Moonshine
                inputs = processor(
                    audio_chunk, sampling_rate=16000, return_tensors="pt"
                ).to(discovered_device)
                with torch.no_grad():
                    generated_ids = model.generate(**inputs)

                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                # Clean up repetitive artifacts from transcription
                text = clean_repetitive_text(text)

                if text:
                    yield {"start": start, "end": end, "speaker": speaker, "text": text}

        # Save transcript to file, streaming segments as they're generated
        output_path = save_transcript(segment_generator(), audio_file, debug=debug)
        print(f"\n✅ Transcript saved to: {output_path}")

    finally:
        # Clean up temporary audio file
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.unlink(temp_audio_file)
