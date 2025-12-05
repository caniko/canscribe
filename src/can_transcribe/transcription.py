import os
from collections.abc import Generator
from pathlib import Path
from typing import Iterator

import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.pipelines.utils.hook import ProgressHook
from transformers import AutoProcessor, MoonshineForConditionalGeneration

from .audio import get_audio_path
from .config import MOONSHINE_MODEL, PYANNOTE_MODEL

# Allow pyannote classes in torch.load for weights_only=True (PyTorch 2.6+)
torch.serialization.add_safe_globals([Problem, Resolution, Specifications])


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

        # 2. Load Transcription Model (Moonshine)
        processor = AutoProcessor.from_pretrained(moonshine_model)
        model = MoonshineForConditionalGeneration.from_pretrained(moonshine_model).to(
            discovered_device
        )

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

                if text:
                    yield {"start": start, "end": end, "speaker": speaker, "text": text}

        # Save transcript to file, streaming segments as they're generated
        output_path = save_transcript(segment_generator(), audio_file, debug=debug)
        print(f"\n✅ Transcript saved to: {output_path}")

    finally:
        # Clean up temporary audio file
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.unlink(temp_audio_file)
