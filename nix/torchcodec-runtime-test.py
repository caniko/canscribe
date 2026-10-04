"""Hosted packaging gate: decode PCM audio with the selected FFmpeg backend."""

import math
import struct
import tempfile
import wave
from pathlib import Path

from torchcodec.decoders import AudioDecoder

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "fixture.wav"
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * index / 16000)))
        for index in range(1600)
    )
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(frames)
    samples = AudioDecoder(str(path)).get_all_samples()
    assert samples.sample_rate == 16000, samples.sample_rate
    assert tuple(samples.data.shape) == (1, 1600), samples.data.shape
    assert samples.data.abs().mean().item() > 0.01
