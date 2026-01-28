"""Pytest fixtures for can-transcribe integration tests."""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

# Path to the test video in the project root
BOBBY_VIDEO = Path(__file__).parent.parent / "bobby.mp4"

# Clip timing: 7:31-9:24 in seconds
CLIP_START_SECONDS = 7 * 60 + 31  # 451 seconds
CLIP_DURATION_SECONDS = (9 * 60 + 24) - CLIP_START_SECONDS  # 113 seconds


def has_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def has_hf_token() -> bool:
    """Check if HuggingFace token is set."""
    return bool(os.environ.get("HF_TOKEN"))


def pyannote_audio_decoder_works() -> bool:
    """Check if pyannote.audio's AudioDecoder is properly available.

    pyannote.audio 4.x has a bug where AudioDecoder may not be defined
    if torchcodec/FFmpeg libraries are not properly installed.
    """
    try:
        # This import triggers the AudioDecoder check
        from pyannote.audio.core.io import get_audio_metadata
        # Try to verify AudioDecoder is defined in the module
        import pyannote.audio.core.io as io_module
        return hasattr(io_module, 'AudioDecoder') or 'AudioDecoder' in dir(io_module)
    except Exception:
        return False


# Skip markers for GPU-required tests
requires_cuda = pytest.mark.skipif(
    not has_cuda(),
    reason="CUDA not available"
)

requires_hf_token = pytest.mark.skipif(
    not has_hf_token(),
    reason="HF_TOKEN environment variable not set"
)

requires_gpu_and_token = pytest.mark.skipif(
    not has_cuda() or not has_hf_token(),
    reason="Requires CUDA and HF_TOKEN"
)

requires_pyannote_audio_decoder = pytest.mark.skipif(
    not pyannote_audio_decoder_works(),
    reason="pyannote.audio AudioDecoder not available (torchcodec/FFmpeg issue)"
)


@pytest.fixture(scope="session")
def bobby_video_path() -> Path:
    """Return path to the bobby.mp4 video file."""
    if not BOBBY_VIDEO.exists():
        pytest.skip(f"Test video not found: {BOBBY_VIDEO}")
    return BOBBY_VIDEO


@pytest.fixture(scope="session")
def bobby_clip(bobby_video_path: Path) -> Generator[Path, None, None]:
    """
    Extract a 113-second clip (7:31-9:24) from bobby.mp4 to a temporary file.

    Uses ffmpeg to extract the clip with these parameters:
    - Start: 451 seconds (7:31)
    - Duration: 113 seconds (until 9:24)

    The clip is cached for the entire test session.
    """
    # Create a temporary file that persists for the session
    temp_dir = tempfile.mkdtemp(prefix="can_transcribe_test_")
    clip_path = Path(temp_dir) / "bobby_clip.mp4"

    # Extract clip using ffmpeg
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",  # Overwrite output file if exists
            "-ss", str(CLIP_START_SECONDS),  # Start time
            "-i", str(bobby_video_path),  # Input file
            "-t", str(CLIP_DURATION_SECONDS),  # Duration
            "-c", "copy",  # Copy streams without re-encoding (fast)
            str(clip_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        pytest.fail(f"Failed to extract clip: {result.stderr}")

    if not clip_path.exists():
        pytest.fail(f"Clip file not created: {clip_path}")

    yield clip_path

    # Cleanup after all tests
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test outputs."""
    return tmp_path
