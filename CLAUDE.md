# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

```bash
# Enter dev environment (installs deps, activates venv)
nix develop

# Sync dependencies (choose ONE platform extra)
uv sync --extra nvidia        # NVIDIA CUDA 12.8
uv sync --extra amd           # AMD ROCm 6.4 (Linux only)
uv sync --extra apple         # Apple Silicon (MPS)

# Optional: Flash Attention for NVIDIA Ampere+ GPUs (CC 8.0+)
uv sync --extra nvidia --extra flash

# Run type checking
uv run mypy .

# Format code
uv run ruff format .

# Run the CLI
ct <audio-or-video-file>

# Run system diagnostics
ct check [video-file]
```

## Architecture Overview

This is an audio/video transcription CLI (`ct`) that combines speech-to-text with speaker diarization and optional visual analysis.

### Core Pipeline (`src/can_transcribe/`)

**Entry Point**: `main.py` - Typer CLI with two commands:
- `transcribe` (default): Main transcription with optional `--visual` mode
- `check`: System diagnostics for GPU, FFmpeg, decoders, and dependencies

**Audio Processing Flow**:
1. `audio.py` - Extracts audio from video via FFmpeg (hardware-accelerated)
2. `transcription.py` - Main transcription pipeline:
   - Pyannote diarization → speaker segments
   - Moonshine ASR → text per segment
   - Streams results to `transcript-<filename>.txt`

**Visual Analysis Flow** (enabled with `--visual`):
1. `visual_transcription.py` - Orchestrates combined audio+visual pipeline
2. `vision/router.py` - InsightFace-based frame routing:
   - `SPEAKING_FACE`: Face with mouth movement → face analysis
   - `STATIC_FACE`: Face without movement → face analysis
   - `NO_FACE`: No faces → VLM scene description
3. `vision/face_analysis.py` - InsightFace embeddings + HuggingFace emotion detection
4. `vision/scene_analysis.py` - Qwen3-VL-2B-Thinking for scene descriptions

### Key Design Patterns

- **Lazy model loading**: All ML models are loaded on first use via `@property` accessors
- **Generator streaming**: Transcript segments are generated and saved incrementally
- **Flash Attention auto-detection**: `config.py:supports_flash_attention()` checks GPU capability and library availability
- **Safe globals for PyTorch 2.6+**: Pyannote classes added to `torch.serialization.add_safe_globals()`

### Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| ASR | `UsefulSensors/moonshine-tiny` | Speech-to-text |
| Diarization | `pyannote/speaker-diarization-community-1` | Speaker identification |
| Face Detection | InsightFace `buffalo_sc` (router) / `buffalo_l` (analysis) | Face routing and embeddings |
| Emotion | `dima806/facial_emotions_image_detection` | Emotion classification |
| Scene VLM | `Qwen/Qwen3-VL-2B-Thinking` | Non-face frame descriptions |

### Environment Requirements

- Python 3.13
- GPU recommended: NVIDIA (CUDA), AMD (ROCm), or Apple Silicon (MPS)
- FFmpeg with hardware acceleration support
- `HF_TOKEN` environment variable for pyannote model access
- OpenCV provided by Nix (opencv4Full with VAAPI)
