# can-transcribe

Simple audio transcription CLI that uses cutting edge models 🤖

Transcribes audio/video files with **speaker diarization** (identifies who is speaking) using:
- **[Moonshine](https://huggingface.co/UsefulSensors/moonshine-tiny)** - Fast, accurate speech-to-text
- **[pyannote](https://huggingface.co/pyannote/speaker-diarization-community-1)** - Speaker diarization (who spoke when)

## Features

- 🎬 Supports video files (mp4, mkv, avi, mov, webm, flv, wmv) - audio is extracted automatically
- 🎤 Speaker diarization - identifies different speakers
- 🚀 GPU acceleration with CUDA 13.0
- 📝 Streams transcription to file as it processes
- ⚡ Uses efficient Moonshine-tiny model by default

## Prerequisites

### Hugging Face Token

You need a Hugging Face account and API token to use the pyannote diarization model:

1. Create an account at [huggingface.co](https://huggingface.co)
2. Accept the model terms at [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
3. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Set the environment variable:
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```

### System Requirements

- Python 3.13
- NVIDIA GPU with CUDA 13.0 support (recommended) or CPU fallback
- FFmpeg (for video file support)

## Installation

### Using Nix (recommended)

```bash
nix develop
```

This automatically sets up:
- Python 3.13
- uv package manager
- FFmpeg
- CUDA libraries via system driver

### Using uv

```bash
uv sync
```

Note: PyTorch is installed from the CUDA 13.0 index for GPU support.

## Usage

### Basic Usage

```bash
ct "/path/to/audio-or-video.mp4"
```

Output is saved to `transcript-<filename>.txt` in the same directory as the input file.

### CLI Options

```bash
ct [OPTIONS] AUDIO_FILE
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--model` | `-m` | Moonshine model for transcription | `UsefulSensors/moonshine-tiny` |
| `--diarization` | `-d` | Pyannote diarization model | `pyannote/speaker-diarization-community-1` |
| `--cpu` | | Force CPU usage instead of CUDA | `false` |
| `--debug` | | Print each segment as it's transcribed | `false` |

### Examples

```bash
# Basic transcription
ct ~/Videos/meeting.mp4

# Use larger model for better accuracy
ct -m UsefulSensors/moonshine-base ~/Videos/meeting.mp4

# Force CPU (slower but works without GPU)
ct --cpu ~/Videos/meeting.mp4

# Debug mode - see segments as they're transcribed
ct --debug ~/Videos/meeting.mp4
```

### Output Format

The transcript is saved as a text file with timestamps and speaker labels:

```
[0.00s - 2.34s] SPEAKER_00: Hello, welcome to the meeting.
[2.50s - 5.12s] SPEAKER_01: Thanks for having me.
[5.30s - 8.45s] SPEAKER_00: Let's get started with the agenda.
```

## Models

### Transcription (Moonshine)

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `UsefulSensors/moonshine-tiny` | Small | Fast | Good |
| `UsefulSensors/moonshine-base` | Medium | Moderate | Better |

### Diarization (pyannote)

| Model | Description |
|-------|-------------|
| `pyannote/speaker-diarization-community-1` | Free community model, requires HF token |

## Troubleshooting

### CUDA not available

If you see `⚠️ CUDA not available, falling back to CPU`:

1. Ensure you have an NVIDIA GPU
2. On NixOS, the flake automatically adds `/run/opengl-driver/lib` to `LD_LIBRARY_PATH`
3. Check that your driver supports CUDA 13.0: `nvidia-smi`

### PyTorch pickle errors

If you see `WeightsUnpickler error` about pyannote classes, the code already handles this by adding safe globals. If you encounter new classes, they may need to be added to the allowlist in `transcription.py`.

### FFmpeg not found

For video file support, ensure FFmpeg is installed:
- NixOS: Included in the flake
- Ubuntu: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`

## Development

```bash
# Enter dev environment
nix develop

# Run type checking
uv run mypy .

# Format code
uv format
```

## License

MIT
