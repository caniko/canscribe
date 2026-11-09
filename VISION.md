# canscribe vision

A command-line transcriber for meetings and recordings: give it an audio or
video file, get a speaker-attributed timestamped transcript next to the input.
It extracts audio with hardware-accelerated FFmpeg, diarizes with pyannote,
transcribes with Parakeet by default, and streams segments to disk as they are
produced. Optional visual mode analyses video frames and attaches scene or face
context to the segments. Researchers, operators and anyone with recordings they
own are the users.

## Problem

Getting a usable transcript normally means running three tools in sequence -
extract audio, separate speakers, transcribe - and stitching their outputs by
timestamp. Long recordings make it worse: an hour in, a crash loses everything.
Local GPU machines have three different acceleration paths, and the failure mode
of a misconfigured one is an opaque library error.

## Principles

- One command, one artifact. The default invocation transcribes and writes
  `transcript-<filename>.txt` beside the input; there is no intermediate
  project format to manage.
- Stream, do not batch. Segments are generated and appended incrementally, and
  `--resume` parses existing timestamps to skip completed work, so a partial
  transcript survives a crash.
- Lazy model loading. Every model is loaded on first use; a run that never
  reaches vision never pays for InsightFace or the scene VLM.
- Explicit platform extras. `nvidia`, `amd`, `apple` and `cpu` are separate,
  mutually exclusive dependency sets rather than an auto-detected guess.
- Compatibility is named, not hidden. Parakeet is the default ASR backend;
  Moonshine stays available explicitly as the lightweight compatibility path.
- Diagnose with evidence. `canscribe check` and `canscribe doctor` probe GPU,
  FFmpeg, decoders and library resolution (`LD_LIBRARY_PATH`, RPATH/RUNPATH,
  stale ROCm libdrm paths) instead of failing deep inside a model import.
- Failing loudly beats false silence: known harmless ROCm libdrm noise is
  suppressed by a narrow context manager, not globally.

## Scope

- `transcribe` (default) with optional `--visual`, `--resume` and speaker-count
  controls.
- `check` and `doctor` system diagnostics.
- Backends: NVIDIA Parakeet, Moonshine compatibility, pyannote diarization,
  optional Flash Attention on Ampere+.
- Visual analysis: keyframe extraction, InsightFace face/emotion analysis, and
  Qwen3-VL scene descriptions.
- Packaging and publication: Nix dev shell with Harbor Python, optional CUDA
  Flash Attention, and PyPI publishing through Simit-generated workflows.

## Non-goals

- Hosted transcription or a service. This is a local CLI over local media.
- Training models; every component is a published checkpoint consumed at
  runtime.
- Supporting Python versions other than the pinned 3.13 ABI.
- Producing a scrubbed or publishable transcript: output is whatever was said,
  written next to the source file.

## Direction

The unreleased changelog is maintenance and robustness: richer diagnostics
(`--verbose`, `--probe`), narrow ROCm stderr silencing, incremental writing
through `append_segment`, and the `--resume` flag with its `TranscriptionRequest`
field. Build and CI moved to Harbor Python and Simit 0.18 workflows with GitHub
as the canonical forge.

## Place in the fleet

Canix pins it in `root/hosts/atlas/simit-updater.nix` for automated dependency
updates; it is otherwise invoked locally by the operator with an `HF_TOKEN` on
a GPU host. Package builds follow the fleet's Nix/Harbor Python conventions.
