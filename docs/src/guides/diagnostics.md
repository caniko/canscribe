# Diagnostics

Run diagnostics with:

```sh
canscribe check
```

To include an FFmpeg video decode probe, pass a video file:

```sh
canscribe check ~/Videos/meeting.mp4
```

The check command reports Python package imports, Torch device availability, FFmpeg availability, decoder behavior, pyannote import readiness, and torchcodec audio decoding.

## Nix Smoke Probe

Inside `nix develop`, the flake also provides:

```sh
canscribe-smoke
```

The smoke probe prints Torch package versions, FFmpeg version, backend detection, GPU device details when available, and a small GPU kernel probe.

## Common Failure Points

- `HF_TOKEN` is missing or the pyannote model terms have not been accepted.
- FFmpeg is unavailable outside the Nix shell.
- The selected uv platform extra does not match the machine.
- ROCm runtime libraries from the system shadow the PyTorch wheel runtime.
