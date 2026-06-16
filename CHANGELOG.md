# Changelog

## [Unreleased]

### Added

- **torching diagnostic CLI**: New standalone `torching` package with `--verbose`
  and `--probe` flags. Scans bundled ROCm libdrm for stale hardcoded paths,
  validates `LD_LIBRARY_PATH`, traces RPATH/RUNPATH resolution. Integrated as
  `canscribe doctor`.
- **ROCm stderr silencing**: `silence_stderr()` context manager suppresses the
  harmless `/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory`
  noise from ROCm's bundled libdrm during the first GPU probe.
- **`--resume` / `-r` flag**: New flag on `transcribe` command. Parses existing
  transcript timestamps, skips already-processed segments, and saves
  incrementally so partial work survives crashes.
- **uv workspace**: Added `src/torching` as a workspace member.
- **Nix uv-format check**: Added `uv-format` CI check alongside typecheck.
- **PyPI publish CI**: New Forgejo workflow triggered by version tags.
- **simit config**: Enabled `uv-format` check and `with_pypi_publish`.

### Changed

- `save_transcript` refactored into `append_segment` for incremental writing.
- `TranscriptionRequest` gains `resume: bool` field.
