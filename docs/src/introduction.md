# Introduction

canscribe is an audio and video transcription CLI. It extracts audio from media files, runs speaker diarization, transcribes each speaker segment, and writes a timestamped transcript while processing continues.

The default speech-to-text backend is NVIDIA Parakeet TDT 0.6B v3. Moonshine remains available as a compatibility backend. Diarization uses `pyannote/speaker-diarization-community-1`, which requires a Hugging Face token and acceptance of the model terms.

The optional visual mode adds frame analysis for video inputs. It routes frames through face detection, face analysis, or scene description depending on whether a speaking face, static face, or no face is detected.

## Main Commands

```sh
canscribe <audio-or-video-file>
canscribe --visual <video-file>
canscribe check [video-file]
```

`ct` is installed as a compatibility alias for `canscribe`.

## Outputs

Transcripts are written next to the input file as `transcript-<filename>.txt`. Each line includes the segment start, segment end, speaker label, and transcribed text.
