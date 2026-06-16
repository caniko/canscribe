from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import torch

from .audio import get_audio_path
from .backends import (
    AsrBackend,
    DiarizationBackend,
    create_asr_backend,
    create_diarization_backend,
    load_mono_audio,
)
from .config import (
    DEFAULT_ASR_BACKEND,
    DEFAULT_DIARIZATION_BACKEND,
    MOONSHINE_MODEL,
    PARAKEET_MODEL,
    PYANNOTE_MODEL,
    VIDEO_EXTENSIONS,
    get_device,
)
from .formatting import (
    append_segment,
    default_transcript_path,
    read_transcript_end_time,
    save_transcript,
)
from .types import (
    BackendSetupError,
    TranscriptSegment,
    TranscriptionRequest,
    TranscriptionResult,
)


class TranscriptionPipeline:
    """Coordinates diarization, ASR, optional vision, and transcript writing."""

    def __init__(
        self,
        *,
        asr_backend: AsrBackend | None = None,
        diarization_backend: DiarizationBackend | None = None,
    ) -> None:
        self._asr_backend = asr_backend
        self._diarization_backend = diarization_backend

    def run(self, request: TranscriptionRequest) -> TranscriptionResult:
        input_path = request.resolved_input_path()
        if request.visual and input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise BackendSetupError("Visual analysis requires a video input file.")
        _validate_speaker_count_options(request)

        device = _resolve_device(request.device)
        asr_model = request.asr_model or _default_asr_model(request.asr_backend)
        diarization_model = request.diarization_model or PYANNOTE_MODEL
        normalized_request = replace(
            request,
            asr_model=asr_model,
            diarization_model=diarization_model,
        )

        asr_backend = self._asr_backend or create_asr_backend(
            normalized_request.asr_backend,
            model_name=asr_model,
            device=device,
        )
        diarization_backend = self._diarization_backend or create_diarization_backend(
            normalized_request.diarization_backend,
            model_name=diarization_model,
            device=device,
        )

        audio_path, temp_audio_file = get_audio_path(str(input_path))

        # Resolve output path early so we can resume
        resolved_output = (
            Path(normalized_request.output_path)
            if normalized_request.output_path is not None
            else default_transcript_path(input_path)
        )
        processed_until: float = 0.0
        if normalized_request.resume:
            processed_until = read_transcript_end_time(resolved_output)
            if processed_until > 0:
                print(f"Resuming from {processed_until:.2f}s")

        try:
            diarization_segments, warnings = diarization_backend.diarize(
                audio_path,
                speaker_count=normalized_request.speaker_count,
                min_speakers=normalized_request.min_speakers,
                max_speakers=normalized_request.max_speakers,
            )
            waveform, sample_rate = load_mono_audio(audio_path)

            segments = [
                segment
                for segment in diarization_segments
                if segment.end - segment.start
                >= normalized_request.min_segment_duration
            ]
            visual_context = (
                _build_visual_context(str(input_path), segments, device)
                if normalized_request.visual
                else {}
            )

            total = len(segments)
            transcript_segments: list[TranscriptSegment] = []
            prev_time = processed_until
            skipped = 0

            open_mode = "a" if processed_until > 0 else "w"
            with resolved_output.open(open_mode, encoding="utf-8") as handle:
                for idx, segment in enumerate(segments):
                    if segment.end <= processed_until:
                        skipped += 1
                        continue

                    start_sample = max(0, int(segment.start * sample_rate))
                    end_sample = max(start_sample, int(segment.end * sample_rate))
                    audio_chunk = waveform[start_sample:end_sample]
                    if audio_chunk.numel() == 0:
                        continue

                    asr_result = asr_backend.transcribe_chunk(audio_chunk, sample_rate)
                    if not asr_result.text:
                        continue

                    ts = TranscriptSegment(
                        start=segment.start,
                        end=segment.end,
                        speaker=segment.speaker,
                        text=asr_result.text,
                        words=asr_result.words,
                        **visual_context.get((segment.speaker, segment.start), {}),
                    )
                    transcript_segments.append(ts)

                    append_segment(handle, ts, debug=normalized_request.debug)
                    progress_pct = (idx + 1 - skipped) / max(total - skipped, 1) * 100
                    if normalized_request.debug:
                        print(
                            f"  [{progress_pct:5.1f}%] segment {idx + 1}/{total} "
                            f"({ts.start:.1f}s - {ts.end:.1f}s)",
                        )

            backend_metadata = {
                "asr_backend": asr_backend.name,
                "asr_model": asr_backend.model_name,
                "diarization_backend": diarization_backend.name,
                "diarization_model": diarization_backend.model_name,
                "device": device,
            }
            backend_metadata.update(_speaker_count_metadata(normalized_request))
            if processed_until > 0:
                backend_metadata["resumed_from"] = f"{processed_until:.2f}s"

            return TranscriptionResult(
                segments=tuple(transcript_segments),
                output_path=resolved_output,
                backend_metadata=backend_metadata,
                warnings=warnings,
            )
        finally:
            if temp_audio_file and os.path.exists(temp_audio_file):
                os.unlink(temp_audio_file)


def _validate_speaker_count_options(request: TranscriptionRequest) -> None:
    counts = {
        "speaker_count": request.speaker_count,
        "min_speakers": request.min_speakers,
        "max_speakers": request.max_speakers,
    }
    for name, value in counts.items():
        if value is not None and value < 1:
            raise BackendSetupError(f"{name} must be a positive integer.")

    if request.speaker_count is not None and (
        request.min_speakers is not None or request.max_speakers is not None
    ):
        raise BackendSetupError(
            "speaker_count cannot be combined with min_speakers or max_speakers."
        )

    if (
        request.min_speakers is not None
        and request.max_speakers is not None
        and request.min_speakers > request.max_speakers
    ):
        raise BackendSetupError(
            "min_speakers must be less than or equal to max_speakers."
        )


def _speaker_count_metadata(request: TranscriptionRequest) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if request.speaker_count is not None:
        metadata["speaker_count"] = str(request.speaker_count)
    if request.min_speakers is not None:
        metadata["min_speakers"] = str(request.min_speakers)
    if request.max_speakers is not None:
        metadata["max_speakers"] = str(request.max_speakers)
    return metadata


def _resolve_device(policy: str) -> str:
    if policy == "auto":
        return get_device()
    if policy in {"cpu", "cuda", "mps"}:
        return policy
    raise BackendSetupError(
        f"Unknown device policy '{policy}'. Expected one of: auto, cpu, cuda, mps."
    )


def _default_asr_model(backend: str) -> str:
    if backend == DEFAULT_ASR_BACKEND:
        return PARAKEET_MODEL
    if backend == "moonshine":
        return MOONSHINE_MODEL
    raise BackendSetupError(
        f"Unknown ASR backend '{backend}'. Expected one of: parakeet, moonshine."
    )


def _build_visual_context(
    video_file: str,
    segments: list,
    device: str,
) -> dict[tuple[str, float], dict[str, str]]:
    from .vision import (
        FaceAnalyzer,
        FrameRouter,
        SceneAnalyzer,
        extract_frames_at_timestamps,
    )
    from .vision.router import FrameType

    midpoints = [(segment.start + segment.end) / 2 for segment in segments]
    keyframes = extract_frames_at_timestamps(video_file, midpoints)
    keyframe_map = {keyframe.timestamp: keyframe for keyframe in keyframes}

    router = FrameRouter(device=device)
    face_analyzer = FaceAnalyzer(device=device)
    scene_analyzer = SceneAnalyzer(device=device)
    speaker_face_map: dict[str, str] = {}
    context: dict[tuple[str, float], dict[str, str]] = {}

    try:
        for segment, timestamp in zip(segments, midpoints):
            keyframe = keyframe_map.get(timestamp)
            if keyframe is None:
                continue

            routing = router.route(keyframe.frame)
            segment_context: dict[str, str] = {}

            if routing.frame_type in (FrameType.SPEAKING_FACE, FrameType.STATIC_FACE):
                face_results = face_analyzer.analyze_faces(
                    keyframe.frame,
                    predetected_faces=routing.faces,
                )
                if face_results:
                    main_face = max(
                        face_results, key=lambda face: face.bbox[2] * face.bbox[3]
                    )
                    speaker_face_map.setdefault(segment.speaker, main_face.face_id)
                    segment_context["face_id"] = speaker_face_map[segment.speaker]
                    segment_context["emotion"] = main_face.emotion
            elif routing.frame_type == FrameType.NO_FACE:
                scene_result = scene_analyzer.describe_frame(keyframe.frame, timestamp)
                segment_context["visual_description"] = scene_result.description

            if segment_context:
                context[(segment.speaker, segment.start)] = segment_context
        return context
    finally:
        router.reset_tracking()
        face_analyzer.reset()
