"""
Visual transcription module.

Combines audio transcription with visual analysis for enriched output.
"""

import os
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.pipelines.utils.hook import ProgressHook
from transformers import AutoProcessor, MoonshineForConditionalGeneration

from .audio import get_audio_path
from .config import MOONSHINE_MODEL, PYANNOTE_MODEL, supports_flash_attention
from .transcription import clean_repetitive_text
from .vision import FaceAnalyzer, FrameRouter, SceneAnalyzer, extract_frames_at_timestamps
from .vision.router import FrameType

# Allow pyannote classes in torch.load for weights_only=True (PyTorch 2.6+)
torch.serialization.add_safe_globals([Problem, Resolution, Specifications])


@dataclass
class VisualSegment:
    """A transcript segment with optional visual context."""

    start: float
    end: float
    speaker: str
    text: str
    face_id: str | None = None  # Matched face identity
    emotion: str | None = None  # Emotion adjective
    visual_description: str | None = None  # Scene description


def save_visual_transcript(
    segments: Generator[VisualSegment, None, None],
    audio_file: str,
    debug: bool = False,
) -> Path:
    """Save enriched transcript with visual annotations."""
    audio_path = Path(audio_file)
    output_path = audio_path.parent / f"transcript-{audio_path.stem}.txt"

    with open(output_path, "w") as f:
        for segment in segments:
            # Format speaker with face ID and emotion if available
            if segment.face_id and segment.emotion:
                speaker_info = f"{segment.speaker} ({segment.face_id}, {segment.emotion})"
            elif segment.face_id:
                speaker_info = f"{segment.speaker} ({segment.face_id})"
            else:
                speaker_info = segment.speaker

            line = f"[{segment.start:.2f}s - {segment.end:.2f}s] {speaker_info}: {segment.text}\n"
            f.write(line)

            # Add visual description if available
            if segment.visual_description:
                visual_line = f"    [Visual: {segment.visual_description}]\n"
                f.write(visual_line)
                if debug:
                    print(visual_line, end="")

            if debug:
                print(line, end="")

    return output_path


def transcribe_with_vision(
    video_file: str,
    *,
    moonshine_model: str = MOONSHINE_MODEL,
    diarization_model: str = PYANNOTE_MODEL,
    device: str | None = None,
    debug: bool = False,
) -> None:
    """
    Transcribe a video file with speaker diarization and visual analysis.

    Uses tiered visual analysis:
    - Speaking faces: InsightFace + DeepFace for identity and emotion
    - Non-face frames: Qwen3-VL-2B-Thinking for scene descriptions

    Args:
        video_file: Path to the video file.
        moonshine_model: Moonshine model for transcription.
        diarization_model: Pyannote diarization model.
        device: Device to run on ("cuda" or "cpu"). Auto-detected if None.
        debug: Enable debug output.
    """
    discovered_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_flash = supports_flash_attention()

    print(f"🚀 Running on {discovered_device.upper()} with visual analysis...")
    if debug:
        print(f"🔧 Flash Attention 2 enabled: {use_flash}")

    # Extract audio from video
    audio_path, temp_audio_file = get_audio_path(video_file)

    try:
        # 1. Load Diarization Pipeline
        print("📦 Loading diarization model...")
        diarization_pipeline = Pipeline.from_pretrained(
            diarization_model, token=os.environ["HF_TOKEN"]
        )
        if diarization_pipeline is None:
            raise RuntimeError(f"Failed to load diarization model: {diarization_model}")
        diarization_pipeline = diarization_pipeline.to(torch.device(discovered_device))

        # 2. Load Transcription Model (Moonshine)
        print("📦 Loading transcription model...")
        processor = AutoProcessor.from_pretrained(moonshine_model)
        model = MoonshineForConditionalGeneration.from_pretrained(
            moonshine_model,
            attn_implementation="flash_attention_2" if use_flash else "sdpa",
            dtype=torch.float16 if use_flash else torch.float32,
        ).to(discovered_device)

        # 3. Initialize Visual Analysis Components
        print("📦 Loading visual analysis models...")
        router = FrameRouter(device=discovered_device)
        face_analyzer = FaceAnalyzer(device=discovered_device)
        scene_analyzer = SceneAnalyzer(device=discovered_device)

        # 4. Run Diarization
        print("🔎 Diarizing (Finding Dominant Speakers)...")
        with ProgressHook() as hook:
            diarization_result = diarization_pipeline(audio_path, hook=hook)

        # 5. Load Audio for Slicing
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )
            waveform = resampler(waveform)
            sample_rate = 16000

        # Get speaker timeline
        try:
            timeline = diarization_result.exclusive_speaker_diarization
        except AttributeError:
            print("⚠️ Warning: 'exclusive_speaker_diarization' not found. Fallback to standard.")
            timeline = diarization_result

        # 6. Collect segment timestamps for frame extraction
        segments_list = []
        for segment, _, speaker in timeline.itertracks(yield_label=True):
            duration = segment.end - segment.start
            if duration >= 0.2:
                segments_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": speaker,
                })

        # 7. Extract keyframes at segment midpoints
        print("🎬 Extracting keyframes...")
        midpoints = [(s["start"] + s["end"]) / 2 for s in segments_list]
        keyframes = extract_frames_at_timestamps(video_file, midpoints)

        # Build timestamp to keyframe mapping
        keyframe_map = {kf.timestamp: kf for kf in keyframes}

        # 8. Correlate faces with speakers based on temporal overlap
        print("👤 Analyzing faces and correlating with speakers...")
        speaker_face_map: dict[str, str] = {}  # SPEAKER_XX -> Person X
        speaker_emotion_map: dict[str, tuple[str, str]] = {}  # SPEAKER_XX -> (face_id, emotion)

        for seg, kf_ts in zip(segments_list, midpoints):
            kf = keyframe_map.get(kf_ts)
            if kf is None:
                continue

            # Route frame
            routing = router.route(kf.frame)

            if routing.frame_type in (FrameType.SPEAKING_FACE, FrameType.STATIC_FACE):
                # Analyze faces
                face_results = face_analyzer.analyze_faces(
                    kf.frame,
                    predetected_faces=routing.faces,
                )

                if face_results:
                    # Use the most prominent face (largest bbox)
                    main_face = max(face_results, key=lambda f: f.bbox[2] * f.bbox[3])
                    speaker = seg["speaker"]

                    # Map speaker to face if not already mapped
                    if speaker not in speaker_face_map:
                        speaker_face_map[speaker] = main_face.face_id

                    # Store current emotion for this segment
                    speaker_emotion_map[f"{speaker}_{seg['start']}"] = (
                        main_face.face_id,
                        main_face.emotion,
                    )

        print("🎙️ Transcribing Segments with Visual Context...")

        def visual_segment_generator() -> Generator[VisualSegment, None, None]:
            """Generate enriched transcript segments."""
            for seg, kf_ts in zip(segments_list, midpoints):
                start = seg["start"]
                end = seg["end"]
                speaker = seg["speaker"]

                # Slice and transcribe audio
                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)
                audio_chunk = waveform[0, start_sample:end_sample]

                inputs = processor(
                    audio_chunk, sampling_rate=16000, return_tensors="pt"
                ).to(discovered_device)

                with torch.no_grad():
                    generated_ids = model.generate(**inputs)

                text = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0].strip()
                text = clean_repetitive_text(text)

                if not text:
                    continue

                # Get face ID and emotion for this segment
                emotion_key = f"{speaker}_{start}"
                face_id = speaker_face_map.get(speaker)
                emotion = None

                if emotion_key in speaker_emotion_map:
                    face_id, emotion = speaker_emotion_map[emotion_key]

                # Get visual description for non-face frames
                visual_description = None
                kf = keyframe_map.get(kf_ts)

                if kf is not None:
                    routing = router.route(kf.frame)
                    if routing.frame_type == FrameType.NO_FACE:
                        # Use VLM for scene description
                        scene_result = scene_analyzer.describe_frame(
                            kf.frame, kf_ts
                        )
                        visual_description = scene_result.description

                yield VisualSegment(
                    start=start,
                    end=end,
                    speaker=speaker,
                    text=text,
                    face_id=face_id,
                    emotion=emotion,
                    visual_description=visual_description,
                )

        # Save enriched transcript
        output_path = save_visual_transcript(
            visual_segment_generator(), video_file, debug=debug
        )
        print(f"\n✅ Enriched transcript saved to: {output_path}")

        # Final face clustering for identity consolidation
        identity_mapping = face_analyzer.cluster_identities()
        if identity_mapping and debug:
            print(f"🔄 Identity consolidation: {identity_mapping}")

    finally:
        # Clean up
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.unlink(temp_audio_file)
        router.reset_tracking()
        face_analyzer.reset()
