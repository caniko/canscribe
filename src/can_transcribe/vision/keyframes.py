"""
Keyframe extraction using PySceneDetect.

Intelligently extracts keyframes at scene changes to reduce
redundant visual analysis.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scenedetect import detect, ContentDetector, AdaptiveDetector  # type: ignore[import-untyped]


@dataclass
class Keyframe:
    """A keyframe extracted from video."""

    frame: np.ndarray  # BGR image
    timestamp: float  # Seconds
    scene_index: int  # Which scene this frame belongs to


@dataclass
class Scene:
    """A detected scene in the video."""

    start_time: float
    end_time: float
    keyframe: Keyframe


def extract_keyframes(
    video_path: str | Path,
    *,
    method: str = "content",
    threshold: float = 27.0,
    min_scene_len: float = 1.0,
    fallback_interval: float = 5.0,
) -> list[Keyframe]:
    """
    Extract keyframes from video at scene boundaries.

    Args:
        video_path: Path to video file.
        method: Detection method ("content" or "adaptive").
        threshold: Detection threshold (higher = fewer scenes).
        min_scene_len: Minimum scene length in seconds.
        fallback_interval: Sample interval for long static scenes.

    Returns:
        List of Keyframe objects.
    """
    video_path = Path(video_path)

    # Select detector
    if method == "adaptive":
        detector = AdaptiveDetector(
            adaptive_threshold=threshold,
            min_scene_len=int(min_scene_len * 30),  # Assume 30fps
        )
    else:
        detector = ContentDetector(
            threshold=threshold,
            min_scene_len=int(min_scene_len * 30),
        )

    # Detect scenes
    scene_list = detect(str(video_path), detector)

    # Open video for frame extraction
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    keyframes = []

    if not scene_list:
        # No scenes detected, sample uniformly
        timestamps = list(np.arange(0, duration, fallback_interval))
        for i, ts in enumerate(timestamps):
            frame = _extract_frame_at(cap, float(ts), fps)
            if frame is not None:
                keyframes.append(Keyframe(frame=frame, timestamp=float(ts), scene_index=i))
    else:
        for i, (start, end) in enumerate(scene_list):
            start_sec = start.get_seconds()
            end_sec = end.get_seconds()
            scene_duration = end_sec - start_sec

            # Extract keyframe at scene start
            frame = _extract_frame_at(cap, start_sec, fps)
            if frame is not None:
                keyframes.append(
                    Keyframe(frame=frame, timestamp=start_sec, scene_index=i)
                )

            # For long scenes, add intermediate samples
            if scene_duration > fallback_interval * 2:
                intermediate_times = np.arange(
                    start_sec + fallback_interval,
                    end_sec,
                    fallback_interval,
                )
                for ts in intermediate_times:
                    frame = _extract_frame_at(cap, float(ts), fps)
                    if frame is not None:
                        keyframes.append(
                            Keyframe(frame=frame, timestamp=float(ts), scene_index=i)
                        )

    cap.release()
    return keyframes


def extract_scenes(
    video_path: str | Path,
    *,
    method: str = "content",
    threshold: float = 27.0,
    min_scene_len: float = 1.0,
) -> list[Scene]:
    """
    Detect scenes and extract one keyframe per scene.

    Args:
        video_path: Path to video file.
        method: Detection method ("content" or "adaptive").
        threshold: Detection threshold.
        min_scene_len: Minimum scene length in seconds.

    Returns:
        List of Scene objects with keyframes.
    """
    video_path = Path(video_path)

    if method == "adaptive":
        detector = AdaptiveDetector(
            adaptive_threshold=threshold,
            min_scene_len=int(min_scene_len * 30),
        )
    else:
        detector = ContentDetector(
            threshold=threshold,
            min_scene_len=int(min_scene_len * 30),
        )

    scene_list = detect(str(video_path), detector)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    scenes = []

    for i, (start, end) in enumerate(scene_list):
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()

        # Extract keyframe at scene midpoint
        mid_sec = (start_sec + end_sec) / 2
        frame = _extract_frame_at(cap, mid_sec, fps)

        if frame is not None:
            keyframe = Keyframe(frame=frame, timestamp=mid_sec, scene_index=i)
            scenes.append(
                Scene(start_time=start_sec, end_time=end_sec, keyframe=keyframe)
            )

    cap.release()
    return scenes


def extract_frames_at_timestamps(
    video_path: str | Path,
    timestamps: list[float],
) -> list[Keyframe]:
    """
    Extract frames at specific timestamps.

    Useful for aligning visual analysis with transcript segments.

    Args:
        video_path: Path to video file.
        timestamps: List of timestamps in seconds.

    Returns:
        List of Keyframe objects.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    keyframes = []

    for i, ts in enumerate(timestamps):
        frame = _extract_frame_at(cap, ts, fps)
        if frame is not None:
            keyframes.append(Keyframe(frame=frame, timestamp=ts, scene_index=i))

    cap.release()
    return keyframes


def _extract_frame_at(
    cap: cv2.VideoCapture,
    timestamp: float,
    fps: float,
) -> np.ndarray | None:
    """Extract a single frame at the given timestamp."""
    frame_num = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    return frame if ret else None
