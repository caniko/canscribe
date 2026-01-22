"""
Frame router for tiered visual analysis.

Uses fast face detection to route frames:
- Speaking face detected → specialized face/emotion analysis
- No speaking face → general VLM scene description
"""

from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FrameType(Enum):
    """Classification of frame content for routing."""

    SPEAKING_FACE = auto()  # Face detected with mouth movement
    STATIC_FACE = auto()  # Face detected, not speaking
    NO_FACE = auto()  # No face, use general VLM


@dataclass
class RoutingResult:
    """Result of frame routing decision."""

    frame_type: FrameType
    faces: list  # List of detected face objects from InsightFace
    frame: np.ndarray  # The original frame


class FrameRouter:
    """
    Routes video frames to appropriate analysis pipeline.

    Uses InsightFace SCRFD for fast face detection (~3-10ms),
    then checks mouth landmarks for speaking detection.
    """

    def __init__(self, device: str = "cuda", detection_threshold: float = 0.5):
        """
        Initialize the frame router.

        Args:
            device: Device to run on ("cuda" or "cpu").
            detection_threshold: Confidence threshold for face detection.
        """
        self.device = device
        self.detection_threshold = detection_threshold
        self._face_app: FaceAnalysis | None = None
        self._prev_mouth_heights: dict[int, list[float]] = {}  # Track mouth movement per face

    @property
    def face_app(self) -> FaceAnalysis:
        """Lazy-load InsightFace model."""
        if self._face_app is None:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device == "cuda"
                else ["CPUExecutionProvider"]
            )
            self._face_app = FaceAnalysis(
                name="buffalo_sc",  # Smaller model for fast detection
                providers=providers,
            )
            self._face_app.prepare(ctx_id=0 if self.device == "cuda" else -1)
        return self._face_app

    def _get_mouth_height(self, face) -> float | None:
        """
        Calculate mouth opening height from face landmarks.

        Uses the 5-point landmarks if available, or falls back to
        detecting significant mouth movement from landmark positions.
        """
        if face.landmark_2d_106 is None:
            return None

        # Landmark indices for mouth (106-point model)
        # Upper lip: 84-95, Lower lip: 96-103
        upper_lip_idx = 89  # Center of upper lip
        lower_lip_idx = 99  # Center of lower lip

        landmarks = face.landmark_2d_106
        upper = landmarks[upper_lip_idx]
        lower = landmarks[lower_lip_idx]

        return float(np.linalg.norm(upper - lower))

    def _is_speaking(self, face, face_id: int) -> bool:
        """
        Detect if a face is speaking based on mouth movement.

        Tracks mouth height over recent frames and detects variation.
        """
        mouth_height = self._get_mouth_height(face)
        if mouth_height is None:
            return False

        # Initialize tracking for new faces
        if face_id not in self._prev_mouth_heights:
            self._prev_mouth_heights[face_id] = []

        history = self._prev_mouth_heights[face_id]
        history.append(mouth_height)

        # Keep only last 5 measurements
        if len(history) > 5:
            history.pop(0)

        # Need at least 3 measurements to detect movement
        if len(history) < 3:
            return False

        # Check for significant variation (speaking causes mouth movement)
        std_dev = np.std(history)
        return bool(std_dev > 2.0)  # Threshold for "speaking" vs static

    def route(self, frame: np.ndarray) -> RoutingResult:
        """
        Analyze a frame and determine routing.

        Args:
            frame: BGR image as numpy array (from cv2).

        Returns:
            RoutingResult with frame type and detected faces.
        """
        # Detect faces
        faces = self.face_app.get(frame)

        if not faces:
            return RoutingResult(
                frame_type=FrameType.NO_FACE,
                faces=[],
                frame=frame,
            )

        # Check if any face is speaking
        for i, face in enumerate(faces):
            if self._is_speaking(face, i):
                return RoutingResult(
                    frame_type=FrameType.SPEAKING_FACE,
                    faces=faces,
                    frame=frame,
                )

        return RoutingResult(
            frame_type=FrameType.STATIC_FACE,
            faces=faces,
            frame=frame,
        )

    def reset_tracking(self) -> None:
        """Reset mouth movement tracking (call between videos)."""
        self._prev_mouth_heights.clear()
