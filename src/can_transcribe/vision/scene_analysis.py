"""
Scene analysis module using Qwen3-VL-2B-Thinking.

Provides general visual descriptions for frames without speaking faces,
such as screen shares, presentations, or environment shots.
"""

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from ..config import QWEN3_VL_MODEL, supports_flash_attention


@dataclass
class SceneDescription:
    """Result of scene analysis."""

    description: str
    timestamp: float


class SceneAnalyzer:
    """
    Analyzes video frames/segments using Qwen3-VL-2B-Thinking.

    Features:
    - Native video support with timestamp alignment
    - 32-language OCR for screen shares
    - Spatial reasoning for complex scenes
    - Auto-detects Flash Attention 2 support
    """

    def __init__(
        self,
        model_name: str = QWEN3_VL_MODEL,
        device: str = "cuda",
        max_new_tokens: int = 256,
    ):
        """
        Initialize scene analyzer.

        Args:
            model_name: Qwen3-VL model to use.
            device: Device to run on ("cuda" or "cpu").
            max_new_tokens: Maximum tokens to generate.
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._use_flash = supports_flash_attention()

    @property
    def model(self):
        """Lazy-load Qwen3-VL model."""
        if self._model is None:
            print(f"🔧 Loading {self.model_name} (Flash Attention: {self._use_flash})...")

            if self._use_flash:
                self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                )
            else:
                self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    attn_implementation="sdpa",
                    device_map="auto" if self.device == "cuda" else None,
                )
                if self.device != "cuda":
                    self._model = self._model.to(self.device)

        return self._model

    @property
    def processor(self):
        """Lazy-load processor."""
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.model_name)
        return self._processor

    def describe_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        prompt: str | None = None,
    ) -> SceneDescription:
        """
        Generate a description for a single frame.

        Args:
            frame: BGR image as numpy array (from cv2).
            timestamp: Timestamp of the frame in seconds.
            prompt: Custom prompt for description. Defaults to general scene description.

        Returns:
            SceneDescription with generated text.
        """
        # Convert BGR to RGB PIL Image
        rgb_frame = frame[:, :, ::-1]  # BGR to RGB
        image = Image.fromarray(rgb_frame)

        if prompt is None:
            prompt = (
                "Describe what you see in this image concisely. "
                "Focus on: people's actions and posture, any text on screen, "
                "the setting or environment. Be brief and factual."
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process with chat template
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # Decode, removing input tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Clean up thinking tokens if present
        # Qwen3-VL-Thinking may include <think>...</think> blocks
        if "<think>" in output_text:
            # Extract only the final answer after thinking
            parts = output_text.split("</think>")
            if len(parts) > 1:
                output_text = parts[-1].strip()

        return SceneDescription(
            description=output_text.strip(),
            timestamp=timestamp,
        )

    def describe_video_segment(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        prompt: str | None = None,
    ) -> SceneDescription:
        """
        Generate a description for a video segment.

        Uses Qwen3-VL's native video understanding.

        Args:
            video_path: Path to video file.
            start_time: Start timestamp in seconds.
            end_time: End timestamp in seconds.
            prompt: Custom prompt for description.

        Returns:
            SceneDescription with generated text.
        """
        if prompt is None:
            prompt = (
                "Describe what happens in this video segment concisely. "
                "Focus on: actions, any text shown, changes in the scene. "
                "Be brief and factual."
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{video_path}",
                        "video_start": start_time,
                        "video_end": end_time,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process with chat template
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Clean up thinking tokens
        if "<think>" in output_text:
            parts = output_text.split("</think>")
            if len(parts) > 1:
                output_text = parts[-1].strip()

        return SceneDescription(
            description=output_text.strip(),
            timestamp=start_time,
        )
