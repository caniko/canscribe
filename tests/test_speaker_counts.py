from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from canscribe.backends import PyannoteCommunityBackend
from canscribe.pipeline import TranscriptionPipeline
from canscribe.types import (
    AsrResult,
    BackendSetupError,
    DiarizationSegment,
    TranscriptionRequest,
)


class FakeAsrBackend:
    name = "fake-asr"
    model_name = "fake-asr-model"

    def transcribe_chunk(
        self, audio_chunk: torch.Tensor, sample_rate: int
    ) -> AsrResult:
        return AsrResult(text="hello")


class FakeDiarizationBackend:
    name = "fake-diarization"
    model_name = "fake-diarization-model"

    def __init__(self) -> None:
        self.calls: list[dict[str, int | str | None]] = []

    def diarize(
        self,
        audio_path: str,
        *,
        speaker_count: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> tuple[list[DiarizationSegment], tuple[str, ...]]:
        self.calls.append(
            {
                "audio_path": audio_path,
                "speaker_count": speaker_count,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            }
        )
        return [DiarizationSegment(0.0, 1.0, "SPEAKER_00")], ()


def test_pipeline_forwards_exact_speaker_count(monkeypatch, tmp_path: Path) -> None:
    diarization_backend = FakeDiarizationBackend()
    _patch_pipeline_audio(monkeypatch, tmp_path)

    result = TranscriptionPipeline(
        asr_backend=FakeAsrBackend(),
        diarization_backend=diarization_backend,
    ).run(TranscriptionRequest(tmp_path / "meeting.wav", speaker_count=3))

    assert diarization_backend.calls == [
        {
            "audio_path": str(tmp_path / "meeting.wav"),
            "speaker_count": 3,
            "min_speakers": None,
            "max_speakers": None,
        }
    ]
    assert result.backend_metadata["speaker_count"] == "3"


def test_pipeline_forwards_speaker_range(monkeypatch, tmp_path: Path) -> None:
    diarization_backend = FakeDiarizationBackend()
    _patch_pipeline_audio(monkeypatch, tmp_path)

    result = TranscriptionPipeline(
        asr_backend=FakeAsrBackend(),
        diarization_backend=diarization_backend,
    ).run(
        TranscriptionRequest(
            tmp_path / "meeting.wav",
            min_speakers=2,
            max_speakers=5,
        )
    )

    assert diarization_backend.calls[0]["speaker_count"] is None
    assert diarization_backend.calls[0]["min_speakers"] == 2
    assert diarization_backend.calls[0]["max_speakers"] == 5
    assert result.backend_metadata["min_speakers"] == "2"
    assert result.backend_metadata["max_speakers"] == "5"


def test_pipeline_preserves_auto_speaker_count(monkeypatch, tmp_path: Path) -> None:
    diarization_backend = FakeDiarizationBackend()
    _patch_pipeline_audio(monkeypatch, tmp_path)

    result = TranscriptionPipeline(
        asr_backend=FakeAsrBackend(),
        diarization_backend=diarization_backend,
    ).run(TranscriptionRequest(tmp_path / "meeting.wav"))

    assert diarization_backend.calls[0]["speaker_count"] is None
    assert diarization_backend.calls[0]["min_speakers"] is None
    assert diarization_backend.calls[0]["max_speakers"] is None
    assert "speaker_count" not in result.backend_metadata
    assert "min_speakers" not in result.backend_metadata
    assert "max_speakers" not in result.backend_metadata


@pytest.mark.parametrize(
    ("transcription_request", "message"),
    [
        (TranscriptionRequest("meeting.wav", speaker_count=0), "positive integer"),
        (TranscriptionRequest("meeting.wav", min_speakers=0), "positive integer"),
        (
            TranscriptionRequest("meeting.wav", speaker_count=2, max_speakers=4),
            "cannot be combined",
        ),
        (
            TranscriptionRequest("meeting.wav", min_speakers=5, max_speakers=2),
            "less than or equal",
        ),
    ],
)
def test_pipeline_rejects_invalid_speaker_counts(
    transcription_request: TranscriptionRequest,
    message: str,
) -> None:
    with pytest.raises(BackendSetupError, match=message):
        TranscriptionPipeline(
            asr_backend=FakeAsrBackend(),
            diarization_backend=FakeDiarizationBackend(),
        ).run(transcription_request)


def test_pyannote_backend_passes_speaker_options_to_pipeline(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    _install_fake_progress_hook(monkeypatch)
    monkeypatch.setattr(
        "canscribe.backends._load_audio_tensor",
        lambda audio_path: (torch.zeros(1, 16000), 16000),
    )

    backend = PyannoteCommunityBackend(device="cpu")
    backend._pipeline = FakePyannotePipeline(calls)  # type: ignore[attr-defined]

    segments, warnings = backend.diarize(
        "meeting.wav",
        speaker_count=4,
        min_speakers=None,
        max_speakers=None,
    )

    assert warnings == ()
    assert segments == [DiarizationSegment(0.0, 1.0, "SPEAKER_00")]
    assert calls[0]["num_speakers"] == 4
    assert "min_speakers" not in calls[0]
    assert "max_speakers" not in calls[0]


def test_pyannote_backend_omits_speaker_options_in_auto_mode(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    _install_fake_progress_hook(monkeypatch)
    monkeypatch.setattr(
        "canscribe.backends._load_audio_tensor",
        lambda audio_path: (torch.zeros(1, 16000), 16000),
    )

    backend = PyannoteCommunityBackend(device="cpu")
    backend._pipeline = FakePyannotePipeline(calls)  # type: ignore[attr-defined]

    backend.diarize("meeting.wav")

    assert "num_speakers" not in calls[0]
    assert "min_speakers" not in calls[0]
    assert "max_speakers" not in calls[0]


def _patch_pipeline_audio(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "canscribe.pipeline.get_audio_path",
        lambda path: (str(tmp_path / "meeting.wav"), None),
    )
    monkeypatch.setattr(
        "canscribe.pipeline.load_mono_audio",
        lambda path: (torch.ones(16000), 16000),
    )
    monkeypatch.setattr(
        "canscribe.pipeline.save_transcript",
        lambda segments, input_path, output_path=None, debug=False: (
            tmp_path / "out.txt"
        ),
    )


class FakeProgressHook:
    def __enter__(self) -> FakeProgressHook:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _install_fake_progress_hook(monkeypatch) -> None:
    module_names = [
        "pyannote",
        "pyannote.audio",
        "pyannote.audio.pipelines",
        "pyannote.audio.pipelines.utils",
        "pyannote.audio.pipelines.utils.hook",
    ]
    for module_name in module_names:
        monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))

    hook_module = cast(Any, sys.modules["pyannote.audio.pipelines.utils.hook"])
    setattr(hook_module, "ProgressHook", FakeProgressHook)


class FakePyannotePipeline:
    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self.calls = calls

    def __call__(self, file: dict[str, Any], **kwargs: Any) -> FakeDiarizationResult:
        self.calls.append({"file": file, **kwargs})
        return FakeDiarizationResult()


class FakeDiarizationResult:
    exclusive_speaker_diarization = None

    def __init__(self) -> None:
        self.exclusive_speaker_diarization = FakeTimeline()


class FakeTimeline:
    def itertracks(self, *, yield_label: bool) -> list[tuple[FakeSegment, None, str]]:
        assert yield_label
        return [(FakeSegment(), None, "SPEAKER_00")]


class FakeSegment:
    start = 0.0
    end = 1.0
