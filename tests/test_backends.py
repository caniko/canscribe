from __future__ import annotations

import pytest

from canscribe.backends import (
    MoonshineAsrBackend,
    ParakeetAsrBackend,
    PyannoteCommunityBackend,
    create_asr_backend,
    create_diarization_backend,
)
from canscribe.config import MOONSHINE_MODEL, PARAKEET_MODEL, PYANNOTE_MODEL
from canscribe.types import BackendSetupError


def test_create_asr_backend_uses_parakeet_default_model() -> None:
    backend = create_asr_backend("parakeet", device="cpu")

    assert isinstance(backend, ParakeetAsrBackend)
    assert backend.name == "parakeet"
    assert backend.model_name == PARAKEET_MODEL


def test_create_asr_backend_supports_moonshine_compatibility_model() -> None:
    backend = create_asr_backend("moonshine", device="cpu")

    assert isinstance(backend, MoonshineAsrBackend)
    assert backend.name == "moonshine"
    assert backend.model_name == MOONSHINE_MODEL


def test_create_asr_backend_rejects_unknown_backend() -> None:
    with pytest.raises(BackendSetupError, match="Unknown ASR backend"):
        create_asr_backend("whisper", device="cpu")


def test_create_diarization_backend_uses_pyannote_default_model() -> None:
    backend = create_diarization_backend("pyannote-community", device="cpu")

    assert isinstance(backend, PyannoteCommunityBackend)
    assert backend.name == "pyannote-community"
    assert backend.model_name == PYANNOTE_MODEL


def test_create_diarization_backend_rejects_unknown_backend() -> None:
    with pytest.raises(BackendSetupError, match="Unknown diarization backend"):
        create_diarization_backend("unknown", device="cpu")
