from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from torching.checks.environment import (
    check_key_env_vars,
    check_ld_library_path,
    check_python_path,
)
from torching.checks.gpu_libs import scan_bundled_libdrm
from torching.checks.types import CheckResult


class TestScanBundledLibdrm:
    def test_no_files_found(self, tmp_path: Path) -> None:
        results = scan_bundled_libdrm(tmp_path)
        assert results == []

    def test_finds_hardcoded_path(self, tmp_path: Path) -> None:
        so_dir = tmp_path / ".venv/lib/python3.13/site-packages/torch/lib"
        so_dir.mkdir(parents=True)
        so_file = so_dir / "libdrm_amdgpu.so"
        so_file.write_bytes(b"/opt/amdgpu/share/libdrm/amdgpu.ids\x00garbage")

        results = scan_bundled_libdrm(tmp_path)
        assert len(results) == 1
        lib_path, ids_path = results[0]
        assert lib_path == so_file
        assert ids_path == "/opt/amdgpu/share/libdrm/amdgpu.ids"

    def test_finds_in_triton_too(self, tmp_path: Path) -> None:
        so_dir = tmp_path / ".venv/lib/python3.13/site-packages/triton/backends/amd/lib"
        so_dir.mkdir(parents=True)
        so_file = so_dir / "libdrm_amdgpu.so"
        so_file.write_bytes(b"/opt/amdgpu/share/libdrm/amdgpu.ids\x00")

        results = scan_bundled_libdrm(tmp_path)
        assert len(results) == 1

    def test_ignores_non_matching_files(self, tmp_path: Path) -> None:
        so_dir = tmp_path / ".venv/lib/python3.13/site-packages/torch/lib"
        so_dir.mkdir(parents=True)
        so_file = so_dir / "libdrm_amdgpu.so"
        so_file.write_bytes(b"some random bytes without the path")

        results = scan_bundled_libdrm(tmp_path)
        assert results == []


class TestCheckLdLibraryPath:
    def test_missing_entry(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "torching.checks.environment.os.environ",
            {"LD_LIBRARY_PATH": "/nonexistent/path"},
        )
        results = check_ld_library_path()
        assert any(r.status == "WARN" for r in results)


class TestCheckKeyEnvVars:
    def test_reports_ok_for_set_var(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "torching.checks.environment.os.environ",
            {"HF_TOKEN": "abc123"},
        )
        results = check_key_env_vars()
        hf = [r for r in results if r.name == "HF_TOKEN"]
        assert len(hf) == 1
        assert hf[0].status == "OK"

    def test_reports_info_for_unset_var(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "torching.checks.environment.os.environ",
            {},
        )
        results = check_key_env_vars()
        hf = [r for r in results if r.name == "HF_TOKEN"]
        assert len(hf) == 1
        assert hf[0].status == "INFO"


class TestCheckPythonPath:
    def test_info_when_not_set(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "torching.checks.environment.os.environ",
            {},
        )
        results = check_python_path()
        assert results[0].status == "INFO"

    def test_ok_when_all_valid(self, tmp_path: Path, monkeypatch) -> None:
        d = tmp_path / "p"
        d.mkdir()
        monkeypatch.setattr(
            "torching.checks.environment.os.environ",
            {"PYTHONPATH": str(d)},
        )
        results = check_python_path()
        assert results[0].status == "OK"

    def test_warn_on_missing_entry(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "torching.checks.environment.os.environ",
            {"PYTHONPATH": "/does/not/exist"},
        )
        results = check_python_path()
        assert results[0].status == "WARN"


class TestCheckResultType:
    def test_slots(self) -> None:
        r = CheckResult("OK", "test", "detail")
        with pytest.raises(AttributeError):
            r.foo = "bar"  # type: ignore[attr-defined]

    def test_fields(self) -> None:
        r = CheckResult("OK", "test", "detail")
        assert r.status == "OK"
        assert r.name == "test"
        assert r.detail == "detail"

    def test_optional_fields(self) -> None:
        r = CheckResult("WARN", "test", "detail", explanation="because", fix="do x")
        assert r.explanation == "because"
        assert r.fix == "do x"

    def test_optional_fields_default_none(self) -> None:
        r = CheckResult("OK", "test", "detail")
        assert r.explanation is None
        assert r.fix is None


class TestCliInvocation:
    def test_torching_exits_zero(self) -> None:
        result = subprocess.run(
            ["torching"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

    def test_torching_verbose_exits_zero(self) -> None:
        result = subprocess.run(
            ["torching", "--verbose"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

    def test_torching_probe_exits_zero(self) -> None:
        result = subprocess.run(
            ["torching", "--probe"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0

    def test_canscribe_doctor_exits_zero(self) -> None:
        result = subprocess.run(
            ["canscribe", "doctor"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

    def test_canscribe_doctor_verbose_exits_zero(self) -> None:
        result = subprocess.run(
            ["canscribe", "doctor", "--verbose"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

    def test_canscribe_doctor_probe_exits_zero(self) -> None:
        result = subprocess.run(
            ["canscribe", "doctor", "--probe"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0
