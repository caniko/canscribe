from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List

from .types import CheckResult, Status


def _find_project_root() -> Path:
    """Walk up from sys.prefix / cwd to find the repo root with pyproject.toml."""
    candidates = [
        Path(sys.prefix).parent,  # .venv/
        Path.cwd(),
    ]
    for base in candidates:
        for parent in [base] + list(base.parents):
            if (parent / "pyproject.toml").exists():
                return parent
    return Path.cwd()


_BUNDLED_PATTERNS = [
    "torch/lib/libdrm_amdgpu.so",
    "triton/backends/amd/lib/libdrm_amdgpu.so",
]


def scan_bundled_libdrm(root: Path) -> List[tuple[Path, str]]:
    """Find bundled libdrm_amdgpu.so files and the hardcoded paths inside them."""
    results: List[tuple[Path, str]] = []
    for pattern in _BUNDLED_PATTERNS:
        full = root / ".venv/lib/python3.13/site-packages" / pattern
        if not full.exists():
            continue
        try:
            output = subprocess.run(
                ["strings", str(full)],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except FileNotFoundError:
            try:
                with open(full, "rb") as f:
                    data = f.read()
                output_str = data.decode("latin-1")
            except Exception:
                continue
        else:
            output_str = output.stdout

        for line in output_str.splitlines():
            if "amdgpu.ids" in line and line.startswith("/"):
                if (full, line) not in results:
                    results.append((full, line))
    return results


def check_bundled_libdrm_paths() -> List[CheckResult]:
    """Check if bundled libdrm_amdgpu.so has stale hardcoded /opt/amdgpu/ paths."""
    root = _find_project_root()
    bundles = scan_bundled_libdrm(root)
    results: List[CheckResult] = []

    if not bundles:
        results.append(
            CheckResult("INFO", "Bundled libdrm", "no bundled libdrm_amdgpu.so found")
        )
        return results

        # Find the nixpkgs libdrm path on LD_LIBRARY_PATH for the fix suggestion
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    nix_ids_path: str | None = None
    for entry in ld_path.split(":"):
        candidate = Path(entry) / "libdrm_amdgpu.so"
        if candidate.exists():
            nix_pkg_root = Path(entry).parent
            candidate_ids = nix_pkg_root / "share/libdrm/amdgpu.ids"
            if candidate_ids.exists():
                nix_ids_path = str(candidate_ids)
                break

    for lib_path, hardcoded_path in bundles:
        rel_lib = os.path.relpath(str(lib_path), str(root))
        exists = os.path.exists(hardcoded_path)
        if exists:
            results.append(
                CheckResult(
                    "OK",
                    "Bundled libdrm",
                    f"{rel_lib} → {hardcoded_path} (found)",
                )
            )
        else:
            explanation = (
                "libamdhip64.so and libdrm_amdgpu.so both use RPATH $ORIGIN, "
                "which resolves to torch/lib/ at load time. "
                "The dynamic linker checks RPATH before LD_LIBRARY_PATH, "
                "so the bundled (venv) lib is always loaded instead of the "
                "nixpkgs version — even though the nixpkgs lib is on "
                "LD_LIBRARY_PATH with the correct amdgpu.ids path."
            )
            fix = None
            if nix_ids_path:
                fix = (
                    f"sudo mkdir -p /opt/amdgpu/share/libdrm && "
                    f"sudo ln -s {nix_ids_path} /opt/amdgpu/share/libdrm/amdgpu.ids"
                )
            results.append(
                CheckResult(
                    "WARN",
                    "Bundled libdrm",
                    f"{rel_lib} has stale path {hardcoded_path} → file not found",
                    explanation=explanation,
                    fix=fix,
                )
            )

    return results


def check_bundled_libdrm_version_mismatch() -> List[CheckResult]:
    """Compare nixpkgs-provided libdrm version with bundled version."""
    root = _find_project_root()
    results: List[CheckResult] = []

    bundled_patterns = [
        root / ".venv/lib/python3.13/site-packages/torch/lib/libdrm_amdgpu.so",
        root
        / ".venv/lib/python3.13/site-packages/triton/backends/amd/lib/libdrm_amdgpu.so",
    ]

    bundled_vers: List[str] = []
    for path in bundled_patterns:
        if path.exists():
            ver = _extract_so_version(path)
            if ver:
                bundled_vers.append(f"{path.name} ({ver})")

    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    nix_vers: List[str] = []
    for entry in ld_path.split(":"):
        candidate = Path(entry) / "libdrm_amdgpu.so"
        if candidate.exists():
            ver = _extract_so_version(candidate)
            if ver:
                nix_vers.append(f"{candidate} ({ver})")

    if not bundled_vers and not nix_vers:
        return results

    if nix_vers:
        for entry in ld_path.split(":"):
            candidate = Path(entry) / "libdrm_amdgpu.so"
            if candidate.exists():
                ver = _extract_so_version(candidate)
                rel = os.path.relpath(str(candidate), str(root))
                results.append(
                    CheckResult(
                        "INFO",
                        "libdrm (nixpkgs)",
                        f"{rel} ({ver})" if ver else rel,
                    )
                )

    if bundled_vers:
        results.append(
            CheckResult(
                "INFO",
                "libdrm (bundled)",
                "; ".join(bundled_vers),
                explanation="These bundled libs are loaded at runtime due to RPATH $ORIGIN, "
                "bypassing the nixpkgs version on LD_LIBRARY_PATH.",
            )
        )

    return results


def _extract_so_version(path: Path) -> str | None:
    """Extract version string from an .so file using `strings`."""
    try:
        output = subprocess.run(
            ["strings", str(path)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        for line in output.stdout.splitlines():
            line = line.strip()
            if re.match(r"^\d+\.\d+\.\d+$", line) and line.count(".") == 2:
                return line
    except Exception:
        pass
    return None
