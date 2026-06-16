from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from .types import CheckResult


def _get_runpath_rpath(so_path: Path) -> tuple[Optional[str], Optional[str]]:
    """Extract RUNPATH and RPATH from an ELF shared object."""
    try:
        result = subprocess.run(
            ["readelf", "-d", str(so_path)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except FileNotFoundError:
        return None, None

    runpath = None
    rpath = None
    for line in result.stdout.splitlines():
        m = re.match(
            r"\s*0x[0-9a-f]+\s+\(RUNPATH\)\s+Library runpath:\s*\[(.+)\]", line
        )
        if m:
            runpath = m.group(1)
        m = re.match(r"\s*0x[0-9a-f]+\s+\(RPATH\)\s+Library rpath:\s*\[(.+)\]", line)
        if m:
            rpath = m.group(1)

    return runpath, rpath


def _resolve_origin(path: str, so_path: Path) -> str:
    """Resolve $ORIGIN in a RUNPATH/RPATH string."""
    return path.replace("$ORIGIN", str(so_path.parent))


def check_library_resolution(so_name: str = "libdrm_amdgpu.so") -> List[CheckResult]:
    """Trace which copy of a given .so would be loaded based on RPATH/RUNPATH."""
    results: List[CheckResult] = []

    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    ld_preload = os.environ.get("LD_PRELOAD", "")

    if ld_preload:
        if so_name in ld_preload:
            results.append(
                CheckResult(
                    "INFO", f"{so_name} resolution", f"LD_PRELOAD contains {so_name}"
                )
            )

    # Find all copies in LD_LIBRARY_PATH
    ld_copies: list[Path] = []
    for entry in ld_path.split(":"):
        candidate = Path(entry) / so_name
        if candidate.exists():
            ld_copies.append(candidate)

    if ld_copies:
        results.append(
            CheckResult(
                "INFO",
                f"{so_name} on LD_LIBRARY_PATH",
                "; ".join(str(p) for p in ld_copies),
            )
        )
    else:
        results.append(
            CheckResult("INFO", f"{so_name} on LD_LIBRARY_PATH", "not found")
        )

    _check_rpath_of_parents(so_name, results)

    return results


def _check_rpath_of_parents(so_name: str, results: List[CheckResult]) -> None:
    """Check RPATH/RUNPATH of known ROCm parent libs for the target .so."""
    root = Path.cwd()
    parents_to_check = [
        root / ".venv/lib/python3.13/site-packages/torch/lib/libamdhip64.so",
        root / ".venv/lib/python3.13/site-packages/torch/lib/libdrm_amdgpu.so",
        root
        / ".venv/lib/python3.13/site-packages/triton/backends/amd/lib/libdrm_amdgpu.so",
    ]

    for path in parents_to_check:
        if not path.exists():
            continue
        runpath, rpath = _get_runpath_rpath(path)
        tag = "RPATH" if rpath else ("RUNPATH" if runpath else "no RPATH/RUNPATH")
        tag_detail = rpath or runpath or ""
        if tag_detail:
            resolved = _resolve_origin(tag_detail, path)
            results.append(
                CheckResult(
                    "INFO",
                    f"  {path.name}",
                    f"{tag}: {resolved}",
                )
            )
        else:
            results.append(CheckResult("INFO", f"  {path.name}", f"{tag}"))


def run_so_path_checks() -> List[CheckResult]:
    """Run all library-path-related checks."""
    results: List[CheckResult] = []
    results.extend(check_library_resolution("libdrm_amdgpu.so"))
    return results
