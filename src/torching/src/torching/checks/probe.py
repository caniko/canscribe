from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List

from .types import CheckResult


def _detect_venv_dir() -> Path | None:
    """Find the venv dir by looking at sys.prefix."""
    prefix = Path(sys.prefix)
    if (prefix / "pyvenv.cfg").exists():
        return prefix
    # Check if we're in a .venv
    parent = prefix.parent
    if prefix.name == ".venv" or (parent / ".venv").exists():
        return prefix
    return None


def probe_gpu_runtime() -> List[CheckResult]:
    """Run a minimal torch GPU workload and check for runtime issues.

    Requires torch to be installed. Gracefully handles torch not being
    available.
    """
    results: List[CheckResult] = []

    try:
        import torch as _torch
    except ImportError:
        results.append(
            CheckResult("INFO", "Runtime probe", "torch not available — skipping")
        )
        return results

    if not _torch.cuda.is_available():
        results.append(
            CheckResult("INFO", "Runtime probe", "no GPU available — skipping")
        )
        return results

    # Run a small GPU kernel in a subprocess to capture stderr + /proc/self/maps
    probe_code = """
import torch
import sys

# Run a tiny GPU workload
x = torch.ones((1,), device='cuda')
y = x + 1
torch.cuda.synchronize()
result = float(y.cpu()[0])

# Read /proc/self/maps for libdrm_amdgpu
maps_lines = []
with open('/proc/self/maps') as f:
    for line in f:
        if 'libdrm_amdgpu' in line:
            maps_lines.append(line.strip())

# Print structured output
print(f'GPU_RESULT={result}')
print(f'GPU_NAME={torch.cuda.get_device_name(0)}')
print(f'GPU_MEMORY={torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB')
for ml in maps_lines:
    print(f'MAPS:{ml}')
    break  # just first entry
"""

    probe = subprocess.run(
        [sys.executable, "-c", probe_code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    # Count stderr lines with amdgpu.ids
    stderr_ids_lines = [
        line for line in probe.stderr.splitlines() if "amdgpu.ids" in line
    ]

    # Parse stdout
    stdout_lines = probe.stdout.splitlines()
    gpu_result = ""
    gpu_name = ""
    gpu_memory = ""
    loaded_lib = ""
    for line in stdout_lines:
        if line.startswith("GPU_RESULT="):
            gpu_result = line[len("GPU_RESULT=") :]
        elif line.startswith("GPU_NAME="):
            gpu_name = line[len("GPU_NAME=") :]
        elif line.startswith("GPU_MEMORY="):
            gpu_memory = line[len("GPU_MEMORY=") :]
        elif line.startswith("MAPS:"):
            loaded_lib = (
                line[len("MAPS:") :].rsplit("/", 1)[-1]
                if "/" in line
                else line[len("MAPS:") :]
            )

    if probe.returncode != 0:
        detail = probe.stderr.strip().splitlines()
        last = detail[-1] if detail else f"exit {probe.returncode}"
        results.append(
            CheckResult("FAIL", "Runtime probe", f"GPU kernel failed: {last}")
        )
        return results

    results.append(
        CheckResult(
            "OK",
            "GPU kernel",
            f"{gpu_name}, {gpu_memory} — result={gpu_result}",
        )
    )

    if stderr_ids_lines:
        n = len(stderr_ids_lines)
        results.append(
            CheckResult(
                "WARN" if n > 0 else "OK",
                "Runtime stderr",
                f"amdgpu.ids error appears ({n} line{'s' if n > 1 else ''})",
                explanation=(
                    "The bundled libdrm_amdgpu.so prints this message to stderr "
                    "each time it fails to open the hardcoded path. "
                    "The error is harmless — GPU operations work correctly — "
                    "but it is noisy and confusing."
                ),
                fix=(
                    "Create a symlink so the bundled lib finds the file: "
                    "sudo mkdir -p /opt/amdgpu/share/libdrm && "
                    "sudo ln -s <nix-store-path>/share/libdrm/amdgpu.ids "
                    "/opt/amdgpu/share/libdrm/amdgpu.ids"
                ),
            )
        )
    else:
        results.append(CheckResult("OK", "Runtime stderr", "no amdgpu.ids errors"))

    if loaded_lib:
        # Check if loaded lib is the bundled one (from venv) or nixpkgs
        is_bundled = "site-packages" in (stdout_lines[0] if stdout_lines else "")
        # Actually let's extract the full path from maps
        for line in stdout_lines:
            if line.startswith("MAPS:"):
                full_path = line[len("MAPS:") :]
                # Extract path after last space
                parts = full_path.split()
                if parts:
                    loaded_path = parts[-1]
                else:
                    loaded_path = full_path
                break
        else:
            loaded_path = loaded_lib

        venv_dir = _detect_venv_dir()
        if venv_dir and "site-packages" in loaded_path:
            explanation = (
                "libamdhip64.so has RPATH $ORIGIN, which resolves to torch/lib/ "
                "at load time. The dynamic linker checks RPATH before "
                "LD_LIBRARY_PATH, so the bundled (venv) libdrm_amdgpu.so is "
                "loaded instead of the nixpkgs one — even though nixpkgs has "
                "the correct path."
            )
            fix = "See the 'Bundled libdrm' WARN above for a suggested fix."
            results.append(
                CheckResult(
                    "WARN",
                    "Runtime libdrm loading",
                    f"{loaded_path} loaded (bundled, not nixpkgs)",
                    explanation=explanation,
                    fix=fix,
                )
            )
        else:
            results.append(CheckResult("OK", "Runtime libdrm loading", loaded_path))

    return results
