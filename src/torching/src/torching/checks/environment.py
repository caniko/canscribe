from __future__ import annotations

import os
from pathlib import Path
from typing import List

from .types import CheckResult, Status


def check_ld_library_path() -> List[CheckResult]:
    """Validate all entries in LD_LIBRARY_PATH exist."""
    results: List[CheckResult] = []
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")

    if not ld_path:
        results.append(CheckResult("INFO", "LD_LIBRARY_PATH", "not set"))
        return results

    entries = ld_path.split(":")
    missing: List[str] = []
    ok_count = 0
    for entry in entries:
        if entry and Path(entry).is_dir():
            ok_count += 1
        elif entry:
            missing.append(entry)

    total = len([e for e in entries if e])
    detail = (
        f"{ok_count} valid, {len(missing)} missing"
        if missing
        else f"{ok_count} entries, all valid"
    )
    status: Status = "OK" if not missing else "WARN"
    results.append(CheckResult(status, "LD_LIBRARY_PATH", detail))

    for m in missing:
        results.append(CheckResult("WARN", "  missing entry", m))

    return results


def check_key_env_vars() -> List[CheckResult]:
    """Check presence of important environment variables."""
    results: List[CheckResult] = []

    checks = [
        ("HF_TOKEN", "HuggingFace token for model access"),
        ("ROCR_VISIBLE_DEVICES", "ROCm visible devices"),
        ("HIP_VISIBLE_DEVICES", "HIP visible devices"),
        ("CUDA_VISIBLE_DEVICES", "CUDA visible devices"),
    ]

    for var, purpose in checks:
        val = os.environ.get(var, "")
        if val:
            results.append(CheckResult("OK", var, purpose))
        else:
            results.append(CheckResult("INFO", var, f"not set ({purpose})"))

    return results


def check_python_path() -> List[CheckResult]:
    """Check PYTHONPATH entries for validity."""
    results: List[CheckResult] = []
    pp = os.environ.get("PYTHONPATH", "")

    if not pp:
        results.append(CheckResult("INFO", "PYTHONPATH", "not set"))
        return results

    entries = pp.split(":")
    missing: List[str] = []
    for entry in entries:
        if entry and not Path(entry).exists():
            missing.append(entry)

    if missing:
        results.append(
            CheckResult(
                "WARN",
                "PYTHONPATH",
                f"{len(entries)} entries, {len(missing)} missing",
            )
        )
        for m in missing:
            results.append(CheckResult("WARN", "  missing entry", m))
    else:
        results.append(
            CheckResult("OK", "PYTHONPATH", f"{len(entries)} entries, all valid")
        )

    return results
