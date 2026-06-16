from __future__ import annotations

from typing import List

import typer

from .checks.environment import (
    check_key_env_vars,
    check_ld_library_path,
    check_python_path,
)
from .checks.gpu_libs import (
    check_bundled_libdrm_paths,
    check_bundled_libdrm_version_mismatch,
)
from .checks.lib_path import run_so_path_checks
from .checks.probe import probe_gpu_runtime
from .checks.types import CheckResult


def run_checks(
    probe: bool = False,
) -> List[CheckResult]:
    results: List[CheckResult] = []
    results.extend(check_ld_library_path())
    results.extend(check_python_path())
    results.extend(check_key_env_vars())
    results.extend(check_bundled_libdrm_paths())
    results.extend(check_bundled_libdrm_version_mismatch())
    results.extend(run_so_path_checks())
    if probe:
        results.extend(probe_gpu_runtime())
    return results


def print_results(
    results: List[CheckResult],
    verbose: bool = False,
) -> None:
    for r in results:
        print(f"[{r.status:5s}] {r.name}: {r.detail}")
        if verbose and (r.explanation or r.fix):
            if r.explanation:
                for line in r.explanation.split("\n"):
                    print(f"       {line}")
            if r.fix:
                print(f"       fix: {r.fix}")


def main(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show explanations and fix suggestions for each check result",
    ),
    probe: bool = typer.Option(
        False,
        "--probe",
        "-p",
        help="Run a GPU kernel probe to check runtime behavior (requires torch + GPU)",
    ),
) -> None:
    results = run_checks(probe=probe)
    print_results(results, verbose=verbose)
    has_fail = any(r.status == "FAIL" for r in results)
    raise SystemExit(1 if has_fail else 0)


def cli() -> None:
    typer.run(main)


if __name__ == "__main__":
    cli()
