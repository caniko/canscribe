"""Check the published wheel, without relying on checkout-local uv constraints."""

import sys
import tomllib
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

from packaging.markers import Marker
from packaging.requirements import Requirement

with ZipFile(sys.argv[1]) as wheel:
    metadata_files = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
    assert len(metadata_files) == 1, metadata_files
    metadata = BytesParser().parsebytes(wheel.read(metadata_files[0]))

requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
for extra in ("", "cpu", "nvidia", "amd", "apple"):
    active = [
        requirement
        for requirement in requirements
        if requirement.marker is None or requirement.marker.evaluate({"extra": extra})
    ]
    for name, supported, incompatible in (
        ("torch", "2.9.1", "2.10.0"),
        ("torchaudio", "2.9.1", "2.10.0"),
        ("torchvision", "0.24.1", "0.25.0"),
    ):
        if name == "torchvision" and not extra:
            continue
        matches = [requirement for requirement in active if requirement.name == name]
        assert matches, (extra, name)
        assert all(supported in requirement.specifier for requirement in matches), matches
        assert any(incompatible not in requirement.specifier for requirement in matches), matches
        assert all("+" not in str(requirement.specifier) and requirement.url is None for requirement in matches), matches

# The CPU index also contains a higher-sorting local version with only an ARM
# Windows wheel. Ensure supported x86_64 routes select compatible wheels instead.
sources = tomllib.loads(Path("pyproject.toml").read_text())["tool"]["uv"]["sources"]["torchvision"]
for platform, machine, wheel_platform in (
    ("linux", "x86_64", "manylinux_2_28_x86_64"),
    ("win32", "AMD64", "win_amd64"),
):
    active = [
        source for source in sources
        if source.get("extra") == "cpu" and (
            "marker" not in source or Marker(source["marker"]).evaluate(
                {"sys_platform": platform, "platform_machine": machine}
            )
        )
    ]
    assert len(active) == 1, (platform, active)
    assert active[0].get("url", "").endswith(f"%2Bcpu-cp313-cp313-{wheel_platform}.whl"), active
