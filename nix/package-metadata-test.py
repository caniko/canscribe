"""Check the published wheel, without relying on checkout-local uv constraints."""

import sys
from email.parser import BytesParser
from zipfile import ZipFile

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
