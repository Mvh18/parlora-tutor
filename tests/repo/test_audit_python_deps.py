"""The Python licence gate: denied licences, reviewed exceptions by name."""

import pytest

from scripts.audit_python_deps import (
    REVIEWED_LGPL,
    REVIEWED_PROPRIETARY,
    licence_problems,
)

pytestmark = pytest.mark.unit


def row(name: str, licence: str) -> dict[str, str]:
    return {"Name": name, "Version": "1.0", "License": licence}


def test_permissive_licences_pass() -> None:
    report = [row("fastapi", "MIT License"), row("numpy", "BSD License"), row("x", "Apache-2.0")]
    assert licence_problems(report) == []


@pytest.mark.parametrize(
    "licence",
    [
        "GNU General Public License v3 (GPLv3)",
        "GPL-3.0-or-later",
        "AGPL-3.0",
        "UNKNOWN",
        "Other/Proprietary License",
        "NVIDIA Proprietary Software",
    ],
)
def test_denied_licences_fail_naming_the_package(licence: str) -> None:
    assert licence_problems([row("some-package", licence)]) == [f"some-package: {licence}"]


@pytest.mark.parametrize("name", sorted(REVIEWED_LGPL))
def test_reviewed_lgpl_packages_pass(name: str) -> None:
    licence = "GNU Lesser General Public License v2 or later (LGPLv2+)"
    assert licence_problems([row(name, licence)]) == []


@pytest.mark.parametrize(
    "name",
    [
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12",
        "nvidia-cuda-nvrtc-cu12",
        "nvidia-cublas",
        "nvidia-cudnn-cu13",
        "nvidia-cuda-runtime",
        "nvidia-nvjitlink",
    ],
)
def test_reviewed_nvidia_cuda_wheels_pass(name: str) -> None:
    assert licence_problems([row(name, "Other/Proprietary License")]) == []


def test_proprietary_licence_outside_the_list_fails() -> None:
    report = [row("nvidia-ml-py", "Other/Proprietary License"), row("acme-sdk", "Proprietary")]
    assert licence_problems(report) == [
        "nvidia-ml-py: Other/Proprietary License",
        "acme-sdk: Proprietary",
    ]


def test_cuda_exception_does_not_excuse_a_gpl_licence() -> None:
    assert licence_problems([row("nvidia-cublas-cu12", "GPL-3.0")]) == [
        "nvidia-cublas-cu12: GPL-3.0"
    ]


def test_names_are_compared_normalised() -> None:
    assert licence_problems([row("NVIDIA_cuBLAS_cu12", "Other/Proprietary License")]) == []


def test_every_exception_has_a_justification() -> None:
    reviewed = {**REVIEWED_LGPL, **REVIEWED_PROPRIETARY}
    assert {"soxr", "num2words", "nvidia-*-cu12", "nvidia-cudnn-cu13"} <= set(reviewed)
    assert not any(name.startswith("nvidia-ml") for name in reviewed)
    assert all(len(reason) > 20 for reason in reviewed.values())
