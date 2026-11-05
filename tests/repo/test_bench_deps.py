"""The optional `bench` dependency group stays out of setup and free of GPL code."""

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

# GPL grapheme-to-phoneme packages and the package that pulls them in (ADR 0004).
GPL_G2P = {"phonemizer", "phonemizer-fork", "espeakng-loader", "kokoro-onnx"}
BENCH_PACKAGES = {
    "faster-whisper",
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12",
    "onnxruntime",
    "misaki",
    "spacy",
    "en-core-web-sm",
    "num2words",
    "soundfile",
    "numpy",
    "huggingface-hub",
    "nvidia-ml-py",
}


def pyproject(repo_root: Path) -> dict[str, Any]:
    return tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))


def lock(repo_root: Path) -> dict[str, Any]:
    return tomllib.loads((repo_root / "uv.lock").read_text(encoding="utf-8"))


def requirement_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
    assert match, requirement
    return re.sub(r"[-_.]+", "-", match.group(0)).lower()


def recipe_body(repo_root: Path, name: str) -> str:
    text = (repo_root / "justfile").read_text(encoding="utf-8")
    body = re.search(rf"^{name}\b[^\n]*:\n((?:[ \t]+\S[^\n]*\n)+)", text, re.M)
    assert body, f"recipe `{name}` has no body"
    return body.group(1)


def test_bench_group_holds_the_benchmark_dependencies(repo_root: Path) -> None:
    group = pyproject(repo_root)["dependency-groups"]["bench"]
    assert {requirement_name(r) for r in group} == BENCH_PACKAGES


def test_bench_group_is_not_a_default_group(repo_root: Path) -> None:
    defaults = pyproject(repo_root)["tool"]["uv"].get("default-groups", ["dev"])
    assert defaults == ["dev"]


def test_setup_does_not_install_the_bench_group(repo_root: Path) -> None:
    body = recipe_body(repo_root, "setup")
    assert "bench" not in body
    assert "--all-groups" not in body


def test_bench_group_is_locked(repo_root: Path) -> None:
    root = next(p for p in lock(repo_root)["package"] if p["name"] == "parlora-workspace")
    locked = {d["name"] for d in root["dev-dependencies"]["bench"]}
    assert locked == BENCH_PACKAGES


def test_lockfile_has_no_gpl_g2p_packages(repo_root: Path) -> None:
    names = {p["name"] for p in lock(repo_root)["package"]}
    assert not names & GPL_G2P, f"GPL G2P packages in uv.lock: {sorted(names & GPL_G2P)}"


def test_misaki_is_installed_without_extras(repo_root: Path) -> None:
    group = pyproject(repo_root)["dependency-groups"]["bench"]
    misaki = next(r for r in group if requirement_name(r) == "misaki")
    assert "[" not in misaki, "misaki extras pull in GPL packages (ADR 0004)"
