"""Security baseline: dependency updates, disclosure policy, forbidden files, ruleset."""

import itertools
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REQUIRED_ECOSYSTEMS = {"uv", "npm", "docker", "docker-compose", "github-actions"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
MODEL_SUFFIXES = {".gguf", ".safetensors", ".onnx", ".pt", ".bin"}
GPL_PACKAGES = {"phonemizer", "phonemizer-fork", "espeakng-loader"}
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path.name} is not a mapping"
    return data


def tracked_files(repo_root: Path) -> list[PurePosixPath]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout
    return [PurePosixPath(name) for name in out.split("\0") if name]


def is_synthetic_fixture(path: PurePosixPath) -> bool:
    return any(a == "fixtures" and b == "synthetic" for a, b in itertools.pairwise(path.parts))


def test_dependabot_covers_all_ecosystems_with_cooldown(repo_root: Path) -> None:
    config = load_yaml(repo_root / ".github" / "dependabot.yml")
    updates = config["updates"]
    assert {u["package-ecosystem"] for u in updates} >= REQUIRED_ECOSYSTEMS
    without_cooldown = [u["package-ecosystem"] for u in updates if "cooldown" not in u]
    assert not without_cooldown, f"ecosystems without cooldown: {without_cooldown}"


def test_dependabot_version_updates_are_monthly(repo_root: Path) -> None:
    config = load_yaml(repo_root / ".github" / "dependabot.yml")
    not_monthly = [
        u["package-ecosystem"] for u in config["updates"] if u["schedule"]["interval"] != "monthly"
    ]
    assert not not_monthly, f"ecosystems not on a monthly schedule: {not_monthly}"


def test_security_policy_routes_privately(repo_root: Path) -> None:
    text = (repo_root / "SECURITY.md").read_text(encoding="utf-8")
    assert "private vulnerability reporting" in text.lower()
    assert not EMAIL.search(text), "SECURITY.md must not publish an email address"


def test_no_audio_outside_synthetic_fixtures(repo_root: Path) -> None:
    offenders = [
        str(p)
        for p in tracked_files(repo_root)
        if p.suffix.lower() in AUDIO_SUFFIXES and not is_synthetic_fixture(p)
    ]
    assert not offenders, f"audio outside fixtures/synthetic/: {offenders}"


def test_no_model_weights(repo_root: Path) -> None:
    offenders = [str(p) for p in tracked_files(repo_root) if p.suffix.lower() in MODEL_SUFFIXES]
    assert not offenders, f"model weights must not be committed: {offenders}"


def test_ruleset_protects_the_default_branch(repo_root: Path) -> None:
    ruleset = json.loads((repo_root / ".github" / "rulesets" / "main.json").read_text("utf-8"))
    assert ruleset["target"] == "branch"
    assert ruleset["enforcement"] == "active"
    assert ruleset["conditions"]["ref_name"]["include"] == ["~DEFAULT_BRANCH"]
    rules = {rule["type"]: rule.get("parameters", {}) for rule in ruleset["rules"]}
    assert {"deletion", "non_fast_forward", "required_linear_history"} <= rules.keys()
    assert rules["pull_request"]["allowed_merge_methods"] == ["squash"]
    assert rules["pull_request"]["required_review_thread_resolution"] is True
    checks = [c["context"] for c in rules["required_status_checks"]["required_status_checks"]]
    assert checks == ["ci-ok"]


def test_dependency_review_denies_gpl_g2p_packages(repo_root: Path) -> None:
    config = load_yaml(repo_root / ".github" / "dependency-review-config.yml")
    denied = {purl.removeprefix("pkg:pypi/") for purl in config["deny-packages"]}
    assert denied >= GPL_PACKAGES


def test_dependency_review_excuses_exactly_the_locked_cuda_wheels(repo_root: Path) -> None:
    import fnmatch
    import tomllib

    from scripts.audit_python_deps import REVIEWED_PROPRIETARY

    config = load_yaml(repo_root / ".github" / "dependency-review-config.yml")
    allowed = {p.removeprefix("pkg:pypi/") for p in config["allow-dependencies-licenses"]}
    lock = tomllib.loads((repo_root / "uv.lock").read_text(encoding="utf-8"))
    cuda = {
        p["name"]
        for p in lock["package"]
        if any(fnmatch.fnmatchcase(p["name"], pattern) for pattern in REVIEWED_PROPRIETARY)
    }
    assert cuda, "no CUDA wheels in uv.lock"
    excused = {name for name in allowed if name.startswith("nvidia-")}
    assert excused == cuda, "dependency review and the Python gate disagree on CUDA wheels"


def test_security_gate_is_part_of_ci_ok(repo_root: Path) -> None:
    jobs = load_yaml(repo_root / ".github" / "workflows" / "ci.yml")["jobs"]
    callers = [
        name for name, job in jobs.items() if job.get("uses") == "$/.github/workflows/security.yml"
    ]
    assert callers, "ci.yml does not call the security workflow"
    assert set(callers) <= set(jobs["ci-ok"]["needs"])


def test_security_workflow_runs_the_python_dependency_gate(repo_root: Path) -> None:
    jobs = load_yaml(repo_root / ".github" / "workflows" / "security.yml")["jobs"]
    assert {"gitleaks", "zizmor", "python-deps", "dependency-review"} <= jobs.keys()
    commands = " ".join(str(step.get("run", "")) for step in jobs["python-deps"]["steps"])
    assert "scripts/audit_python_deps.py" in commands
