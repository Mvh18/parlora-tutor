"""GitHub Actions workflows are hardened: explicit permissions, pinned actions."""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

SHA_PINNED = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")


def workflow_files(repo_root: Path) -> list[Path]:
    return sorted((repo_root / ".github" / "workflows").glob("*.y*ml"))


def load(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path.name} is not a mapping"
    return data


def steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for job in workflow.get("jobs", {}).values() for step in job.get("steps", [])]


def test_workflows_exist(repo_root: Path) -> None:
    assert workflow_files(repo_root), "no workflows found"


def test_every_workflow_declares_top_level_permissions(repo_root: Path) -> None:
    missing = [p.name for p in workflow_files(repo_root) if "permissions" not in load(p)]
    assert not missing, f"workflows without top-level permissions: {missing}"


def test_third_party_actions_are_pinned_to_a_commit_sha(repo_root: Path) -> None:
    unpinned = []
    for path in workflow_files(repo_root):
        workflow = load(path)
        uses = [s["uses"] for s in steps(workflow) if "uses" in s]
        uses += [j["uses"] for j in workflow.get("jobs", {}).values() if "uses" in j]
        for ref in uses:
            # Same-repository references ($/ is GitHub's self-repository syntax).
            if ref.startswith(("$/", "./", "docker://")):
                continue
            if not SHA_PINNED.match(ref):
                unpinned.append(f"{path.name}: {ref}")
    assert not unpinned, f"actions not pinned to a full SHA: {unpinned}"


def test_checkout_does_not_persist_credentials(repo_root: Path) -> None:
    offenders = []
    for path in workflow_files(repo_root):
        for step in steps(load(path)):
            is_checkout = str(step.get("uses", "")).startswith("actions/checkout@")
            if is_checkout and step.get("with", {}).get("persist-credentials") is not False:
                offenders.append(path.name)
    assert not offenders, f"checkout without persist-credentials: false in {offenders}"


def test_ci_ok_aggregates_every_ci_job(repo_root: Path) -> None:
    jobs = load(repo_root / ".github" / "workflows" / "ci.yml")["jobs"]
    others = set(jobs) - {"ci-ok"}
    assert set(jobs["ci-ok"]["needs"]) == others, "a CI job is missing from ci-ok"
    assert jobs["ci-ok"]["if"] == "always()"


@pytest.mark.parametrize(
    ("job", "expected"),
    [
        ("server", "just test-server"),
        ("contract", "export_openapi.py --check"),
        ("web", "just test-web"),
        ("web", "pnpm --filter @parlora/web build"),
        ("e2e", "just e2e"),
    ],
)
def test_ci_jobs_run_their_recipes(repo_root: Path, job: str, expected: str) -> None:
    jobs = load(repo_root / ".github" / "workflows" / "ci.yml")["jobs"]
    commands = " ".join(step.get("run", "") for step in jobs[job]["steps"])
    assert expected in commands


def test_contract_job_fails_on_a_stale_web_client(repo_root: Path) -> None:
    jobs = load(repo_root / ".github" / "workflows" / "ci.yml")["jobs"]
    commands = " ".join(step.get("run", "") for step in jobs["contract"]["steps"])
    assert "just gen-client" in commands
    assert "git diff --exit-code" in commands


def test_codeql_analyses_typescript(repo_root: Path) -> None:
    workflow = load(repo_root / ".github" / "workflows" / "codeql.yml")
    languages = workflow["jobs"]["analyze"]["strategy"]["matrix"]["language"]
    assert "javascript-typescript" in languages


def test_docker_job_checks_the_runtime_image(repo_root: Path) -> None:
    jobs = load(repo_root / ".github" / "workflows" / "ci.yml")["jobs"]
    commands = " ".join(step.get("run", "") for step in jobs["docker"]["steps"])
    assert "compose.nogpu.yaml" in commands
    assert "config" in commands
    assert ".Config.User" in commands
    assert "/api/v1/health/live" in commands
