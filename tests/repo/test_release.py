"""Release workflow: issue forms, labels, PR template, PR titles, versions."""

import json
import subprocess
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

FORMS = ["feature", "bug", "chore", "spike"]
COMMIT_TYPES = {
    "feat", "fix", "perf", "refactor", "docs", "test",
    "build", "ci", "style", "chore", "revert",
}  # fmt: skip
CONTACT = "team.parlora@gmail.com"


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def label_names(repo_root: Path) -> set[str]:
    labels = load_yaml(repo_root / ".github" / "labels.yml")
    return {label["name"] for label in labels}


def tracked(repo_root: Path, name: str) -> list[PurePosixPath]:
    out = subprocess.run(
        ["git", "ls-files", "-z", f"*{name}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [PurePosixPath(p) for p in out.split("\0") if p and PurePosixPath(p).name == name]


# ── Issue forms and labels ──────────────────────────────────────────────────


@pytest.mark.parametrize("form", FORMS)
def test_issue_form_is_valid(repo_root: Path, form: str) -> None:
    data = load_yaml(repo_root / ".github" / "ISSUE_TEMPLATE" / f"{form}.yml")
    assert data["name"]
    assert data["description"]
    assert isinstance(data["body"], list)
    assert data["body"]
    assert set(data.get("labels", [])) <= label_names(repo_root), "unknown label in form"


def test_blank_issues_are_disabled_and_security_is_linked(repo_root: Path) -> None:
    config = load_yaml(repo_root / ".github" / "ISSUE_TEMPLATE" / "config.yml")
    assert config["blank_issues_enabled"] is False
    urls = [link["url"] for link in config["contact_links"]]
    assert any("security" in url for url in urls)


def test_labels_cover_the_taxonomy(repo_root: Path) -> None:
    labels = load_yaml(repo_root / ".github" / "labels.yml")
    names = [label["name"] for label in labels]
    assert len(names) == len(set(names)), "duplicate labels"
    assert all(label.get("color") and label.get("description") for label in labels)
    required = {"type/epic", "type/feature", "type/bug", "type/chore", "type/spike", "type/docs"}
    required |= {"area/integration", "status/needs-triage", "good first issue", "help wanted"}
    required |= {f"phase/{n}" for n in range(9)}
    assert required <= set(names)


# ── Pull requests ───────────────────────────────────────────────────────────


def test_pull_request_template_sections(repo_root: Path) -> None:
    text = (repo_root / ".github" / "pull_request_template.md").read_text(encoding="utf-8")
    for heading in [
        "## Summary",
        "## How to test",
        "## Breaking changes",
        "## Type of change",
        "## Deploy considerations",
        "## Checklist",
    ]:
        assert heading in text, f"missing {heading}"
    assert "Closes #" in text
    checklist = text.split("## Checklist", 1)[1]
    for item in ["Tests", "CLA", "No secrets", "personal data"]:
        assert item in checklist, f"checklist lacks {item!r}"


def test_pr_title_check_allows_exactly_the_commit_types(repo_root: Path) -> None:
    workflow = load_yaml(repo_root / ".github" / "workflows" / "pr-title.yml")
    step = next(
        s
        for s in workflow["jobs"]["title"]["steps"]
        if "action-semantic-pull-request" in s.get("uses", "")
    )
    assert set(step["with"]["types"].split()) == COMMIT_TYPES


def test_dependabot_titles_follow_the_pr_title_check(repo_root: Path) -> None:
    # An explicit lowercase prefix makes Dependabot write "build(deps): bump x";
    # without one it copies the history's style and writes "Bump", which the
    # lowercase subject rule of the PR title check rejects.
    config = load_yaml(repo_root / ".github" / "dependabot.yml")
    expected = {"prefix": "build", "prefix-development": "build", "include": "scope"}
    offenders = [
        u["package-ecosystem"] for u in config["updates"] if u.get("commit-message") != expected
    ]
    assert not offenders, f"updates without the build(deps) commit message: {offenders}"


# ── Versioning ──────────────────────────────────────────────────────────────


def test_every_manifest_carries_the_released_version(repo_root: Path) -> None:
    released = json.loads((repo_root / ".release-please-manifest.json").read_text("utf-8"))["."]
    versions = {}
    for path in tracked(repo_root, "pyproject.toml"):
        project = tomllib.loads((repo_root / path).read_text(encoding="utf-8"))["project"]
        versions[str(path)] = project["version"]
    for path in tracked(repo_root, "package.json"):
        versions[str(path)] = json.loads((repo_root / path).read_text("utf-8"))["version"]
    lock = tomllib.loads((repo_root / "uv.lock").read_text(encoding="utf-8"))
    for package in lock["package"]:
        if "virtual" in package["source"] or "editable" in package["source"]:
            versions[f"uv.lock:{package['name']}"] = package["version"]
    contract = json.loads((repo_root / "contracts" / "openapi.json").read_text("utf-8"))
    versions["contracts/openapi.json"] = contract["info"]["version"]
    stale = {name: v for name, v in versions.items() if v != released}
    assert not stale, f"versions differ from {released}: {stale}"


def test_release_please_bumps_every_manifest(repo_root: Path) -> None:
    config = json.loads((repo_root / "release-please-config.json").read_text("utf-8"))
    package = config["packages"]["."]
    assert config["bump-minor-pre-major"] is True
    assert package["release-type"] == "node"  # bumps the root package.json
    bumped = {entry["path"] for entry in package["extra-files"]}
    manifests = {str(p) for p in tracked(repo_root, "pyproject.toml")}
    manifests |= {str(p) for p in tracked(repo_root, "package.json")} - {"package.json"}
    assert manifests | {"uv.lock"} <= bumped


def test_release_please_bumps_the_contract_version(repo_root: Path) -> None:
    # The app reports the package version in the contract's `info.version`, so
    # a release that leaves contracts/openapi.json behind fails the drift check.
    config = json.loads((repo_root / "release-please-config.json").read_text("utf-8"))
    entry = {"type": "json", "path": "contracts/openapi.json", "jsonpath": "$.info.version"}
    assert entry in config["packages"]["."]["extra-files"]


def test_release_please_bumps_every_local_package_in_the_lockfile(repo_root: Path) -> None:
    # release-please parses TOML into tagged values ({start, end, value}) so it
    # can rewrite them in place; a filter must compare `@.name.value`, because
    # `@.name == '...'` never matches and the file is silently skipped.
    config = json.loads((repo_root / "release-please-config.json").read_text("utf-8"))
    lock_paths = {
        entry["jsonpath"]
        for entry in config["packages"]["."]["extra-files"]
        if entry["path"] == "uv.lock"
    }
    lock = tomllib.loads((repo_root / "uv.lock").read_text(encoding="utf-8"))
    local = [
        package["name"]
        for package in lock["package"]
        if "virtual" in package["source"] or "editable" in package["source"]
    ]
    expected = {f"$.package[?(@.name.value == '{name}')].version" for name in local}
    assert lock_paths == expected


# ── Community ───────────────────────────────────────────────────────────────


def test_code_of_conduct_names_the_project_contact(repo_root: Path) -> None:
    assert CONTACT in (repo_root / "CODE_OF_CONDUCT.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["CONTRIBUTING.md", "SUPPORT.md", "CHANGELOG.md"])
def test_community_file_exists(repo_root: Path, name: str) -> None:
    assert (repo_root / name).is_file()


# ── Contributor licence agreement ───────────────────────────────────────────

CLA_PHRASE = "I have read and accept the Parlora CLA"


def test_cla_check_never_checks_out_pull_request_code(repo_root: Path) -> None:
    workflow = load_yaml(repo_root / ".github" / "workflows" / "cla.yml")
    triggers = workflow[True]  # YAML 1.1 reads the `on` key as a boolean
    assert "pull_request_target" in triggers
    steps = [step for job in workflow["jobs"].values() for step in job["steps"]]
    assert not any("actions/checkout" in step.get("uses", "") for step in steps)


def test_cla_phrase_matches_the_agreement(repo_root: Path) -> None:
    workflow = (repo_root / ".github" / "workflows" / "cla.yml").read_text(encoding="utf-8")
    agreement = (repo_root / "CLA.md").read_text(encoding="utf-8")
    assert CLA_PHRASE in workflow
    assert CLA_PHRASE in agreement


def test_cla_exempts_the_maintainer(repo_root: Path) -> None:
    workflow = (repo_root / ".github" / "workflows" / "cla.yml").read_text(encoding="utf-8")
    assert "'Mvh18'" in workflow
