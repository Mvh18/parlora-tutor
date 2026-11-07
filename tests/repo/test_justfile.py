"""The task runner exposes every required recipe with a description."""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REQUIRED_RECIPES = [
    "setup",
    "dev",
    "up",
    "down",
    "fmt",
    "lint",
    "typecheck",
    "test",
    "test-core",
    "test-server",
    "test-web",
    "test-repo",
    "test-bench",
    "bench-setup",
    "bench-llm",
    "bench-fixtures",
    "bench-stt",
    "bench-stt-container",
    "bench-tts",
    "bench-tts-container",
    "bench-memory",
    "e2e",
    "openapi",
    "gen-client",
    "security",
    "check",
]

# A doc comment line followed by the recipe header, e.g. "# Run tests\ntest: lint" or
# "# Start\nup mode=\"gpu\":"; assignments (`name := value`) do not match.
RECIPE = re.compile(r"^# (?P<doc>.+)\n(?:\[.*\]\n)*(?P<name>[a-z][a-z0-9-]*)\b[^:\n]*:(?!=)", re.M)


def documented_recipes(repo_root: Path) -> dict[str, str]:
    text = (repo_root / "justfile").read_text(encoding="utf-8")
    return {m["name"]: m["doc"] for m in RECIPE.finditer(text)}


@pytest.mark.parametrize("name", REQUIRED_RECIPES)
def test_recipe_exists_with_description(repo_root: Path, name: str) -> None:
    recipes = documented_recipes(repo_root)
    assert name in recipes, f"justfile lacks a documented `{name}` recipe"
    assert recipes[name].strip()


def test_windows_shell_is_powershell(repo_root: Path) -> None:
    text = (repo_root / "justfile").read_text(encoding="utf-8")
    assert re.search(r"^set windows-shell := \[\"powershell(\.exe)?\"", text, re.M)


@pytest.mark.parametrize("name", ["up", "down", "dev"])
def test_stack_recipes_drive_docker_compose(repo_root: Path, name: str) -> None:
    text = (repo_root / "justfile").read_text(encoding="utf-8")
    body = re.search(rf"^{name}\b[^\n]*:\n((?:[ \t]+\S[^\n]*\n)+)", text, re.M)
    assert body, f"recipe `{name}` has no body"
    assert "docker compose" in body.group(1)
