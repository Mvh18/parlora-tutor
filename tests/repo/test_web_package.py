"""The web app stays a minimal dev client: Vite + React, English, port 3000 (design D11)."""

import json
import re
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

WEB = Path("apps/web")
# Hub owns the product UI, so the dev client needs no framework, i18n,
# component or global state library.
FORBIDDEN_DEPENDENCIES = re.compile(
    r"^(next|next-intl|i18next|react-i18next|react-intl|@formatjs/.+|@tanstack/.+|zustand"
    r"|redux|@reduxjs/.+|jotai|@radix-ui/.+|@mui/.+|@chakra-ui/.+|tailwindcss|@tailwindcss/.+)$"
)


def manifest(repo_root: Path) -> dict[str, Any]:
    data = json.loads((repo_root / WEB / "package.json").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_web_package_is_private_and_named(repo_root: Path) -> None:
    package = manifest(repo_root)
    assert package["name"] == "@parlora/web"
    assert package["private"] is True


def test_web_uses_vite_and_react(repo_root: Path) -> None:
    package = manifest(repo_root)
    deps = package.get("dependencies", {}) | package.get("devDependencies", {})
    assert {"vite", "react", "react-dom", "typescript"} <= set(deps)


def test_web_has_no_framework_i18n_component_or_state_library(repo_root: Path) -> None:
    package = manifest(repo_root)
    deps = set(package.get("dependencies", {})) | set(package.get("devDependencies", {}))
    forbidden = sorted(d for d in deps if FORBIDDEN_DEPENDENCIES.match(d))
    assert not forbidden, f"apps/web must stay minimal; remove {forbidden}"


def test_dev_server_and_preview_use_port_3000_strictly(repo_root: Path) -> None:
    config = (repo_root / WEB / "vite.config.ts").read_text(encoding="utf-8")
    for section in ("server", "preview"):
        block = re.search(rf"{section}:\s*\{{([^}}]*)\}}", config)
        assert block, f"vite.config.ts has no `{section}` block"
        assert re.search(r"port:\s*3000\b", block.group(1)), f"{section} port is not 3000"
        assert re.search(r"strictPort:\s*true\b", block.group(1)), f"{section} port is not strict"


def test_document_language_is_english(repo_root: Path) -> None:
    html = (repo_root / WEB / "index.html").read_text(encoding="utf-8")
    assert re.search(r'<html[^>]*\blang="en"', html)


def test_typescript_is_strict(repo_root: Path) -> None:
    config = (repo_root / WEB / "tsconfig.app.json").read_text(encoding="utf-8")
    assert re.search(r'"strict":\s*true', config)


def test_generated_api_schema_is_committed(repo_root: Path) -> None:
    assert (repo_root / WEB / "src/lib/api/schema.d.ts").is_file()
