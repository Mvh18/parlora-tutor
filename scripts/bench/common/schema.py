"""The versioned schema of one benchmark result.

A result is a JSON object with `schema_version`, `component`, `variant`,
`created_at` (UTC, `YYYY-MM-DDTHH:MM:SSZ`) and the blocks:

- `machine`: the allow-list in `scripts.bench.common.machine`.
- `software`: Python and package versions, Ollama version, model digests, weight hashes.
- `settings`: every parameter that affects the numbers.
- `measurements`: raw per-run values and their summaries.
- `checks`: budget and VRAM checks, each with a `name` and a `status`
  (`pass`, `warn` or `fail`).

Bump `SCHEMA_VERSION` when a field changes meaning or a required field is added.
"""

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from scripts.bench.common.machine import machine_problems

SCHEMA_VERSION = 1
COMPONENTS = ("llm", "stt", "tts", "memory")
STATUSES = ("pass", "warn", "fail")
REQUIRED_BLOCKS = ("machine", "software", "settings", "measurements", "checks")
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_HEADER = ("schema_version", "component", "variant", "created_at")
_VARIANT = re.compile(r"^[a-z0-9][a-z0-9.-]*$")


class SchemaError(ValueError):
    """A result does not follow the schema."""


def slug(text: str) -> str:
    """A variant name for a file name: lowercase, `-` for anything but letters, digits, `.`."""
    value = re.sub(r"[^a-z0-9.]+", "-", text.casefold()).strip("-.")
    if not value:
        raise ValueError(f"cannot make a slug from {text!r}")
    return value


def build_result(
    *,
    component: str,
    variant: str,
    machine: dict[str, Any],
    software: dict[str, Any],
    settings: dict[str, Any],
    measurements: dict[str, Any],
    checks: list[dict[str, Any]],
    created_at: datetime | None = None,
) -> dict[str, Any]:
    moment = (created_at or datetime.now(UTC)).astimezone(UTC)
    return {
        "schema_version": SCHEMA_VERSION,
        "component": component,
        "variant": variant,
        "created_at": moment.strftime(TIMESTAMP_FORMAT),
        "machine": machine,
        "software": software,
        "settings": settings,
        "measurements": measurements,
        "checks": checks,
    }


def _check_problems(checks: object) -> list[str]:
    if not isinstance(checks, list):
        return ["checks must be a list"]
    problems = []
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            problems.append(f"checks[{index}] must be an object")
            continue
        if not isinstance(check.get("name"), str) or not check["name"]:
            problems.append(f"checks[{index}].name must be a non-empty string")
        if check.get("status") not in STATUSES:
            problems.append(f"checks[{index}].status must be one of {', '.join(STATUSES)}")
    return problems


def _header_problems(result: Mapping[str, Any]) -> list[str]:
    problems = []
    version = result.get("schema_version")
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}")
    if result.get("component") not in COMPONENTS:
        problems.append(f"component must be one of {', '.join(COMPONENTS)}")
    variant = result.get("variant")
    if not isinstance(variant, str) or not _VARIANT.match(variant):
        problems.append("variant must be a lowercase slug")
    try:
        datetime.strptime(str(result.get("created_at")), TIMESTAMP_FORMAT)
    except ValueError:
        problems.append(f"created_at must be UTC in the form {TIMESTAMP_FORMAT}")
    return problems


def validate_result(result: Mapping[str, Any]) -> None:
    """Raise `SchemaError` listing every problem of `result`."""
    problems = [f"unknown key {key!r}" for key in result if key not in (*_HEADER, *REQUIRED_BLOCKS)]
    problems += [
        f"missing key {key!r}" for key in (*_HEADER, *REQUIRED_BLOCKS) if key not in result
    ]
    problems += _header_problems(result)
    if "machine" in result:
        problems += machine_problems(result["machine"])
    for block in ("software", "settings", "measurements"):
        if block in result and not isinstance(result[block], dict):
            problems.append(f"{block} must be an object")
    if "checks" in result:
        problems += _check_problems(result["checks"])
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as error:
        problems.append(f"result is not plain JSON: {error}")
    if problems:
        raise SchemaError("; ".join(problems))
