"""Shared builders for benchmark tests."""

from datetime import UTC, datetime
from typing import Any

import pytest

from scripts.bench.common.privacy import PrivacyContext
from scripts.bench.common.schema import build_result

CREATED_AT = datetime(2026, 11, 4, 15, 30, 0, tzinfo=UTC)


def sample_result(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "component": "llm",
        "variant": "qwen3-4b",
        "machine": {
            "os": "Windows",
            "os_version": "11",
            "cpu": "Example CPU 8-Core",
            "ram_total_mb": 16_384,
            "gpu": "NVIDIA GeForce RTX 4060",
            "vram_total_mb": 8_188,
            "driver": "616.56",
            "cuda_runtime": None,
            "docker_engine": None,
        },
        "software": {"python": "3.12.9", "ollama": "0.40.10"},
        "settings": {"num_ctx": 4096, "endpoint": "http://127.0.0.1:11434"},
        "measurements": {"ttft_s": {"cold": 2.1, "warm": [0.3], "summary": {"count": 1}}},
        "checks": [{"name": "llm_warm_ttft", "status": "pass", "value": 0.3, "limit": 0.4}],
        "created_at": CREATED_AT,
    }
    fields.update(overrides)
    return build_result(**fields)


@pytest.fixture
def privacy() -> PrivacyContext:
    """A fixed identity, so tests do not depend on who runs them."""
    return PrivacyContext(
        user_names=("ada",), home_dirs=("C:\\Users\\ada",), host_names=("ada-laptop",)
    )
