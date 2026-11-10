"""The conversation prompt states the voice rules and ships with the server image."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_english_conversation_prompt_states_the_voice_rules(repo_root: Path) -> None:
    prompt = (repo_root / "prompts" / "en" / "conversation.md").read_text(encoding="utf-8")
    text = prompt.lower()

    assert "one or two sentences" in text
    assert "first sentence short" in text
    for forbidden in ("markdown", "lists", "emojis"):
        assert forbidden in text, f"the prompt must rule out {forbidden}"
    assert "end with a question" in text


def test_prompts_are_copied_into_the_server_image(repo_root: Path) -> None:
    dockerfile = (repo_root / "docker" / "server.Dockerfile").read_text(encoding="utf-8")
    runtime = dockerfile[dockerfile.index("AS runtime") :]
    assert "COPY prompts /app/prompts" in runtime
    assert "!prompts/" in (repo_root / ".dockerignore").read_text(encoding="utf-8")
