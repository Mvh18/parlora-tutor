"""The prompt loader, and the real VAD and turn analysers built from settings (CPU)."""

from pathlib import Path

import pytest

from parlora_core.errors import UnsupportedLanguage
from parlora_core.languages import LanguageCode
from parlora_server.config import VadSettings
from parlora_server.voice import pipeline as voice
from parlora_server.voice import prompts

pytestmark = pytest.mark.unit


def test_the_english_conversation_prompt_is_found_from_a_checkout() -> None:
    prompt = prompts.load_prompt(LanguageCode("en"))

    assert prompt.startswith("You are a friendly English conversation partner")
    assert prompt == prompt.strip()


def test_prompts_load_by_language_from_a_root(tmp_path: Path) -> None:
    (tmp_path / "es").mkdir()
    (tmp_path / "es" / "conversation.md").write_text("  Eres un compañero.\n", encoding="utf-8")

    assert prompts.load_prompt(LanguageCode("es"), root=tmp_path) == "Eres un compañero."
    with pytest.raises(UnsupportedLanguage, match="fr"):
        prompts.load_prompt(LanguageCode("fr"), root=tmp_path)


def test_prompt_names_cannot_leave_the_folder(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prompt name"):
        prompts.load_prompt(LanguageCode("en"), name="../secrets", root=tmp_path)


def test_outside_a_checkout_prompts_come_from_the_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(prompts, "__file__", str(tmp_path / "site-packages" / "prompts.py"))
    monkeypatch.chdir(tmp_path)
    prompts.prompts_dir.cache_clear()
    try:
        assert prompts.prompts_dir() == tmp_path / "prompts"
    finally:
        prompts.prompts_dir.cache_clear()


def test_silero_vad_takes_the_settings() -> None:
    vad = voice.silero_vad(VadSettings(stop_secs=0.3, confidence=0.6))

    assert vad.params.stop_secs == 0.3
    assert vad.params.confidence == 0.6


def test_smart_turn_falls_back_after_the_configured_silence() -> None:
    analyser = voice.smart_turn(VadSettings(turn_fallback_secs=2.5))

    assert analyser.params.stop_secs == 2.5  # type: ignore[attr-defined]
