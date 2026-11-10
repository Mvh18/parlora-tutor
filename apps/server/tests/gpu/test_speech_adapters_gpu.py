"""The speech-to-text and LLM adapters against the real models. Needs the GPU: never in CI."""

import asyncio
import wave
from pathlib import Path

import httpx2
import pytest

from parlora_core.audio import AudioChunk
from parlora_core.languages import LanguageCode
from parlora_core.ports import Message
from parlora_server.adapters.llm.ollama import load_ollama
from parlora_server.adapters.stt.whisper import load_whisper
from parlora_server.config import LlmSettings, SttSettings

pytestmark = pytest.mark.gpu

FIXTURES = Path(__file__).resolve().parents[4] / "fixtures" / "synthetic" / "stt"
LOCAL_OLLAMA = "http://127.0.0.1:11434"


def fixture(name: str) -> AudioChunk:
    with wave.open(str(FIXTURES / f"{name}.wav"), "rb") as file:
        return AudioChunk(
            file.readframes(file.getnframes()), file.getframerate(), file.getnchannels()
        )


def test_transcribes_a_synthetic_utterance_on_the_gpu() -> None:
    pytest.importorskip("faster_whisper")
    stt = load_whisper(SttSettings())

    transcript = asyncio.run(stt.transcribe(fixture("a1-greeting"), language=LanguageCode("en")))

    words = transcript.text.lower()
    # Proper names vary in spelling (Anna / Ana); check the common words.
    assert "my name is" in words
    assert "spain" in words
    assert transcript.real_time_factor is not None
    assert transcript.real_time_factor < 1.0


def test_streams_a_real_reply_from_the_host_ollama() -> None:
    try:
        httpx2.get(LOCAL_OLLAMA + "/api/version", timeout=2).raise_for_status()
    except httpx2.HTTPError:
        pytest.skip("Ollama is not running on the host")
    settings = LlmSettings(base_url=LOCAL_OLLAMA)
    llm = load_ollama(settings)
    messages = [
        Message("system", "Reply in one short sentence."),
        Message("user", "Say hello to an English learner."),
    ]

    async def collect() -> list[str]:
        try:
            return [chunk async for chunk in llm.stream_reply(messages)]
        finally:
            await llm.aclose()

    chunks = asyncio.run(collect())

    assert len(chunks) > 1, "the reply arrives in several chunks"
    assert "".join(chunks).strip()
