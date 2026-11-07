import asyncio
import json
import subprocess
import sys

import pytest

from parlora_core import ports

pytestmark = pytest.mark.unit

PORTS = [ports.SpeechToText, ports.LanguageModel, ports.TextToSpeech, ports.SessionRepository]

# Lists the top-level modules that importing the ports adds to a fresh interpreter.
PROBE = """
import json, sys
before = set(sys.modules)
import parlora_core.ports
print(json.dumps(sorted({name.split(".")[0] for name in set(sys.modules) - before})))
"""


@pytest.mark.parametrize("port", PORTS)
def test_each_port_is_a_protocol(port: type) -> None:
    assert getattr(port, "_is_protocol", False), f"{port.__name__} is not a typing.Protocol"


def test_importing_the_ports_loads_no_third_party_module() -> None:
    result = subprocess.run(  # noqa: S603 — fixed arguments, current interpreter
        [sys.executable, "-c", PROBE], capture_output=True, text=True, check=True
    )
    loaded = set(json.loads(result.stdout))
    third_party = loaded - set(sys.stdlib_module_names) - {"parlora_core"}
    assert not third_party, f"ports import third-party modules: {sorted(third_party)}"


def test_message_is_a_value() -> None:
    assert ports.Message("user", "hello") == ports.Message("user", "hello")


def test_ports_are_implementable_by_streaming_fakes() -> None:
    asyncio.run(_streaming_fakes())


async def _streaming_fakes() -> None:
    from collections.abc import AsyncIterator, Sequence

    from parlora_core.audio import AudioChunk, Transcript
    from parlora_core.languages import LanguageCode

    class FakeStt:
        async def transcribe(self, audio: AudioChunk, *, language: LanguageCode) -> Transcript:
            return Transcript("hi", language, audio.duration_s, 0.0)

    class FakeLlm:
        async def stream_reply(self, messages: Sequence[ports.Message]) -> AsyncIterator[str]:
            for piece in ("Hel", "lo"):
                yield piece

    class FakeTts:
        async def stream_speech(
            self, text: str, *, language: LanguageCode, voice: str
        ) -> AsyncIterator[AudioChunk]:
            yield AudioChunk(b"\0\0" * len(text), 16_000)

    stt: ports.SpeechToText = FakeStt()
    llm: ports.LanguageModel = FakeLlm()
    tts: ports.TextToSpeech = FakeTts()
    en = LanguageCode("en")

    transcript = await stt.transcribe(AudioChunk(b"\0\0" * 1600, 16_000), language=en)
    reply = "".join([piece async for piece in llm.stream_reply([ports.Message("user", "hi")])])
    audio = [chunk async for chunk in tts.stream_speech(reply, language=en, voice="af_heart")]

    assert transcript.audio_s == 0.1
    assert reply == "Hello"
    assert audio[0].duration_s == pytest.approx(5 / 16_000)
