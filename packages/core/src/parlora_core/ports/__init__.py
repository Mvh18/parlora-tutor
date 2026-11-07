"""Ports the domain needs from the outside world, as `typing.Protocol`s.

Speech and language ports stream: the language model yields text as it writes,
text-to-speech yields audio as it synthesises, so a reply can be spoken before it is
complete. Adapters live in the server, never here.
"""

from parlora_core.ports.llm import LanguageModel, Message
from parlora_core.ports.sessions import SessionRepository
from parlora_core.ports.stt import SpeechToText
from parlora_core.ports.tts import TextToSpeech

__all__ = [
    "LanguageModel",
    "Message",
    "SessionRepository",
    "SpeechToText",
    "TextToSpeech",
]
