"""The voice pipeline of one call, assembled from settings (design D1, D5).

```
transport in → STT service → user aggregator (Silero VAD, Smart Turn v3)
  → LLM service → TTS service (clause aggregation) → transport out → assistant aggregator
```

Silero VAD marks speech with a short `stop_secs`; Smart Turn v3, local on the CPU,
decides whether the learner has finished, and its `stop_secs` is the fallback: after
that much silence the turn ends even when the model is unsure. Learner speech while
the tutor talks interrupts it (Pipecat's default turn start strategies). The
conversation context starts with the system prompt of the call's language.
"""

from dataclasses import dataclass

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from parlora_core.languages import LanguageCode
from parlora_core.ports import LanguageModel, SpeechToText, TextToSpeech
from parlora_server.config import Settings, VadSettings
from parlora_server.voice.services import (
    LanguageModelService,
    SpeechToTextService,
    TextToSpeechService,
)


def vad_params(settings: VadSettings) -> VADParams:
    return VADParams(
        confidence=settings.confidence,
        start_secs=settings.start_secs,
        stop_secs=settings.stop_secs,
        min_volume=settings.min_volume,
    )


def silero_vad(settings: VadSettings) -> VADAnalyzer:
    from pipecat.audio.vad.silero import SileroVADAnalyzer

    return SileroVADAnalyzer(params=vad_params(settings))


def smart_turn(settings: VadSettings) -> BaseTurnAnalyzer:
    from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

    return LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=settings.turn_fallback_secs))


@dataclass(frozen=True)
class VoiceModels:
    """The adapters a call talks to, loaded once at start-up and shared by calls."""

    stt: SpeechToText
    llm: LanguageModel
    tts: TextToSpeech


@dataclass
class CallPipeline:
    """A call's pipeline and the pieces tests and the call registry look at."""

    pipeline: Pipeline
    context: LLMContext
    tts: TextToSpeechService


def build_pipeline(
    *,
    transport_in: FrameProcessor,
    transport_out: FrameProcessor,
    models: VoiceModels,
    settings: Settings,
    language: LanguageCode,
    system_prompt: str,
    vad: VADAnalyzer | None,
    turn_analyzer: BaseTurnAnalyzer,
) -> CallPipeline:
    """Assemble the pipeline of one call; `vad` and `turn_analyzer` come from settings."""
    context = LLMContext(messages=[{"role": "system", "content": system_prompt}])
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad,
            user_turn_strategies=UserTurnStrategies(
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=turn_analyzer)]
            ),
        ),
    )
    tts = TextToSpeechService(models.tts, language=language, voice=settings.tts.voice)
    pipeline = Pipeline(
        [
            transport_in,
            SpeechToTextService(models.stt, language=language),
            aggregators.user(),
            LanguageModelService(models.llm),
            tts,
            transport_out,
            aggregators.assistant(),
        ]
    )
    return CallPipeline(pipeline, context, tts)
