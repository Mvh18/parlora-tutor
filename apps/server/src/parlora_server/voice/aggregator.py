"""Clause aggregation for text-to-speech (design D3).

Pipecat's text-to-speech service gathers the streamed reply into pieces before it
synthesises them; its own aggregator cuts only at sentence ends. This one applies
the domain's clause rule (`parlora_core.speech.ClauseSplitter`): the first piece goes
at the first clause boundary once three words are buffered (at most twelve words),
later pieces at sentence ends. Each reply starts with a fresh splitter.
"""

from collections.abc import AsyncIterator

from pipecat.utils.text.base_text_aggregator import Aggregation, TextType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

from parlora_core.speech import ClauseSplitter


class ClauseTextAggregator(SimpleTextAggregator):
    """Yields clause-sized pieces of the reply.

    It subclasses the aggregator Pipecat's TTS service is typed with and overrides
    every method that service calls, so none of the sentence logic runs.
    """

    def __init__(self) -> None:
        # Pipecat leaves this __init__ (**kwargs) unannotated.
        super().__init__(aggregation_type=TextType.SENTENCE)  # type: ignore[no-untyped-call]
        self._splitter = ClauseSplitter()
        self._pending = ""

    @property
    def text(self) -> Aggregation:
        return Aggregation(self._pending.strip(), TextType.SENTENCE)

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        self._pending += text
        for chunk in self._splitter.feed(text):
            self._pending = self._pending[len(chunk) :]
            if chunk.strip():
                yield Aggregation(chunk.strip(), TextType.SENTENCE)

    async def flush(self) -> Aggregation | None:
        rest = self._splitter.flush()
        await self.reset()
        return Aggregation(rest.strip(), TextType.SENTENCE) if rest else None

    async def handle_interruption(self) -> None:
        await self.reset()

    async def reset(self) -> None:
        self._splitter = ClauseSplitter()
        self._pending = ""
