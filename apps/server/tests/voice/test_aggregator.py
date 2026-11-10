"""The clause aggregator that feeds text-to-speech."""

import asyncio

import pytest

from parlora_server.voice.aggregator import ClauseTextAggregator

pytestmark = pytest.mark.unit


def run(aggregator: ClauseTextAggregator, tokens: list[str]) -> list[str]:
    async def collect() -> list[str]:
        pieces = []
        for token in tokens:
            pieces += [a.text async for a in aggregator.aggregate(token)]
        rest = await aggregator.flush()
        return pieces + ([rest.text] if rest else [])

    return asyncio.run(collect())


def tokens(text: str) -> list[str]:
    words = text.split(" ")
    return [words[0], *(" " + w for w in words[1:])]


def test_first_clause_goes_before_the_rest_of_the_reply() -> None:
    reply = "Sure, I love that question. What do you think?"

    assert run(ClauseTextAggregator(), tokens(reply)) == [
        "Sure, I love that question.",
        "What do you think?",
    ]


def test_a_long_first_sentence_is_cut_at_twelve_words() -> None:
    reply = " ".join(f"word{i}" for i in range(20)) + "."

    first, *_ = run(ClauseTextAggregator(), tokens(reply))

    assert len(first.split()) <= 12
    assert first == " ".join(f"word{i}" for i in range(12))


def test_each_reply_starts_again_with_the_clause_rule() -> None:
    aggregator = ClauseTextAggregator()

    run(aggregator, tokens("Well, that is a good one. Tell me more."))
    second = run(aggregator, tokens("Oh no, that sounds hard. Why?"))

    assert second == ["Oh no, that sounds hard.", "Why?"]


def test_text_holds_what_is_not_spoken_yet() -> None:
    aggregator = ClauseTextAggregator()

    async def feed() -> str:
        async for _ in aggregator.aggregate("Hello there"):
            pass
        return aggregator.text.text

    assert asyncio.run(feed()) == "Hello there"


def test_an_interruption_drops_the_buffered_text() -> None:
    aggregator = ClauseTextAggregator()

    async def interrupted() -> object:
        async for _ in aggregator.aggregate("Half a sente"):
            pass
        await aggregator.handle_interruption()
        return await aggregator.flush()

    assert asyncio.run(interrupted()) is None
