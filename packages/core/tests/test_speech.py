import pytest
from hypothesis import given
from hypothesis import strategies as st

from parlora_core.speech import FIRST_MAX_WORDS, FIRST_MIN_WORDS, ClauseSplitter, spoken_text

pytestmark = pytest.mark.unit


def split(pieces: list[str]) -> list[str]:
    splitter = ClauseSplitter()
    chunks = [chunk for piece in pieces for chunk in splitter.feed(piece)]
    last = splitter.flush()
    return [*chunks, last] if last is not None else chunks


def test_first_clause_goes_early() -> None:
    chunks = split(list("Sure, I love that question. What do you think?"))

    assert [c.strip() for c in chunks] == ["Sure, I love that question.", "What do you think?"]


def test_first_chunk_is_sent_before_the_reply_ends() -> None:
    splitter = ClauseSplitter()

    assert splitter.feed("Sure, I love that question. ") == ["Sure, I love that question."]
    assert splitter.feed("What do") == []
    assert splitter.flush() == " What do"


def test_comma_starts_the_first_chunk_once_three_words_are_buffered() -> None:
    chunks = split(["Well, that is lovely, but tell me more."])

    assert [c.strip() for c in chunks] == ["Well, that is lovely,", "but tell me more."]


def test_long_first_sentence_is_cut_at_a_word_boundary() -> None:
    words = [f"word{i}" for i in range(20)]

    chunks = split([" ".join(words) + "."])

    assert len(chunks[0].split()) == FIRST_MAX_WORDS
    assert chunks[0] == " ".join(words[:FIRST_MAX_WORDS])
    assert "".join(chunks) == " ".join(words) + "."


def test_later_chunks_end_at_sentences_not_commas() -> None:
    chunks = split(["Yes, good idea. Then, after lunch, we walk. Fine!"])

    assert [c.strip() for c in chunks] == [
        "Yes, good idea.",
        "Then, after lunch, we walk.",
        "Fine!",
    ]


def test_decimal_point_is_not_a_boundary() -> None:
    assert [c.strip() for c in split(["It costs 3.50 euros, more or less."])] == [
        "It costs 3.50 euros,",
        "more or less.",
    ]


def test_nothing_to_flush() -> None:
    splitter = ClauseSplitter()

    assert splitter.feed("Hello there friend. ") == ["Hello there friend."]
    assert splitter.flush() is None


def test_whitespace_only_remainder_is_dropped() -> None:
    splitter = ClauseSplitter()

    assert splitter.feed("Hello there friend.") == []
    assert splitter.feed("   ") == ["Hello there friend."]
    assert splitter.flush() is None


WORD = st.text(alphabet="abcdefghij", min_size=1, max_size=6)
TOKEN = st.one_of(
    WORD, WORD.map(lambda w: w + ","), WORD.map(lambda w: w + "."), WORD.map(lambda w: w + "?")
)


@st.composite
def streamed(draw: st.DrawFn) -> tuple[str, list[str]]:
    text = " ".join(draw(st.lists(TOKEN, min_size=1, max_size=40)))
    cuts = sorted(
        draw(st.sets(st.integers(min_value=1, max_value=max(1, len(text) - 1)), max_size=10))
    )
    pieces = [text[a:b] for a, b in zip([0, *cuts], [*cuts, len(text)], strict=True) if a < b]
    return text, pieces


@given(streamed())
def test_chunks_rebuild_the_text_and_never_cut_inside_a_word(case: tuple[str, list[str]]) -> None:
    text, pieces = case

    chunks = split(pieces)

    assert "".join(chunks) == text
    position = 0
    for chunk in chunks[:-1]:
        position += len(chunk)
        assert text[position].isspace(), f"chunk {chunk!r} ends inside a word"


@given(streamed())
def test_first_chunk_word_count_when_more_follow(case: tuple[str, list[str]]) -> None:
    _, pieces = case

    chunks = split(pieces)

    if len(chunks) > 1:
        assert FIRST_MIN_WORDS <= len(chunks[0].split()) <= FIRST_MAX_WORDS


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("**Great!** Let's talk 😀", "Great! Let's talk"),
        ("Try _this_ and `that`.", "Try this and that."),
        ("# Title\n- first\n- second", "Title first second"),
        ("1. one\n2. two", "one two"),
        ("See [the guide](https://example.test) now 👍🏽!", "See the guide now!"),
        ("Plain text stays.", "Plain text stays."),
        ("Café in Zürich ❤️", "Café in Zürich"),
    ],
)
def test_spoken_text(text: str, spoken: str) -> None:
    assert spoken_text(text) == spoken
