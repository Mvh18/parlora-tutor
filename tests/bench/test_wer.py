"""Word error rate after normalisation."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scripts.bench.common.wer import normalise, number_to_words, score, word_error_rate

pytestmark = pytest.mark.unit


def test_spec_example_scores_zero() -> None:
    assert (
        word_error_rate("I would like two coffees, please.", "i would like 2 coffees please") == 0.0
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello, World!", "hello world"),
        ("I'm sure you're right; we've seen it.", "i am sure you are right we have seen it"),
        ("I can't and won't.", "i can not and will not"),
        ("She doesn't know. It's late, let's go.", "she does not know it is late let us go"),
        ("They'll say he'd left", "they will say he would left"),
        ("Typographic I\N{RIGHT SINGLE QUOTATION MARK}m too", "typographic i am too"),
        ("Anna's book", "annas book"),
        ("A well-known fact — really", "a well known fact really"),
        ("We met in 1,250 places", "we met in one thousand two hundred fifty places"),
        ("Pi is 3.14", "pi is three point one four"),
        ("The 1st, 2nd, 3rd and 21st", "the first second third and twenty first"),
        ("My 12th and 40th birthday", "my twelfth and fortieth birthday"),
        ("It costs $5 or €20.50", "it costs five dollars or twenty point five zero euros"),
        ("About 50% of £3", "about fifty percent of three pounds"),
        ("Café in Zürich", "cafe in zurich"),
        ("Zurich cafe", "zurich cafe"),
        ("   ", ""),
    ],
)
def test_normalise(text: str, expected: str) -> None:
    assert normalise(text) == expected.split()


@pytest.mark.parametrize(
    ("number", "words"),
    [
        (0, "zero"),
        (7, "seven"),
        (13, "thirteen"),
        (40, "forty"),
        (99, "ninety nine"),
        (100, "one hundred"),
        (105, "one hundred five"),
        (2026, "two thousand twenty six"),
        (1_000_000, "one million"),
        (3_000_000_042, "three billion forty two"),
    ],
)
def test_number_to_words(number: int, words: str) -> None:
    assert number_to_words(number) == words


def test_negative_numbers_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        number_to_words(-1)


def test_counts_substitutions_deletions_and_insertions() -> None:
    errors = score("the cat sat on the mat", "the bat sat on mat today")

    assert (errors.substitutions, errors.deletions, errors.insertions) == (1, 1, 1)
    assert errors.reference_words == 6
    assert errors.rate == pytest.approx(3 / 6)


def test_rate_can_exceed_one_with_insertions() -> None:
    assert word_error_rate("yes", "yes yes yes") == 2.0


def test_empty_hypothesis_is_all_deletions() -> None:
    assert word_error_rate("one two", "") == 1.0


def test_reference_without_words_is_rejected() -> None:
    with pytest.raises(ValueError, match="reference"):
        word_error_rate("...", "anything")


words = st.lists(st.sampled_from(["a", "cat", "sat", "on", "mat", "two"]), min_size=1, max_size=12)


@given(words)
def test_identical_texts_score_zero(ws: list[str]) -> None:
    text = " ".join(ws)
    assert word_error_rate(text, text) == 0.0


@given(words, st.lists(st.sampled_from(["a", "dog", "ran"]), max_size=12))
def test_errors_are_bounded_by_the_longer_text(ref: list[str], hyp: list[str]) -> None:
    errors = score(" ".join(ref), " ".join(hyp))
    total = errors.substitutions + errors.deletions + errors.insertions
    assert abs(len(ref) - len(hyp)) <= total <= max(len(ref), len(hyp))
