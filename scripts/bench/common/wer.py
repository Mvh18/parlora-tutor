"""Word error rate of a transcription against its reference, after normalisation.

Both texts are normalised the same way before scoring, so formatting choices of the
speech-to-text model do not count as errors:

- Unicode NFKC, typographic apostrophes to ASCII, accents dropped ("café" ->
  "cafe"), case folded.
- English contractions expanded ("I'm" -> "i am", "can't" -> "can not",
  "it's" -> "it is"); a remaining possessive "'s" keeps its letters ("annas").
- Numbers spelled out: integers with thousands separators ("1,250"), decimals digit
  by digit after "point" ("3.14" -> "three point one four"), ordinals ("21st"),
  a currency sign before a number ("$5" -> "five dollars") and "%" -> "percent".
  Years read as pairs ("twenty twenty-six") and clock times are not handled:
  reference texts write them in words.
- Hyphens, dashes and every other punctuation mark become spaces.

The rate is (substitutions + deletions + insertions) / reference words, from a
word-level Levenshtein alignment; it exceeds 1.0 when the hypothesis inserts many
words.
"""

import re
import unicodedata
from dataclasses import dataclass

_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_SCALES = ((10**9, "billion"), (10**6, "million"), (10**3, "thousand"))
_ORDINAL_WORDS = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}
_CURRENCIES = {"$": "dollars", "£": "pounds", "€": "euros"}

_WHOLE_CONTRACTIONS = {
    "won't": "will not",
    "can't": "can not",
    "shan't": "shall not",
    "ain't": "is not",
    "let's": "let us",
}
_IS_CONTRACTIONS = re.compile(r"\b(it|that|what|there|here|he|she|who|where|how)'s\b")
_SUFFIX_CONTRACTIONS = (
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'d\b"), " would"),
    (re.compile(r"'m\b"), " am"),
)
_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_CURRENCY = re.compile(rf"([$£€])\s?({_NUMBER})")
_ORDINAL = re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d+)(?:st|nd|rd|th)\b")
_PLAIN_NUMBER = re.compile(_NUMBER)
_NOT_WORD = re.compile(r"[^\w']+|_")


def number_to_words(number: int) -> str:
    """Spell a non-negative integer in English, without "and" ("one hundred five")."""
    if number < 0:
        raise ValueError(f"only non-negative numbers are spelled, got {number}")
    if number < 20:
        return _ONES[number]
    if number < 100:
        tens, ones = divmod(number, 10)
        return _TENS[tens] + (f" {_ONES[ones]}" if ones else "")
    if number < 1000:
        hundreds, rest = divmod(number, 100)
        return f"{_ONES[hundreds]} hundred" + (f" {number_to_words(rest)}" if rest else "")
    size, name = next((size, name) for size, name in _SCALES if number >= size)
    head, rest = divmod(number, size)
    return f"{number_to_words(head)} {name}" + (f" {number_to_words(rest)}" if rest else "")


def _spell(token: str) -> str:
    whole, _, decimals = token.replace(",", "").partition(".")
    words = number_to_words(int(whole))
    if decimals:
        words += " point " + " ".join(_ONES[int(d)] for d in decimals)
    return words


def _ordinal(token: str) -> str:
    *head, last = number_to_words(int(token.replace(",", ""))).split()
    if last in _ORDINAL_WORDS:
        last = _ORDINAL_WORDS[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return " ".join([*head, last])


def _expand_contractions(text: str) -> str:
    for word, expansion in _WHOLE_CONTRACTIONS.items():
        text = re.sub(rf"\b{re.escape(word)}", expansion, text)
    text = _IS_CONTRACTIONS.sub(r"\1 is", text)
    for pattern, expansion in _SUFFIX_CONTRACTIONS:
        text = pattern.sub(expansion, text)
    return text


def _spell_numbers(text: str) -> str:
    text = _CURRENCY.sub(lambda m: f" {_spell(m[2])} {_CURRENCIES[m[1]]} ", text)
    text = _ORDINAL.sub(lambda m: f" {_ordinal(m[1])} ", text)
    text = _PLAIN_NUMBER.sub(lambda m: f" {_spell(m[0])} ", text)
    return text.replace("%", " percent ")


def normalise(text: str) -> list[str]:
    """The words of `text` after the normalisation described in the module docstring."""
    text = (
        unicodedata.normalize("NFKC", text)
        .replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")
        .replace("\N{LEFT SINGLE QUOTATION MARK}", "'")
    )
    # Accents dropped: "Zürich" and "Zurich" are the same word to a listener.
    decomposed = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in decomposed if not unicodedata.combining(c))
    text = _expand_contractions(text.casefold())
    text = _spell_numbers(text)
    text = _NOT_WORD.sub(" ", text).replace("'", "")
    return text.split()


@dataclass(frozen=True)
class WordErrors:
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def rate(self) -> float:
        return (self.substitutions + self.deletions + self.insertions) / self.reference_words


def score(reference: str, hypothesis: str) -> WordErrors:
    """Align the normalised words and count each kind of error."""
    ref, hyp = normalise(reference), normalise(hypothesis)
    if not ref:
        raise ValueError("the reference text has no words to score against")
    # cost[i][j] = (edits, substitutions, deletions, insertions) for ref[:i] vs hyp[:j].
    previous = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i, ref_word in enumerate(ref, start=1):
        current = [(i, 0, i, 0)]
        for j, hyp_word in enumerate(hyp, start=1):
            diagonal = previous[j - 1]
            if ref_word == hyp_word:
                best = diagonal
            else:
                sub = (diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3])
                up = previous[j]
                dele = (up[0] + 1, up[1], up[2] + 1, up[3])
                left = current[j - 1]
                ins = (left[0] + 1, left[1], left[2], left[3] + 1)
                best = min(sub, dele, ins)
            current.append(best)
        previous = current
    _, substitutions, deletions, insertions = previous[-1]
    return WordErrors(substitutions, deletions, insertions, len(ref))


def word_error_rate(reference: str, hypothesis: str) -> float:
    return score(reference, hypothesis).rate
