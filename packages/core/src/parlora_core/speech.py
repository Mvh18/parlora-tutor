"""Rules for speaking a reply while it is still being written.

The tutor's reply streams from the language model. Text-to-speech starts on the
first clause instead of waiting for the whole reply: the first chunk is cut at the
first clause boundary (`, ; : — . ? !`) once at least `FIRST_MIN_WORDS` words are
buffered, or after `FIRST_MAX_WORDS` words when no boundary comes; later chunks are
cut at sentence ends. A boundary only counts once whitespace follows it, so "3.50"
is never split. Chunks are exact slices of the reply: joined, they give it back.

`spoken_text` removes what a voice must not read: markdown and emojis.
"""

import re
import unicodedata

FIRST_MIN_WORDS = 3
FIRST_MAX_WORDS = 12
CLAUSE_END = ",;:—.?!"
SENTENCE_END = ".?!"
_CLOSERS = "\"')]»\N{RIGHT DOUBLE QUOTATION MARK}\N{RIGHT SINGLE QUOTATION MARK}"


class ClauseSplitter:
    """Feed streamed text; get back the chunks that are ready to be spoken."""

    def __init__(self) -> None:
        self._buffer = ""
        self._first_sent = False

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        chunks = []
        while (cut := self._next_cut()) is not None:
            chunks.append(self._buffer[:cut])
            self._buffer = self._buffer[cut:]
            self._first_sent = True
        return chunks

    def flush(self) -> str | None:
        """The rest of the reply once the stream ends, or `None` if only whitespace is left."""
        rest, self._buffer = self._buffer, ""
        return rest if rest.strip() else None

    def _next_cut(self) -> int | None:
        text = self._buffer
        ends = CLAUSE_END if not self._first_sent else SENTENCE_END
        for index, char in enumerate(text):
            if not (char.isspace() and index and self._is_boundary(text, index, ends)):
                continue
            if self._first_sent:
                return index
            words = len(text[:index].split())
            if words > FIRST_MAX_WORDS:
                break
            if words >= FIRST_MIN_WORDS:
                return index
        if not self._first_sent:
            return self._word_limit_cut(text)
        return None

    @staticmethod
    def _is_boundary(text: str, space: int, ends: str) -> bool:
        before = text[:space].rstrip(_CLOSERS)
        return bool(before) and before[-1] in ends

    @staticmethod
    def _word_limit_cut(text: str) -> int | None:
        words = list(re.finditer(r"\S+", text))
        # Only cut once the next word has started, so the 12th word is known to be complete.
        if len(words) > FIRST_MAX_WORDS:
            return words[FIRST_MAX_WORDS - 1].end()
        return None


_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_LIST_MARKER = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.M)
_HEADING = re.compile(r"^\s*#{1,6}\s*", re.M)
_EMPHASIS = re.compile(r"[*_`~]+")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?])")


def _is_emoji(char: str) -> bool:
    code = ord(char)
    return (
        unicodedata.category(char) == "So"
        or 0x1F000 <= code <= 0x1FAFF
        or 0xFE00 <= code <= 0xFE0F  # variation selectors
        or code == 0x200D  # zero-width joiner
        or 0x1F3FB <= code <= 0x1F3FF  # skin tones
    )


def spoken_text(text: str) -> str:
    """`text` without markdown markup, list markers or emojis, whitespace collapsed."""
    text = _LINK.sub(r"\1", text)
    text = _HEADING.sub("", text)
    text = _LIST_MARKER.sub("", text)
    text = _EMPHASIS.sub("", text)
    text = "".join(" " if _is_emoji(c) else c for c in text)
    text = " ".join(text.split())
    return _SPACE_BEFORE_PUNCTUATION.sub(r"\1", text)
