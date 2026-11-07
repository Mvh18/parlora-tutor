"""Domain error hierarchy.

Every error raised by the core derives from `DomainError`, so adapters can map
the whole family to one response type. Invalid values also derive from
`ValueError` and unknown lookups from `LookupError`, so callers that only know
the built-in exceptions still behave sensibly.
"""


class DomainError(Exception):
    """Base class for every error raised by the domain core."""


class InvalidLanguageCode(DomainError, ValueError):
    """A language code is not a two-letter lowercase ISO 639-1 code."""


class UnsupportedLanguage(DomainError, LookupError):
    """A language is unknown to the registry or unsupported in the requested role."""


class DuplicateLanguage(DomainError, ValueError):
    """A registry received two entries for the same language code."""


class InvalidCefrLevel(DomainError, ValueError):
    """A value is not one of the CEFR levels A1-C2."""


class InvalidLearnerRef(DomainError, ValueError):
    """A learner reference is malformed or looks like personal data."""


class InvalidSessionSettings(DomainError, ValueError):
    """Session settings break a domain rule."""


class InvalidAudio(DomainError, ValueError):
    """Audio or a transcript does not have a valid shape."""
