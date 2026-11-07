import pytest

from parlora_core import errors

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "error",
    [
        errors.InvalidLanguageCode,
        errors.UnsupportedLanguage,
        errors.DuplicateLanguage,
        errors.InvalidCefrLevel,
        errors.InvalidLearnerRef,
        errors.InvalidSessionSettings,
        errors.InvalidAudio,
    ],
)
def test_every_error_is_a_domain_error(error: type[errors.DomainError]) -> None:
    assert issubclass(error, errors.DomainError)


@pytest.mark.parametrize(
    "error",
    [
        errors.InvalidLanguageCode,
        errors.DuplicateLanguage,
        errors.InvalidCefrLevel,
        errors.InvalidLearnerRef,
        errors.InvalidSessionSettings,
        errors.InvalidAudio,
    ],
)
def test_invalid_values_are_value_errors(error: type[errors.DomainError]) -> None:
    assert issubclass(error, ValueError)


def test_unsupported_language_is_a_lookup_error() -> None:
    assert issubclass(errors.UnsupportedLanguage, LookupError)


def test_domain_error_is_not_a_builtin_value_error() -> None:
    assert not issubclass(errors.DomainError, ValueError)
