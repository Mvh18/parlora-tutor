"""Results never carry the user name, home directory or host name."""

import getpass
import socket
from pathlib import Path

import pytest

from scripts.bench.common.privacy import PrivacyContext, find_personal_data
from tests.bench.conftest import sample_result

pytestmark = pytest.mark.unit


def test_clean_result_has_no_findings(privacy: PrivacyContext) -> None:
    assert find_personal_data(sample_result(), privacy) == []


@pytest.mark.parametrize(
    ("block", "key", "value", "reason"),
    [
        ("settings", "cache", "C:\\Users\\ada\\AppData\\Local\\parlora", "home directory"),
        ("settings", "cache", "c:/users/ADA/.cache", "home directory"),
        ("software", "built_by", "Ada", "user name"),
        ("settings", "note", "run on ada-laptop.local", "host name"),
    ],
)
def test_personal_data_in_any_value_is_found(
    privacy: PrivacyContext, block: str, key: str, value: str, reason: str
) -> None:
    result = sample_result()
    result[block][key] = value

    findings = find_personal_data(result, privacy)

    # A home directory or host name may also contain the user name; both are reported.
    assert f"{block}.{key}: {reason}" in findings
    assert all(finding.startswith(f"{block}.{key}: ") for finding in findings)


def test_nested_values_and_lists_are_scanned(privacy: PrivacyContext) -> None:
    result = sample_result()
    result["measurements"]["runs"] = [{"path": "/home/x"}, {"log": "user ada started"}]

    assert find_personal_data(result, privacy) == ["measurements.runs[1].log: user name"]


def test_keys_are_scanned(privacy: PrivacyContext) -> None:
    result = sample_result()
    result["settings"]["ada"] = 1

    assert find_personal_data(result, privacy) == ["settings: user name in key 'ada'"]


def test_user_name_only_matches_a_whole_word(privacy: PrivacyContext) -> None:
    result = sample_result()
    result["settings"]["model"] = "adapter-v2 canada"

    assert find_personal_data(result, privacy) == []


def test_current_identity_is_detected() -> None:
    context = PrivacyContext.current()
    result = sample_result()
    result["settings"]["home"] = str(Path.home() / "models")
    result["settings"]["host"] = socket.gethostname()

    findings = find_personal_data(result, context)

    assert "settings.home: home directory" in findings
    assert "settings.host: host name" in findings


def test_current_context_holds_the_user_unless_it_is_a_service_account() -> None:
    context = PrivacyContext.current()
    user = getpass.getuser()
    expected = () if user.casefold() in {"root", "parlora", "runner"} else (user,)

    assert context.user_names == expected


def test_trivial_home_directories_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/")))

    assert PrivacyContext.current().home_dirs == ()


def test_unknown_user_is_tolerated(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_user() -> str:
        raise OSError("no login name")

    monkeypatch.setattr(getpass, "getuser", no_user)

    assert PrivacyContext.current().user_names == ()
