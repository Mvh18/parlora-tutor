"""Keep personal data out of committed benchmark results.

Every key and string value of a result is scanned for the current user name, home
directory and host name. User and host names match whole words, case-insensitively;
the home directory matches as a path prefix with either separator. Service accounts
(`root`, the image user `parlora`, the CI user `runner`) and a home of `/` are not
personal and are ignored.
"""

import getpass
import platform
import re
import socket
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

SERVICE_ACCOUNTS = frozenset({"root", "parlora", "runner"})


class PrivacyError(ValueError):
    """A result holds personal data and must not be written."""


@dataclass(frozen=True)
class PrivacyContext:
    user_names: tuple[str, ...]
    home_dirs: tuple[str, ...]
    host_names: tuple[str, ...]

    @classmethod
    def current(cls) -> "PrivacyContext":
        try:
            user = getpass.getuser()
        except (OSError, KeyError):
            user = ""
        users = (user,) if user and user.casefold() not in SERVICE_ACCOUNTS else ()
        home = str(Path.home())
        homes = (home,) if len(home.strip("/\\")) > 1 else ()
        hosts: list[str] = []
        for name in (socket.gethostname(), platform.node()):
            for candidate in (name, name.split(".")[0]):
                if candidate and candidate.casefold() != "localhost" and candidate not in hosts:
                    hosts.append(candidate)
        return cls(user_names=users, home_dirs=homes, host_names=tuple(hosts))


def _word(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", re.IGNORECASE)


def _path(text: str) -> str:
    return text.replace("\\", "/").casefold()


def _reasons(text: str, context: PrivacyContext) -> Iterator[str]:
    if any(_path(home) in _path(text) for home in context.home_dirs):
        yield "home directory"
    if any(_word(user).search(text) for user in context.user_names):
        yield "user name"
    if any(_word(host).search(text) for host in context.host_names):
        yield "host name"


def _walk(value: object, where: str) -> Iterator[tuple[str, str, bool]]:
    """(location, text, is_key) for every key and string value."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield where, str(key), True
            yield from _walk(item, f"{where}.{key}" if where else str(key))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            yield from _walk(item, f"{where}[{index}]")
    elif isinstance(value, str):
        yield where, value, False


def find_personal_data(result: object, context: PrivacyContext) -> list[str]:
    """One finding per offending key or value, e.g. `settings.cache: home directory`."""
    findings = []
    for where, text, is_key in _walk(result, ""):
        for reason in _reasons(text, context):
            findings.append(
                f"{where}: {reason} in key {text!r}" if is_key else f"{where}: {reason}"
            )
    return findings
