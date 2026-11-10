"""System prompts for the voice call, one Markdown file per language and purpose.

Prompts live in `prompts/<language>/<name>.md` at the repository root (copied to
`/app/prompts` in the server image) so they can be read and reviewed as content,
not code. Adding a language is a new folder, not a code change.
"""

from functools import cache
from pathlib import Path

from parlora_core.errors import UnsupportedLanguage
from parlora_core.languages import LanguageCode

PROMPTS = "prompts"


@cache
def prompts_dir() -> Path:
    """The repository's `prompts/` folder in a checkout, else `prompts/` in the working dir."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / PROMPTS
        if (candidate / "en").is_dir():
            return candidate
    return Path.cwd() / PROMPTS


def load_prompt(
    language: LanguageCode, name: str = "conversation", root: Path | None = None
) -> str:
    """The text of `prompts/<language>/<name>.md`, stripped."""
    if not name.isidentifier():
        raise ValueError(f"{name!r} is not a prompt name")
    path = (root or prompts_dir()) / str(language) / f"{name}.md"
    if not path.is_file():
        raise UnsupportedLanguage(f"no {name} prompt for {language}")
    return path.read_text(encoding="utf-8").strip()
