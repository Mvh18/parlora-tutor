"""Public documentation: community files, README content, notices, ADRs and links."""

import re
import subprocess
from functools import cache
from pathlib import Path, PurePosixPath

import pytest

pytestmark = pytest.mark.unit

COMMUNITY_FILES = [
    "README.md",
    "README.es.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "SUPPORT.md",
    "NOTICE",
    "AGENTS.md",
    "CLAUDE.md",
]
README_SECTIONS = [
    "Quickstart",
    "Architecture",
    "Privacy",
    "Development",
    "Testing",
    "Roadmap",
    "Contributing",
    "License",
]
CONTACT = "team.parlora@gmail.com"
ADR_SECTIONS = ["Context", "Decision", "Consequences"]
OPEN_SOURCE = re.compile(r"open[ -]source|código abierto", re.IGNORECASE)
NEGATION = re.compile(r"\b(not|no|isn't|nor|ni)\b", re.IGNORECASE)
# The repository was renamed from Mvh18/parlora; the old name only redirects.
OLD_REPO = re.compile(r"Mvh18/parlora(?![\w-])")

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)
FENCE = re.compile(r"^(`{3,}|~{3,}).*?^\1\s*$", re.MULTILINE | re.DOTALL)
INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
REFERENCE_LINK = re.compile(r"^\s*\[[^\]]+\]:\s+<?(\S+?)>?(?:\s+\"[^\"]*\")?\s*$", re.MULTILINE)
SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)


@cache
def doc_files(repo_root: Path) -> tuple[PurePosixPath, ...]:
    """Markdown files git tracks or would track (new files count before `git add`)."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "*.md"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    names = {PurePosixPath(name) for name in out.split("\0") if name}
    return tuple(sorted(n for n in names if (repo_root / n).is_file()))


def read(repo_root: Path, name: str | PurePosixPath) -> str:
    return (repo_root / name).read_text(encoding="utf-8")


def strip_code(text: str) -> str:
    """Drop fenced blocks and inline code, where links and headings are not real."""
    return re.sub(r"`[^`\n]*`", "", FENCE.sub("", text))


def headings(text: str) -> list[str]:
    return [m[2] for m in HEADING.finditer(strip_code(text))]


def slug(heading: str) -> str:
    """GitHub's anchor for a heading: lowercase, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # keep link text only
    text = re.sub(r"[^\w\- ]", "", text.replace("`", "").lower())
    return text.replace(" ", "-")


def anchors(text: str) -> set[str]:
    seen: dict[str, int] = {}
    result = set()
    for heading in headings(text):
        base = slug(heading)
        count = seen.get(base, 0)
        seen[base] = count + 1
        result.add(base if count == 0 else f"{base}-{count}")
    return result


def links(text: str) -> list[str]:
    body = strip_code(text)
    return [m[1] for m in INLINE_LINK.finditer(body)] + [
        m[1] for m in REFERENCE_LINK.finditer(body)
    ]


def sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text)
    return re.split(r"(?<=[.!?])\s+", flat)


@pytest.mark.parametrize("name", COMMUNITY_FILES)
def test_community_file_exists_and_is_not_empty(repo_root: Path, name: str) -> None:
    path = repo_root / name
    assert path.is_file(), f"missing {name}"
    assert path.read_text(encoding="utf-8").strip(), f"{name} is empty"


@pytest.mark.parametrize(
    ("source", "target"), [("README.md", "README.es.md"), ("README.es.md", "README.md")]
)
def test_readmes_link_to_each_other(repo_root: Path, source: str, target: str) -> None:
    targets = {link.split("#")[0] for link in links(read(repo_root, source))}
    assert target in targets, f"{source} does not link to {target}"


@pytest.mark.parametrize("section", README_SECTIONS)
def test_readme_has_section(repo_root: Path, section: str) -> None:
    found = headings(read(repo_root, "README.md"))
    assert any(section.lower() in h.lower() for h in found), (
        f"README.md has no heading for {section}; headings: {found}"
    )


@pytest.mark.parametrize("name", ["README.md", "README.es.md", "CONTRIBUTING.md"])
def test_docs_do_not_claim_open_source(repo_root: Path, name: str) -> None:
    claims = [
        s
        for s in sentences(read(repo_root, name))
        if OPEN_SOURCE.search(s) and not NEGATION.search(s)
    ]
    assert not claims, f"{name} calls the project open source: {claims}"


@pytest.mark.parametrize("name", ["README.md", "README.es.md", "CODE_OF_CONDUCT.md"])
def test_commercial_and_conduct_contact_present(repo_root: Path, name: str) -> None:
    assert CONTACT in read(repo_root, name)


def test_claude_md_imports_agents_md(repo_root: Path) -> None:
    lines = read(repo_root, "CLAUDE.md").splitlines()
    assert "@AGENTS.md" in (line.strip() for line in lines)


def test_notice_names_the_gpl_component(repo_root: Path) -> None:
    text = read(repo_root, "NOTICE")
    assert "espeak-ng" in text
    assert "GPL-3.0" in text


def test_adrs_are_numbered_without_gaps(repo_root: Path) -> None:
    numbers = sorted(
        int(p.name[:4]) for p in (repo_root / "docs" / "adr").glob("[0-9][0-9][0-9][0-9]-*.md")
    )
    assert numbers[:5] == [1, 2, 3, 4, 5], f"ADRs 0001 to 0005 must exist, found {numbers}"
    assert numbers == list(range(1, len(numbers) + 1)), (
        f"ADR numbers have gaps or repeats: {numbers}"
    )


@pytest.mark.parametrize("section", ADR_SECTIONS)
def test_every_adr_has_section(repo_root: Path, section: str) -> None:
    missing = [
        p.name
        for p in sorted((repo_root / "docs" / "adr").glob("[0-9][0-9][0-9][0-9]-*.md"))
        if section not in headings(p.read_text(encoding="utf-8"))
    ]
    assert not missing, f"ADRs without a {section} section: {missing}"


def test_relative_links_resolve(repo_root: Path) -> None:
    """Every relative link and anchor in the Markdown files points at something real.

    External URLs are not fetched: the check stays offline and deterministic.
    """
    broken = []
    for doc in doc_files(repo_root):
        for link in links(read(repo_root, doc)):
            if SCHEME.match(link):
                continue
            path_part, _, anchor = link.partition("#")
            if path_part:
                base = repo_root if path_part.startswith("/") else repo_root / doc.parent
                target = (base / path_part.lstrip("/")).resolve()
            else:
                target = repo_root / doc
            if not target.exists():
                broken.append(f"{doc}: {link}")
            elif (
                anchor
                and target.suffix == ".md"
                and anchor not in anchors(target.read_text(encoding="utf-8"))
            ):
                broken.append(f"{doc}: {link} (no such heading)")
    assert not broken, "broken links:\n" + "\n".join(broken)


def test_docs_use_the_current_repository_name(repo_root: Path) -> None:
    stale = [str(doc) for doc in doc_files(repo_root) if OLD_REPO.search(read(repo_root, doc))]
    for name in [".github/ISSUE_TEMPLATE/config.yml"]:
        if OLD_REPO.search(read(repo_root, name)):
            stale.append(name)
    assert not stale, f"links to the old Mvh18/parlora repository: {stale}"
