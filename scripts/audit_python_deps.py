"""Python dependency gate (ADR 0005): known vulnerabilities and licence policy.

1. Export the locked dependency set and audit it with pip-audit.
2. Check the licences of the installed environment with pip-licenses: GPL and
   AGPL of any version are denied (ADR 0004), and so are a missing licence and a
   proprietary one. Reviewed exceptions are listed below by package name, each
   with its justification.

Run after `uv sync --locked --all-packages --all-groups` so the installed
environment matches the lockfile, including the optional `bench` group. Exits
non-zero when either check fails.
"""

import fnmatch
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Substrings of denied licence names, matched case-insensitively against the
# unnormalised metadata. "GPL" also matches LGPL, which is allowed only for the
# reviewed packages below.
DENIED_LICENCES = ("General Public License", "GPL", "UNKNOWN", "Proprietary")

# Reviewed exceptions. Adding one needs a documented review and an entry in NOTICE.
REVIEWED_LGPL = {
    "soxr": "LGPL-2.1-or-later, used unmodified through normal imports; pulled in by "
    "Pipecat (ADR 0004)",
    "num2words": "LGPL-2.1, used unmodified through normal imports; misaki's English "
    "G2P needs it (ADR 0004)",
}
# Matched as shell-style patterns, and excused only from the proprietary rule.
REVIEWED_PROPRIETARY = {
    "nvidia-*-cu12": "NVIDIA's CUDA user-space libraries that ctranslate2 loads (ADR "
    "0002); installed locally by the optional bench group and never redistributed, "
    "because no image is published",
}


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def licence_problems(packages: Iterable[Mapping[str, str]]) -> list[str]:
    """`name: licence` for every installed package the policy rejects."""
    problems = []
    for package in packages:
        name, licence = normalise(package["Name"]), package["License"]
        if name in REVIEWED_LGPL:
            continue
        denied = {d for d in DENIED_LICENCES if d.casefold() in licence.casefold()}
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in REVIEWED_PROPRIETARY):
            denied.discard("Proprietary")
        if denied:
            problems.append(f"{package['Name']}: {licence}")
    return problems


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode


def audit_vulnerabilities(requirements: Path) -> int:
    export = run(
        [
            "uv",
            "export",
            "--frozen",
            "--all-packages",
            "--all-groups",
            "--no-hashes",
            "--no-emit-workspace",
            "--quiet",
            "--output-file",
            str(requirements),
        ]
    )
    if export != 0:
        return export
    # --no-deps: the export already holds the full resolved set.
    return run(
        [
            sys.executable,
            "-m",
            "pip_audit",
            "--requirement",
            str(requirements),
            "--disable-pip",
            "--no-deps",
            "--progress-spinner",
            "off",
        ]
    )


def check_licences() -> int:
    cmd = [sys.executable, "-m", "piplicenses", "--from=mixed", "--format=json"]
    print("$", " ".join(cmd), flush=True)
    report = subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    packages = json.loads(report.stdout)
    problems = licence_problems(packages)
    print(f"{len(packages)} installed packages checked.")
    for problem in problems:
        print(f"denied licence: {problem}", file=sys.stderr)
    return 1 if problems else 0


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        vulnerabilities = audit_vulnerabilities(Path(tmp) / "requirements.txt")
    licences = check_licences()
    if vulnerabilities or licences:
        print("Python dependency gate failed (see ADR 0005).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
