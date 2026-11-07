"""Speech dependencies live in the server, free of GPL code; bench extras stay apart."""

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

# GPL grapheme-to-phoneme packages and the package that pulls them in (ADR 0004).
GPL_G2P = {"phonemizer", "phonemizer-fork", "espeakng-loader", "kokoro-onnx"}
VOICE_PACKAGES = {
    "pipecat-ai",
    "faster-whisper",
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12",
    "onnxruntime",
    "misaki",
    "spacy",
    "en-core-web-sm",
    "num2words",
    "soundfile",
    "numpy",
    "huggingface-hub",
}
BENCH_ONLY_PACKAGES = {"nvidia-ml-py"}


def toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def requirement_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
    assert match, requirement
    return re.sub(r"[-_.]+", "-", match.group(0)).lower()


def server_requirements(repo_root: Path) -> dict[str, str]:
    deps = toml(repo_root / "apps" / "server" / "pyproject.toml")["project"]["dependencies"]
    return {requirement_name(r): r for r in deps}


def recipe_body(repo_root: Path, name: str) -> str:
    text = (repo_root / "justfile").read_text(encoding="utf-8")
    body = re.search(rf"^{name}\b[^\n]*:\n((?:[ \t]+\S[^\n]*\n)+)", text, re.M)
    assert body, f"recipe `{name}` has no body"
    return body.group(1)


def test_server_depends_on_the_voice_packages(repo_root: Path) -> None:
    missing = VOICE_PACKAGES - set(server_requirements(repo_root))
    assert not missing, f"server lacks voice dependencies: {sorted(missing)}"


def test_pipecat_has_webrtc_and_no_gpl_or_torch_extras(repo_root: Path) -> None:
    pipecat = server_requirements(repo_root)["pipecat-ai"]
    extras = set(re.search(r"\[([^\]]*)\]", pipecat).group(1).split(","))  # type: ignore[union-attr]
    assert "webrtc" in extras
    # `kokoro` pulls in kokoro-onnx (GPL); `local-smart-turn` pulls in torch.
    assert not extras & {"kokoro", "local-smart-turn"}
    assert "<" in pipecat, "Pipecat is pinned to one minor version"


def test_misaki_is_installed_without_extras(repo_root: Path) -> None:
    assert "[" not in server_requirements(repo_root)["misaki"], "misaki extras pull in GPL packages"


def test_bench_group_holds_only_benchmark_extras(repo_root: Path) -> None:
    group = toml(repo_root / "pyproject.toml")["dependency-groups"]["bench"]
    assert {requirement_name(r) for r in group} == BENCH_ONLY_PACKAGES


def test_bench_group_is_not_a_default_group(repo_root: Path) -> None:
    defaults = toml(repo_root / "pyproject.toml")["tool"]["uv"].get("default-groups", ["dev"])
    assert defaults == ["dev"]


def test_setup_does_not_install_the_bench_group(repo_root: Path) -> None:
    body = recipe_body(repo_root, "setup")
    assert "bench" not in body
    assert "--all-groups" not in body


def test_lockfile_has_no_gpl_g2p_packages(repo_root: Path) -> None:
    names = {p["name"] for p in toml(repo_root / "uv.lock")["package"]}
    assert not names & GPL_G2P, f"GPL G2P packages in uv.lock: {sorted(names & GPL_G2P)}"


def test_opencv_is_the_headless_build(repo_root: Path) -> None:
    names = {p["name"] for p in toml(repo_root / "uv.lock")["package"]}
    assert "opencv-python" not in names, "use opencv-python-headless (ADR 0003)"


def dockerfile_stages(repo_root: Path) -> dict[str, tuple[set[str], str]]:
    """Stage name -> (stages it builds from or copies from, its instructions)."""
    text = (repo_root / "docker" / "server.Dockerfile").read_text(encoding="utf-8")
    # Comments may name what another stage installs; only instructions count.
    text = re.sub(r"^\s*#.*\n", "", text, flags=re.M)
    stages: dict[str, tuple[set[str], str]] = {}
    for block in re.split(r"^(?=FROM )", text, flags=re.M)[1:]:
        header = re.match(r"FROM (\S+) AS (\S+)", block)
        assert header, block.splitlines()[0]
        parents = {header.group(1), *re.findall(r"--from=(\S+)", block)}
        stages[header.group(2)] = (parents, block)
    return stages


def ancestry(stages: dict[str, tuple[set[str], str]], target: str) -> set[str]:
    seen, todo = set(), [target]
    while todo:
        name = todo.pop()
        if name in stages and name not in seen:
            seen.add(name)
            todo.extend(stages[name][0])
    return seen


def holders(stages: dict[str, tuple[set[str], str]], pattern: str) -> set[str]:
    return {name for name, (_, body) in stages.items() if re.search(pattern, body)}


def test_bench_group_stays_in_the_bench_target(repo_root: Path) -> None:
    stages = dockerfile_stages(repo_root)
    found = holders(stages, r"--group bench")
    assert found, "no stage installs the bench group"
    assert found <= ancestry(stages, "bench")
    for target in ("dev", "runtime"):
        assert not found & ancestry(stages, target), f"bench group reaches {target}"


@pytest.mark.parametrize("what", [r"\bespeak-ng\b", r"LD_LIBRARY_PATH"])
@pytest.mark.parametrize("target", ["dev", "runtime", "bench"])
def test_voice_runtime_pieces_reach_every_server_image(
    repo_root: Path, what: str, target: str
) -> None:
    stages = dockerfile_stages(repo_root)
    assert holders(stages, what) & ancestry(stages, target), f"{what} missing from {target}"


def test_no_stage_downloads_model_weights(repo_root: Path) -> None:
    text = (repo_root / "docker" / "server.Dockerfile").read_text(encoding="utf-8")
    assert not re.search(r"huggingface-cli|hf download|\.onnx|voices-v1\.0\.bin|curl |wget ", text)


def test_server_images_run_as_the_unprivileged_user(repo_root: Path) -> None:
    stages = dockerfile_stages(repo_root)
    for target in ("dev", "runtime", "bench"):
        users = re.findall(r"^USER (\S+)$", stages[target][1], re.M)
        assert users, f"{target} sets no user"
        assert users[-1] == "parlora"


def test_dependabot_groups_voice_updates_apart(repo_root: Path) -> None:
    import yaml

    config = yaml.safe_load((repo_root / ".github" / "dependabot.yml").read_text("utf-8"))
    uv = next(u for u in config["updates"] if u["package-ecosystem"] == "uv")
    groups = list(uv["groups"])
    # Dependabot puts an update in the first group that matches it.
    assert groups.index("voice") < groups.index("python")
    patterns = uv["groups"]["voice"]["patterns"]
    for name in VOICE_PACKAGES | BENCH_ONLY_PACKAGES:
        assert any(re.fullmatch(p.replace("*", ".*"), name) for p in patterns), name
