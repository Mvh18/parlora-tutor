"""The local Compose stack is self-contained, local-only and coexists with other stacks.

Most checks read the configuration as Compose renders it (`docker compose
config`), so interpolation defaults and override files are what is tested.
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

LOOPBACK = "127.0.0.1"
# Taken by the Laravel course platform and by a native or Sail PostgreSQL (design D19).
RESERVED_HOST_PORTS = {"8000", "5432"}
DOCKERFILE = Path("docker") / "server.Dockerfile"


def render(
    repo_root: Path, tmp_path: Path, *files: str, env: dict[str, str] | None = None
) -> dict[str, Any]:
    """The rendered config of every service (all profiles), ignoring any local `.env`."""
    empty_env_file = tmp_path / "empty.env"
    empty_env_file.write_text("", encoding="utf-8")
    process_env = {
        k: v for k, v in os.environ.items() if not k.startswith(("PARLORA_", "COMPOSE_"))
    }
    args = ["docker", "compose", "--env-file", str(empty_env_file), "--profile", "*"]
    for name in files or ("compose.yaml",):
        args += ["--file", name]
    result = subprocess.run(
        [*args, "config", "--format", "json"],
        cwd=repo_root,
        env=process_env | (env or {}),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"
    config = json.loads(result.stdout)
    assert isinstance(config, dict)
    return config


def published(config: dict[str, Any]) -> list[tuple[str, str, str]]:
    """(service, host_ip, published port) for every published port."""
    return [
        (name, port.get("host_ip", ""), str(port.get("published", "")))
        for name, service in config["services"].items()
        for port in service.get("ports", [])
    ]


def gpu_reservations(service: dict[str, Any]) -> list[dict[str, Any]]:
    devices = service.get("deploy", {}).get("resources", {}).get("reservations", {})
    return [d for d in devices.get("devices", []) if "gpu" in d.get("capabilities", [])]


@pytest.fixture(scope="module")
def raw_compose(repo_root: Path) -> dict[str, Any]:
    data = yaml.safe_load((repo_root / "compose.yaml").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


@pytest.fixture
def config(repo_root: Path, tmp_path: Path) -> dict[str, Any]:
    return render(repo_root, tmp_path)


def test_project_is_named_parlora(config: dict[str, Any]) -> None:
    assert config["name"] == "parlora"


def test_every_published_port_is_bound_to_loopback(repo_root: Path, tmp_path: Path) -> None:
    for files in [
        ("compose.yaml",),
        ("compose.yaml", "compose.nogpu.yaml"),
        ("compose.yaml", "compose.dev.yaml"),
    ]:
        exposed = [
            f"{service} publishes {port} on {host_ip or 'all interfaces'}"
            for service, host_ip, port in published(render(repo_root, tmp_path, *files))
            if host_ip != LOOPBACK
        ]
        assert not exposed, f"{' + '.join(files)}: {exposed}"


def test_default_host_ports_avoid_common_collisions(config: dict[str, Any]) -> None:
    ports = {service: port for service, _, port in published(config)}
    assert ports["server"] == "8700"
    assert ports["postgres"] == "5433"
    clashing = [f"{s}:{p}" for s, _, p in published(config) if p in RESERVED_HOST_PORTS]
    assert not clashing, f"default host ports collide with other local stacks: {clashing}"


@pytest.mark.parametrize(
    ("variable", "service"),
    [("PARLORA_SERVER_HOST_PORT", "server"), ("PARLORA_POSTGRES_HOST_PORT", "postgres")],
)
def test_host_ports_can_be_overridden(
    repo_root: Path, tmp_path: Path, variable: str, service: str
) -> None:
    config = render(repo_root, tmp_path, env={variable: "8900"})
    assert (service, LOOPBACK, "8900") in published(config)


def test_base_stack_has_no_external_dependencies(raw_compose: dict[str, Any]) -> None:
    external = [
        f"{kind} {name}"
        for kind in ("networks", "volumes")
        for name, spec in (raw_compose.get(kind) or {}).items()
        if isinstance(spec, dict) and spec.get("external")
    ]
    assert not external, f"compose.yaml declares external resources: {external}"


def test_images_are_pinned_by_tag_and_digest(config: dict[str, Any]) -> None:
    unpinned = [
        f"{name}: {service['image']}"
        for name, service in config["services"].items()
        if "build" not in service
        and not re.fullmatch(r"[^:@]+:[^:@]+@sha256:[0-9a-f]{64}", service["image"])
    ]
    assert not unpinned, f"images without tag and digest: {unpinned}"


def test_server_starts_after_a_healthy_postgres(config: dict[str, Any]) -> None:
    server = config["services"]["server"]
    assert server["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert "pg_isready" in " ".join(config["services"]["postgres"]["healthcheck"]["test"])
    assert "/api/v1/health/live" in " ".join(server["healthcheck"]["test"])


def test_server_is_built_from_the_runtime_target(config: dict[str, Any]) -> None:
    build = config["services"]["server"]["build"]
    assert Path(build["dockerfile"]).as_posix().endswith(DOCKERFILE.as_posix())
    assert build["target"] == "runtime"


def test_server_reaches_the_host(config: dict[str, Any]) -> None:
    assert "host.docker.internal=host-gateway" in config["services"]["server"]["extra_hosts"]


def test_server_talks_to_the_postgres_service(config: dict[str, Any]) -> None:
    url = config["services"]["server"]["environment"]["PARLORA_DATABASE__URL"]
    assert "@postgres:5432/" in url


def test_gpu_is_reserved_by_default_and_removed_by_nogpu(repo_root: Path, tmp_path: Path) -> None:
    gpu = render(repo_root, tmp_path)["services"]
    nogpu = render(repo_root, tmp_path, "compose.yaml", "compose.nogpu.yaml")["services"]
    assert gpu_reservations(gpu["server"])
    assert gpu_reservations(gpu["ollama"])
    assert not [name for name, service in nogpu.items() if gpu_reservations(service)]


def test_ollama_is_an_optional_profile(config: dict[str, Any]) -> None:
    ollama = config["services"]["ollama"]
    assert ollama["profiles"] == ["ollama"]
    assert any(v["target"] == "/root/.ollama" and v["type"] == "volume" for v in ollama["volumes"])


def test_dev_override_runs_the_dev_target_with_source_mounts(
    repo_root: Path, tmp_path: Path
) -> None:
    server = render(repo_root, tmp_path, "compose.yaml", "compose.dev.yaml")["services"]["server"]
    assert server["build"]["target"] == "dev"
    mounted = {
        Path(v["source"]).relative_to(repo_root).as_posix()
        for v in server["volumes"]
        if v["type"] == "bind"
    }
    assert mounted == {"packages/core/src", "apps/server/src"}


def test_dockerfile_has_dev_and_runtime_targets_and_a_non_root_user(repo_root: Path) -> None:
    text = (repo_root / DOCKERFILE).read_text(encoding="utf-8")
    stages = re.findall(r"^FROM \S+ AS (\S+)$", text, re.MULTILINE)
    assert {"dev", "runtime"} <= set(stages)
    assert stages[-1] == "runtime", "runtime must be the default (last) target"
    users = re.findall(r"^USER (\S+)$", text, re.MULTILINE)
    assert users
    assert not {"root", "0"} & set(users)
    syncs = re.findall(r"uv sync[^\n]*", text)
    assert syncs
    assert all("--frozen" in s or "--locked" in s for s in syncs)


def test_dockerfile_base_images_are_pinned_by_digest(repo_root: Path) -> None:
    text = (repo_root / DOCKERFILE).read_text(encoding="utf-8")
    stages = set(re.findall(r"^FROM \S+ AS (\S+)$", text, re.MULTILINE))
    external = [i for i in re.findall(r"^FROM (\S+)", text, re.MULTILINE) if i not in stages]
    assert external
    assert all(re.search(r"@sha256:[0-9a-f]{64}$", image) for image in external), external


def test_dockerignore_keeps_secrets_out_of_the_build_context(repo_root: Path) -> None:
    lines = (repo_root / ".dockerignore").read_text(encoding="utf-8").splitlines()
    rules = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    # Allow-list: exclude everything, then re-include only what the image needs.
    assert rules[0] == "*"
    assert not [rule for rule in rules if ".env" in rule and rule.startswith("!")]


def test_server_keeps_model_weights_in_a_named_volume(config: dict[str, Any]) -> None:
    mounts = config["services"]["server"].get("volumes", [])
    models = [m for m in mounts if m.get("target") == "/models"]
    assert models, "the server mounts no volume at /models"
    assert models[0]["type"] == "volume"
    assert models[0]["source"] == "parlora-models"


def test_nogpu_turns_the_voice_pipeline_off(repo_root: Path, tmp_path: Path) -> None:
    config = render(repo_root, tmp_path, "compose.yaml", "compose.nogpu.yaml")
    assert config["services"]["server"]["environment"]["PARLORA_VOICE__ENABLED"] == "false"
