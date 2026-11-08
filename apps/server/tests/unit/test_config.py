"""Typed configuration from `PARLORA_` environment variables (design D9, D19)."""

from pathlib import Path

import pytest

from parlora_server.config import Settings, SettingsError, load_settings

pytestmark = pytest.mark.unit

SECRET_URL = "postgresql+asyncpg://parlora:s3cr3t-pa55@db.internal:5432/parlora"  # noqa: S105 - fake


def test_defaults_target_the_local_stack(clean_env: None) -> None:
    settings = Settings(_env_file=None)

    url = settings.database.url.get_secret_value()
    assert url.startswith("postgresql+asyncpg://")
    assert ":5433/" in url, "the local stack avoids the host's default PostgreSQL port"
    assert str(settings.llm.base_url) == "http://host.docker.internal:11434/v1"
    assert settings.cors_origins == ["http://localhost:3000"]


def test_nested_values_come_from_double_underscore_variables(
    monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    monkeypatch.setenv("PARLORA_DATABASE__URL", SECRET_URL)
    monkeypatch.setenv("PARLORA_LLM__BASE_URL", "http://127.0.0.1:11434/v1")

    settings = Settings(_env_file=None)

    assert settings.database.url.get_secret_value() == SECRET_URL
    assert str(settings.llm.base_url) == "http://127.0.0.1:11434/v1"


def test_database_url_is_never_shown_in_reprs(
    monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    monkeypatch.setenv("PARLORA_DATABASE__URL", SECRET_URL)

    settings = Settings(_env_file=None)

    assert "s3cr3t-pa55" not in repr(settings)
    assert "s3cr3t-pa55" not in str(settings.model_dump())


def test_cors_origins_are_a_flat_json_list(
    monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    monkeypatch.setenv("PARLORA_CORS_ORIGINS", '["http://localhost:8000", "https://app.example"]')

    settings = Settings(_env_file=None)

    assert settings.cors_origins == ["http://localhost:8000", "https://app.example"]


@pytest.mark.parametrize(
    "origins",
    [
        '["*"]',
        '["http://localhost:3000", "*"]',
        '["http://localhost:3000/path"]',
        '["localhost:3000"]',
        '["ftp://localhost"]',
    ],
)
def test_invalid_cors_origins_fail_fast(
    monkeypatch: pytest.MonkeyPatch, clean_env: None, origins: str
) -> None:
    monkeypatch.setenv("PARLORA_CORS_ORIGINS", origins)

    with pytest.raises(SettingsError, match="PARLORA_CORS_ORIGINS"):
        load_settings(env_file=None)


@pytest.mark.parametrize(
    "url",
    [
        "not a url",
        "mysql://parlora:s3cr3t-pa55@localhost/parlora",
        "postgresql://parlora:s3cr3t-pa55@localhost/parlora",
        "postgresql+asyncpg://parlora:s3cr3t-pa55@/parlora",
    ],
)
def test_invalid_database_url_names_the_variable_without_echoing_it(
    monkeypatch: pytest.MonkeyPatch, clean_env: None, url: str
) -> None:
    monkeypatch.setenv("PARLORA_DATABASE__URL", url)

    with pytest.raises(SettingsError) as excinfo:
        load_settings(env_file=None)

    message = str(excinfo.value)
    assert "PARLORA_DATABASE__URL" in message
    assert "s3cr3t-pa55" not in message
    assert url not in message


def test_load_settings_reads_a_dotenv_file(tmp_path: Path, clean_env: None) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('PARLORA_CORS_ORIGINS=["http://localhost:4000"]\n', encoding="utf-8")

    settings = load_settings(env_file=env_file)

    assert settings.cors_origins == ["http://localhost:4000"]


def test_tts_provider_defaults_to_the_cpu(clean_env: None) -> None:
    assert Settings(_env_file=None).tts.provider == "cpu"


def test_tts_provider_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    monkeypatch.setenv("PARLORA_TTS__PROVIDER", "cuda")
    assert Settings(_env_file=None).tts.provider == "cuda"

    monkeypatch.setenv("PARLORA_TTS__PROVIDER", "tpu")
    with pytest.raises(SettingsError, match="PARLORA_TTS__PROVIDER"):
        load_settings(env_file=None)
