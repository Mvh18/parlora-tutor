"""Typed settings loaded from `PARLORA_` environment variables (design D9).

Nested groups use a double underscore (`PARLORA_DATABASE__URL`); lists are
JSON (`PARLORA_CORS_ORIGINS='["http://localhost:3000"]'`). Invalid values stop
the application at startup with a message naming the variable, never its value.
`.env.example` documents every variable; a repository test keeps it in sync.
"""

from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    SecretStr,
    ValidationError,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

ENV_PREFIX = "PARLORA_"
NESTED_DELIMITER = "__"

# The local Compose stack publishes PostgreSQL on 5433 so it never collides
# with a PostgreSQL already running on the host (design D12, D19).
DEFAULT_DATABASE_URL = "postgresql+asyncpg://parlora:parlora@localhost:5433/parlora"
DATABASE_DRIVER = "postgresql+asyncpg"


class SettingsError(Exception):
    """The configuration is invalid; the message names variables, not values."""


class DatabaseSettings(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    url: SecretStr = Field(
        default=SecretStr(DEFAULT_DATABASE_URL),
        description="SQLAlchemy URL of the PostgreSQL database, using the asyncpg driver.",
    )

    @field_validator("url")
    @classmethod
    def _postgresql_asyncpg(cls, value: SecretStr) -> SecretStr:
        # Messages never include the URL: it carries the password.
        try:
            url = make_url(value.get_secret_value())
        except ArgumentError:
            raise ValueError("must be a database URL") from None
        if url.drivername != DATABASE_DRIVER:
            raise ValueError(f"must use the {DATABASE_DRIVER}:// scheme")
        if not url.host or not url.database:
            raise ValueError("must name a host and a database")
        return value


class VoiceSettings(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    enabled: bool = Field(
        default=True,
        description=(
            "Load the speech models and accept calls; false (no GPU, CI) reports the voice "
            "checks as disabled."
        ),
    )


class VadSettings(BaseModel):
    """Turn detection: Silero VAD finds speech, Smart Turn v3 decides the turn has ended."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    confidence: float = Field(
        default=0.7, gt=0, lt=1, description="Silero VAD confidence that a frame is speech."
    )
    start_secs: float = Field(
        default=0.2,
        gt=0,
        le=2,
        description="Speech, in seconds, before the learner counts as speaking.",
    )
    stop_secs: float = Field(
        default=0.2,
        gt=0,
        le=2,
        description="Silence, in seconds, before VAD reports that speech stopped.",
    )
    min_volume: float = Field(
        default=0.6, ge=0, le=1, description="Minimum volume for a frame to count as speech."
    )
    turn_fallback_secs: float = Field(
        default=3.0,
        gt=0,
        le=10,
        description="Silence, in seconds, after which a turn ends even if Smart Turn is unsure.",
    )


class SttSettings(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    # ADR 0002 and ADR 0009: about 0.4 s for a 3-9 s utterance on the RTX 4060.
    model: str = Field(
        default="large-v3-turbo",
        min_length=1,
        description="faster-whisper model name, downloaded into the Hugging Face cache.",
    )
    device: Literal["cuda", "cpu"] = Field(
        default="cuda", description="Device for speech-to-text; calls need cuda for latency."
    )
    compute_type: Literal["int8", "int8_float16", "float16", "float32"] = Field(
        default="int8", description="CTranslate2 compute type of the speech-to-text model."
    )
    beam_size: int = Field(default=5, ge=1, le=10, description="Beam size of the decoder.")


class LlmSettings(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    # Ollama's native API, not its OpenAI-compatible /v1 endpoint, which hides the
    # model's load and evaluation timings.
    base_url: HttpUrl = Field(
        default=HttpUrl("http://host.docker.internal:11434"),
        description="Base URL of Ollama's native API (Ollama on the host).",
    )
    model: str = Field(
        default="qwen3:4b-instruct-2507-q4_K_M",
        min_length=1,
        description="Ollama model of the conversation partner.",
    )
    num_ctx: int = Field(
        default=4096, ge=512, description="Context window, in tokens, Ollama loads the model with."
    )
    keep_alive: int = Field(
        default=-1,
        description="Seconds Ollama keeps the model loaded after a request; -1 keeps it resident.",
    )


class TtsSettings(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    # cuda by measurement: first audio p95 about 0.4 s against about 1.9 s on the CPU
    # (benchmarks/results, RTX 4060).
    provider: Literal["cpu", "cuda"] = Field(
        default="cuda",
        description=(
            "onnxruntime provider for Kokoro text-to-speech; cuda falls back to the CPU "
            "with a warning when onnxruntime cannot use it."
        ),
    )
    voice: str = Field(
        default="af_heart",
        pattern=r"^[a-z]{2}_[a-z]+$",
        description="Kokoro voice of the tutor, e.g. af_heart.",
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=NESTED_DELIMITER,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
    )

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    vad: VadSettings = Field(default_factory=VadSettings)
    stt: SttSettings = Field(default_factory=SttSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    tts: TtsSettings = Field(default_factory=TtsSettings)
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        description="Browser origins allowed to call the API (JSON list, no wildcard).",
    )

    @field_validator("cors_origins")
    @classmethod
    def _explicit_origins(cls, origins: list[str]) -> list[str]:
        for origin in origins:
            if origin == "*":
                raise ValueError("the wildcard '*' is not allowed; list each origin")
            parts = urlsplit(origin)
            if parts.scheme not in ("http", "https") or not parts.netloc:
                raise ValueError(f"{origin!r} is not an http(s) origin")
            if parts.path or parts.query or parts.fragment:
                raise ValueError(f"{origin!r} is not an origin (scheme://host[:port] only)")
        return origins


def load_settings(env_file: Path | str | None = ".env") -> Settings:
    """Load the settings, turning validation errors into a `SettingsError`."""
    try:
        return Settings(_env_file=env_file)
    except ValidationError as error:
        problems = [
            f"{env_var(error_['loc'])}: {error_['msg']}"
            for error_ in error.errors(include_input=False)
        ]
        raise SettingsError("invalid configuration:\n  " + "\n  ".join(problems)) from None


def env_var(loc: tuple[int | str, ...]) -> str:
    """The environment variable for a settings field location, e.g. PARLORA_DATABASE__URL."""
    names = [str(part).upper() for part in loc if isinstance(part, str)]
    return ENV_PREFIX + NESTED_DELIMITER.join(names)
