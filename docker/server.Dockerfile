# syntax=docker/dockerfile:1
# Parlora server image (design D12, ADR 0002).
#
# Targets:
#   dev      - editable install; compose.dev.yaml bind-mounts the sources and
#              uvicorn reloads on change.
#   runtime  - the default: a frozen, non-editable install of the server only.
#
# The slim uv image is enough: CUDA user-space libraries will arrive as pip
# wheels with the voice pipeline, cached in the dependency layer below.

FROM ghcr.io/astral-sh/uv:0.13.6-python3.12-trixie-slim@sha256:bf889bda77474aa50c351caf1538b855ef2b7a30add45f26ca50dcb6ade06402 AS base

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# A fixed, unprivileged user; the application never runs as root.
RUN groupadd --system --gid 10001 parlora \
    && useradd --system --uid 10001 --gid parlora --no-create-home --shell /usr/sbin/nologin parlora

WORKDIR /app

# Third-party dependencies only, from the lockfile. This layer changes only
# when uv.lock or a manifest does, so code edits rebuild in seconds.
FROM base AS deps
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/core/pyproject.toml,target=packages/core/pyproject.toml \
    --mount=type=bind,source=apps/server/pyproject.toml,target=apps/server/pyproject.toml \
    uv sync --frozen --no-dev --package parlora-server --no-install-workspace

FROM deps AS dev
ENV PYTHONDONTWRITEBYTECODE=1
COPY pyproject.toml uv.lock ./
COPY packages/core packages/core
COPY apps/server apps/server
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package parlora-server
USER parlora
EXPOSE 8000
# Polling reload: file events do not cross Docker Desktop bind mounts reliably.
CMD ["uvicorn", "--factory", "parlora_server.main:create_app", "--host", "0.0.0.0", "--port", "8000", \
     "--reload", "--reload-dir", "/app/apps/server/src", "--reload-dir", "/app/packages/core/src"]

FROM deps AS build
ENV UV_COMPILE_BYTECODE=1
COPY pyproject.toml uv.lock ./
COPY packages/core packages/core
COPY apps/server apps/server
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package parlora-server --no-editable

FROM base AS runtime
COPY --from=build /opt/venv /opt/venv
USER parlora
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health/live', timeout=2)"]
CMD ["uvicorn", "--factory", "parlora_server.main:create_app", "--host", "0.0.0.0", "--port", "8000"]
