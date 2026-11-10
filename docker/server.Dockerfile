# syntax=docker/dockerfile:1
# Parlora server image (design D12, ADR 0002). Built locally; never published.
#
# Targets:
#   dev      - editable install; compose.dev.yaml bind-mounts the sources and
#              uvicorn reloads on change.
#   bench    - local benchmark image (scripts/bench): the server image plus the
#              benchmark-only extras.
#   runtime  - the default: a frozen, non-editable install of the server only.
#
# Every target carries the voice stack: the CUDA user-space libraries arrive as
# pip wheels in the dependency layer, and the espeak-ng executable speaks
# out-of-vocabulary words as a separate GPL-3.0 process (ADR 0004). Model weights
# are never baked in: they live in a named volume mounted at /models.

FROM ghcr.io/astral-sh/uv:0.13.7-python3.12-trixie-slim@sha256:d4a752dd52f42e0c508bb0b6a912f15922bb32e42f66af30671c3fdc0d759f21 AS base

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    # ctranslate2 does not find the cuBLAS/cuDNN wheel libraries by itself (ADR 0002).
    LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib \
    HF_HOME=/models/huggingface \
    PARLORA_MODELS_DIR=/models

RUN apt-get update \
    && apt-get install --yes --no-install-recommends espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# A fixed, unprivileged user; the application never runs as root. The models
# volume takes /models' owner on first use, so that user can fill the caches.
RUN groupadd --system --gid 10001 parlora \
    && useradd --system --uid 10001 --gid parlora --no-create-home --shell /usr/sbin/nologin parlora \
    && mkdir /models && chown parlora:parlora /models

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

# Component benchmarks (scripts/bench), local only. Adds the benchmark-only
# extras; the repository is mounted at /app at run time.
FROM deps AS bench
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/core/pyproject.toml,target=packages/core/pyproject.toml \
    --mount=type=bind,source=apps/server/pyproject.toml,target=apps/server/pyproject.toml \
    uv sync --frozen --no-dev --group bench --inexact --no-install-workspace
# The benchmarks share the server's adapters: install the workspace packages
# editable, so at run time they load from the repository mounted at /app.
COPY pyproject.toml uv.lock ./
COPY packages/core packages/core
COPY apps/server apps/server
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package parlora-server --inexact
# Benchmark weights live in their own named volume mounted here.
RUN mkdir /cache && chown parlora:parlora /cache
ENV PARLORA_BENCH_CACHE=/cache \
    HF_HOME=/cache/huggingface
USER parlora
CMD ["python", "-c", "import faster_whisper, onnxruntime, misaki, spacy, pipecat; print('bench image ready')"]

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
