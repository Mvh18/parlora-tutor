# Parlora development environment: the whole toolchain lives here, so the host
# only needs Docker Desktop, an editor with Dev Containers support, and git.
# Every tool is pinned; base images by tag and digest, binaries by checksum.

FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv
FROM node:24.21.0-bookworm-slim@sha256:d6aa754f16b3197301076f047b5def2f02ea1dbbc2ca920407d46d7ec7f87b20 AS node
FROM docker:29.9.0-cli@sha256:1a4c7cb63513f349bdad01fcc6e0f3f2f67d37b9da86f14dc0d4a0942eecda00 AS docker

FROM python:3.12.15-bookworm@sha256:5560e9ab8709f459489e5b8aa696eda8a07ef821e14bb122be62d91234bfa98b

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000
ARG JUST_VERSION=1.58.0
ARG JUST_SHA256=4a5cc2f53e6f0f8c59092a6cc38291eb729d46a7dd95d3ae582008881b84931d
ARG GH_VERSION=2.102.0
ARG GH_SHA256=bb766f710eef8ede859c18578c72c327597cd4c8a85b06001b1f3843c6019386
ARG PNPM_VERSION=12.11.2
ARG PLAYWRIGHT_VERSION=1.64.0

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# uv, Node 24 (with npm and corepack), Docker CLI with the Compose and Buildx plugins
COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=docker /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker /usr/local/libexec/docker/cli-plugins /usr/local/libexec/docker/cli-plugins
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
 && ln -s ../lib/node_modules/corepack/dist/corepack.js /usr/local/bin/corepack

# just and gh from their release archives, verified by checksum
RUN cd /tmp \
 && curl -fsSLo just.tgz "https://github.com/casey/just/releases/download/${JUST_VERSION}/just-${JUST_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
 && echo "${JUST_SHA256}  just.tgz" | sha256sum -c - \
 && tar -xzf just.tgz just && install -m 0755 just /usr/local/bin/just \
 && curl -fsSLo gh.tgz "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_amd64.tar.gz" \
 && echo "${GH_SHA256}  gh.tgz" | sha256sum -c - \
 && tar -xzf gh.tgz && install -m 0755 "gh_${GH_VERSION}_linux_amd64/bin/gh" /usr/local/bin/gh \
 && rm -rf /tmp/*

# System libraries Chromium needs for the Playwright smoke tests (browsers
# themselves are downloaded by the web project into a cache volume).
RUN apt-get update \
 && npx --yes "playwright@${PLAYWRIGHT_VERSION}" install-deps chromium \
 && rm -rf /var/lib/apt/lists/* /root/.npm

# pnpm through corepack, pinned to the version in package.json#packageManager
ENV COREPACK_HOME=/usr/local/share/corepack \
    COREPACK_ENABLE_DOWNLOAD_PROMPT=0
# (pnpm 12 fetches its native binary on first run, so run it once here.)
RUN corepack enable pnpm && corepack install --global "pnpm@${PNPM_VERSION}" \
 && pnpm --version \
 && chmod -R a+rX "${COREPACK_HOME}"

# Non-root user. Folders that receive named volumes are created here so the
# volumes inherit the user's ownership.
RUN groupadd --gid "${USER_GID}" "${USERNAME}" \
 && useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home --shell /bin/bash "${USERNAME}" \
 && install -d -o "${USERNAME}" -g "${USERNAME}" \
      /workspaces/parlora/.venv /workspaces/parlora/node_modules \
      "/home/${USERNAME}/.cache/uv" "/home/${USERNAME}/.cache/pre-commit" \
      "/home/${USERNAME}/.cache/ms-playwright" "/home/${USERNAME}/.local/share/pnpm" \
 && chown -R "${USERNAME}:${USERNAME}" "/home/${USERNAME}/.cache" "/home/${USERNAME}/.local" \
 && git config --system --add safe.directory /workspaces/parlora

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PNPM_HOME="/home/${USERNAME}/.local/share/pnpm" \
    PLAYWRIGHT_BROWSERS_PATH="/home/${USERNAME}/.cache/ms-playwright"

USER ${USERNAME}
WORKDIR /workspaces/parlora
