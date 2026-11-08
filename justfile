# Parlora task runner. Recipes are thin calls to uv, pnpm and docker so they
# behave the same in bash (Linux, Dev Container) and PowerShell (Windows).
# Anything non-trivial belongs in scripts/*.py, never in shell.

set shell := ["bash", "-euo", "pipefail", "-c"]
set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

# List the available recipes
default:
    @just --list --unsorted

# Install dependencies from the lockfiles (unchanged) and the git hooks
setup:
    uv sync --locked
    pnpm install --frozen-lockfile
    pnpm --filter @parlora/web exec playwright install chromium
    uv run pre-commit install --install-hooks

# Start the stack with the server reloading on code changes; `just dev nogpu` without an NVIDIA GPU
dev mode="gpu":
    @echo 'dev: server on http://127.0.0.1:8700 (reloads on change); run the dev client with `pnpm --dir apps/web dev` in a second terminal.'
    docker compose -f compose.yaml -f compose.dev.yaml {{ if mode == "nogpu" { "-f compose.nogpu.yaml" } else if mode == "gpu" { "" } else { error("mode must be gpu or nogpu") } }} up --build

# Start the Compose stack (PostgreSQL + server) and wait until healthy; `just up nogpu` without an NVIDIA GPU
up mode="gpu":
    docker compose -f compose.yaml {{ if mode == "nogpu" { "-f compose.nogpu.yaml" } else if mode == "gpu" { "" } else { error("mode must be gpu or nogpu") } }} up --detach --build --wait

# Stop the Compose stack (data volumes are kept)
down:
    docker compose --profile '*' down

# Format Python and everything Prettier handles
fmt:
    uv run ruff format .
    uv run ruff check --fix .
    pnpm exec prettier --write .

# Lint, check formatting and the import contracts without changing files
lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run lint-imports
    pnpm exec prettier --check .
    pnpm --filter @parlora/web lint

# Type-check Python (mypy --strict) and TypeScript (tsc)
typecheck:
    uv run mypy
    pnpm --filter @parlora/web typecheck

# Run every test suite
test: test-repo test-bench test-core test-server test-web

# Run the repository invariant tests, including the documentation and its relative links
test-repo:
    uv run pytest tests/repo

# Run the benchmark logic and report tests (no GPU, no Ollama) with a coverage gate (85 % lines and branches)
test-bench:
    uv run pytest tests/bench -m "not gpu" --cov=scripts.bench.common --cov=scripts.bench.report --cov-report=term-missing --cov-report=json:build/coverage-bench.json
    uv run python scripts/coverage_gate.py build/coverage-bench.json --lines 85 --branches 85

# Run the domain core tests with its coverage gate (95 % lines and branches)
test-core:
    uv run pytest packages/core/tests -m "not gpu" --cov=parlora_core --cov-report=term-missing --cov-report=json:build/coverage-core.json
    uv run python scripts/coverage_gate.py build/coverage-core.json --lines 95 --branches 95

# Run the server tests (unit, integration with Docker, contract) with its coverage gate (85 %)
test-server:
    uv run pytest apps/server/tests -m "not gpu" --cov=parlora_server --cov-report=term-missing --cov-report=json:build/coverage-server.json
    uv run python scripts/coverage_gate.py build/coverage-server.json --lines 85 --branches 85

# Run the web unit tests with its coverage gate (85 % on every metric)
test-web:
    pnpm --filter @parlora/web test:coverage

# Run the Playwright smoke tests against the production build (needs `just up` first)
e2e:
    pnpm --filter @parlora/web e2e

# Export the OpenAPI contract to contracts/openapi.json
openapi:
    uv run python scripts/export_openapi.py

# Generate the TypeScript API client from the OpenAPI contract
gen-client:
    pnpm --filter @parlora/web gen-client

# Install the optional bench group and check that the GPU and the host's Ollama answer
bench-setup:
    uv sync --group bench
    uv run --group bench python -m scripts.bench.host

# Benchmark the LLM through the host's Ollama (default: qwen3 4B instruct, gemma3:4b, qwen3:8b)
bench-llm *models:
    uv run --group bench python -m scripts.bench.llm {{ models }}

# Regenerate the synthetic speech fixtures with Kokoro in the bench image (espeak-ng for OOV words)
bench-fixtures:
    uv run python -m scripts.bench.container --writable fixtures/synthetic/stt scripts.bench.fixtures

# Benchmark speech-to-text natively (faster-whisper on the GPU)
bench-stt *args:
    uv run --group bench python -m scripts.bench.stt {{ args }}

# Benchmark speech-to-text in the bench image with the GPU (repository read-only)
bench-stt-container *args:
    uv run python -m scripts.bench.container --gpu --writable benchmarks/results scripts.bench.stt {{ args }}

# Benchmark Kokoro text-to-speech natively (`--provider cpu|cuda`; OOV words silent without espeak-ng)
bench-tts *args:
    uv run --group bench python -m scripts.bench.tts {{ args }}

# Benchmark Kokoro text-to-speech in the bench image (`--provider cpu|cuda`; OOV words through espeak-ng)
bench-tts-container *args:
    uv run python -m scripts.bench.container --writable benchmarks/results scripts.bench.tts {{ args }}

# Check that speech-to-text and each LLM fit on the GPU together (hard VRAM limit), and record RAM
bench-memory *models:
    uv run --group bench python -m scripts.bench.memory {{ models }}

# Regenerate docs/benchmarks.md from the latest result of each benchmark
bench-report:
    uv run python -m scripts.bench.report

# Run every benchmark in order, then regenerate the report (needs the GPU, Ollama and `just up`)
bench: bench-llm bench-stt bench-stt-container bench-tts bench-tts-container bench-memory bench-report

# Scan for secrets, audit workflows and check Python dependencies (ADR 0005)
security:
    uv run pre-commit run gitleaks --all-files
    uv run zizmor .
    uv run python scripts/audit_python_deps.py

# Run every gate CI runs (except jobs that need GitHub)
check: lint typecheck test security
