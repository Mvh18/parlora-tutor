# AGENTS.md

Instructions for AI coding agents and human contributors working in this
repository. [CONTRIBUTING.md](CONTRIBUTING.md) is the human-oriented version of
the workflow; this file is the precise reference. Machine-specific notes
(hardware, local paths) belong in `CLAUDE.local.md` or another git-ignored
file, never here.

## What this repository is

Parlora Tutor is the local voice-AI engine of Parlora: conversation,
corrections and speaking evaluation, exposed through a versioned API. It has
no product UI. Parlora Hub (a separate repository) is the learner-facing
product and consumes Tutor only through its API; `apps/web` is a minimal dev
client for development, testing and demos. Every committed artefact is in
English, except `README.es.md`.

## Stack

| Area    | Choice                                                                                |
| ------- | ------------------------------------------------------------------------------------- |
| Python  | 3.12, uv workspace (`packages/core`, `apps/server`), one `uv.lock`                    |
| Server  | FastAPI, pydantic-settings (`PARLORA_` prefix), SQLAlchemy 2 async + asyncpg, Alembic |
| Web     | Vite + React, TypeScript strict, plain CSS, openapi-typescript + openapi-fetch, pnpm  |
| Runtime | Docker Compose: PostgreSQL + server (GPU optional), Ollama on the host                |
| Tests   | pytest, Hypothesis, Testcontainers; Vitest, Testing Library, MSW; Playwright + axe    |
| Static  | ruff, mypy `--strict`, import-linter, ESLint, Prettier, `tsc`                         |
| Tasks   | just (from the uv dev group), PowerShell on Windows and bash elsewhere                |
| Release | Conventional Commits, squash merges, release-please                                   |

## Commands

Run everything from the repository root through uv, so the pinned tools are
used: `uv run just <recipe>`.

| Command            | Use                                                                 |
| ------------------ | ------------------------------------------------------------------- |
| `just setup`       | Install from the lockfiles (unchanged) and install the git hooks    |
| `just check`       | Every gate CI runs; run it before you say a change is done          |
| `just fmt`         | Format Python and Prettier files                                    |
| `just lint`        | ruff, format check, import contracts, Prettier, ESLint              |
| `just typecheck`   | mypy `--strict` and `tsc`                                           |
| `just test-core`   | Domain core tests, coverage gate 95 % lines and branches            |
| `just test-server` | Server tests (integration tests start PostgreSQL in Docker), 85 %   |
| `just test-web`    | Web unit tests, 85 % on every metric                                |
| `just test-repo`   | Repository invariants, documentation and relative links             |
| `just up` / `down` | Start / stop the local stack; `just up nogpu` without an NVIDIA GPU |
| `just e2e`         | Playwright smoke against the running stack                          |
| `just openapi`     | Export `contracts/openapi.json`                                     |
| `just gen-client`  | Regenerate `apps/web/src/lib/api/schema.d.ts` from the contract     |
| `just security`    | gitleaks, zizmor, `pip-audit` and the licence check                 |

Recipes stay thin calls to `uv`, `pnpm` and `docker compose` that work in both
shells. Any logic beyond that goes into a Python script under `scripts/`.

## Architecture rules

- **The domain core is framework-free.** `parlora_core` imports only the
  standard library. import-linter forbids `fastapi`, `starlette`, `pydantic`,
  `sqlalchemy`, `pipecat` and `parlora_server` there. Value objects are frozen
  dataclasses that validate in `__post_init__`; errors subclass `DomainError`.
- **Ports are `typing.Protocol`s** in `parlora_core.ports`. Adapters that talk
  to models, the database or the network live in `parlora_server.adapters`.
- **The server is layered:** `main` → `api` → `db` → `config`. A layer may
  import the ones below it, never above (enforced by import-linter).
- **Language is a first-class concept.** Never hard-code English: target and
  feedback languages come from the `LanguageRegistry`, and adding a language
  is a registry entry plus content.
- **Learners are opaque references.** A `LearnerRef` is chosen by the
  integrating platform and is never an email address. Tutor stores no
  identity data and no users table.
- **The API is contract-first.** All routes live under `/api/v1`. Errors are
  RFC 9457 problem documents (`application/problem+json`) with no stack traces
  or settings. After any API change, run `just openapi` and `just gen-client`
  and commit both files; the contract tests fail on drift. A breaking change
  needs a new API version.
- **Each system owns its data.** Tutor never reads another system's database
  and nothing reads Tutor's. Integration features are optional and stay off
  unless configured; Tutor must keep working stand-alone.
- **Settings** come from `PARLORA_*` environment variables (nested groups use
  `__`). Every setting is documented in `.env.example`; secrets are
  `SecretStr`.

See [docs/architecture.md](docs/architecture.md) and the decision records in
[docs/adr/](docs/adr/).

## Testing rules

- Write the failing test first for domain logic, then the code.
- Every change ships with its tests. Coverage gates: core 95 % lines and
  branches, server 85 %, web 85 % on every metric. Do not lower a gate or
  exclude code to pass it.
- Mark tests with `unit`, `integration`, `contract`, `gpu` or `slow`
  (markers are strict). `gpu` tests never run in CI.
- Use Hypothesis for validation rules and ordering, Testcontainers for
  PostgreSQL, MSW for the web client's HTTP calls.
- Repository-wide rules (ignore policy, licence, documentation, workflows)
  are pytest tests in `tests/repo/`; when you add such a rule, add its test.

## Security and privacy rules

- Never commit real voice recordings, transcripts or personal data. Test
  audio is synthetic and lives under a `fixtures/synthetic/` folder.
- Never commit secrets, `.env` files, model weights (`*.gguf`,
  `*.safetensors`, `*.onnx`, `*.pt`, `*.bin`) or local absolute paths.
- No GPL or AGPL code in the Parlora process. LGPL is allowed only for the
  reviewed packages listed in ADR 0004; an unknown licence fails CI. GPL
  tools such as `espeak-ng` run only as separate executables.
- Published container ports bind to `127.0.0.1`.
- GitHub Actions: pin every third-party action to a full commit SHA with a
  version comment, keep `permissions` minimal, use
  `persist-credentials: false`, and do not add `pull_request_target`
  workflows. `just security` runs zizmor on them.
- Report vulnerabilities through private vulnerability reporting
  ([SECURITY.md](SECURITY.md)), never in a public issue.

## Workflow

Development is spec-driven:

1. **Spec.** Larger work starts as a spec (proposal, design, scenarios,
   tasks). The GitHub issue carries the public version of it — context,
   scope and Gherkin acceptance criteria — and is the source of truth for
   the change.
2. **Branch** from `main` as `<type>/<issue>-<slug>`, e.g. `feat/21-domain-core`.
3. **Implement** test-first, in small commits that follow Conventional
   Commits: `<type>(<scope>): <description>`, imperative and lowercase, with
   a body that explains why and a `Refs #N` or `Closes #N` footer. Scopes:
   `core`, `server`, `web`, `infra`, `ci`, `docs`, `deps`, `repo`.
4. **Verify** with `uv run just check`, and `just up` + `just e2e` when the
   runtime or web client changed.
5. **Pull request** with the template; the title follows Conventional Commits
   because pull requests are squash-merged. The required check is `ci-ok`.
6. **Release:** release-please derives versions and `CHANGELOG.md` from the
   commits on `main`. Never bump versions or edit the changelog by hand.

Decisions that matter to contributors are recorded as ADRs in `docs/adr/`
(Context, Decision, Consequences). Update the README, this file or an ADR in
the same pull request when behaviour or workflow changes.
