# ADR 0006 — Hexagonal domain core

- **Status:** Accepted
- **Date:** 2026-10-28
- **Issue:** #21 (domain core)

## Context

Tutor's rules — which languages a session may use, how CEFR levels compare,
what a learner reference may contain, which language feedback is given in —
must outlive the technology around them. The voice stack (Pipecat,
faster-whisper, Ollama, Kokoro), the web framework and the database will
change across phases, and future rules such as evaluation scoring must be
testable without a GPU, a server or a database.

## Decision

1. **`parlora-core` depends only on the Python standard library.** Its value
   objects are frozen dataclasses that validate in `__post_init__` and raise
   subclasses of `DomainError`.
2. **Ports are `typing.Protocol`s** in `parlora_core.ports`: `SpeechToText`,
   `LanguageModel`, `TextToSpeech` and `SessionRepository`. They are
   provisional until the voice loop (Phase 1) settles their shape.
3. **Adapters live in the server** (`parlora_server.adapters`), never in the
   core.
4. **The boundary is enforced, not documented only.** import-linter forbids
   `parlora_core` from importing `fastapi`, `starlette`, `pydantic`,
   `sqlalchemy`, `pipecat` or `parlora_server`, and enforces the server's
   layers `main` → `api` → `db` → `config`. Both run in `just lint` and CI.
5. **The core carries the strictest coverage gate:** 95 % of lines and
   branches, with property-based tests (Hypothesis) for validation and
   ordering rules. It also runs on Windows in CI.

## Consequences

- Domain logic is fast to test and runs anywhere Python runs.
- The server translates between Pydantic models and core value objects; this
  mapping is a small, deliberate duplication.
- A new speech, model or storage technology is a new adapter, with no change
  to the core.
- Ports may change in Phase 1 without a compatibility promise.

## Options ruled out

- **Pydantic models in the core:** convenient, but it couples the domain to a
  framework version, and integrators may not use Pydantic.
- **No separate core package:** rules would drift into route handlers and
  pipeline code, where they are hard to test.
- **Architecture rules by convention only:** they erode silently; a failing
  check does not.
