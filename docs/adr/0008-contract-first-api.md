# ADR 0008 — Contract-first API

- **Status:** Accepted
- **Date:** 2026-10-28
- **Issue:** #22 (versioned API)

## Context

Tutor is consumed by clients it does not control: its own dev client today,
and Parlora Hub — a separate Laravel application in another repository — from
Phase 7. Hub must never import Tutor code or read Tutor's database, so the
API is the only integration surface. It has to be explicit, versioned and
checked, or the two systems drift apart unnoticed.

## Decision

1. **Versioned paths.** Every route lives under `/api/v1`. A breaking change
   needs a new version; additive changes (new routes, new optional fields,
   new keys in open maps such as the readiness `checks`) do not.
2. **The contract is a committed file.** `just openapi` exports the server's
   OpenAPI document to `contracts/openapi.json`. Its `info.version` is the
   package version, which release-please bumps. A contract test fails, with a
   diff, when the committed file differs from the generated one.
3. **Clients are generated from the contract.** The dev client's TypeScript
   types come from `openapi-typescript` (`just gen-client`), committed and
   checked for drift in CI. Hub generates its client from the same file and
   keeps a versioned copy of it.
4. **One error format.** Every error is an RFC 9457 problem document
   (`application/problem+json` with `type`, `title`, `status`, `detail`,
   `instance`), never a stack trace or a settings value.
5. **Order of change.** A contract change lands in Tutor first, then in Hub.
6. **Voice messages join the contract.** The RTVI message schema for the
   voice channel is published under `contracts/` when the integration
   arrives, so Hub's call screen is built against a documented schema.

## Consequences

- Every API change is visible in review as a diff of `contracts/openapi.json`
  and the generated schema.
- Contributors must run `just openapi` and `just gen-client` after changing
  the API; CI fails otherwise.
- Clients can rely on problem documents and must accept unknown keys in open
  maps.

## Options ruled out

- **GraphQL or tRPC:** no benefit for a small, integration-oriented API, and
  tRPC requires a TypeScript server.
- **Unversioned paths:** the first breaking change would break every
  integrator at once.
- **Hand-written client types:** they drift from the server silently.
