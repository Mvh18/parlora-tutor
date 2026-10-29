# ADR 0007 — Language as a first-class concept

- **Status:** Accepted
- **Date:** 2026-10-28
- **Issue:** #21 (domain core)

## Context

Tutor starts with English as the only language to practise, but it is meant
to support more, and learners need corrections explained in a language they
understand — for beginners, often their native language. If English were
hard-coded in prompts, validation or the API, every new language would be a
refactor.

## Decision

1. **`LanguageCode`** is a lowercase two-letter ISO 639-1 code. Region
   subtags are left for later: accent and voice are a speech-synthesis
   choice, not part of a language's identity.
2. **`Language`** has a code, an English name, a native name and two roles:
   `as_target` (a language learners can practise) and `as_feedback` (a
   language corrections can be explained in).
3. **`LanguageRegistry`** is the single, immutable list of supported
   languages. `default_registry()` ships English (target and feedback) and
   Spanish (feedback only). Adding a language means a registry entry plus its
   content (prompts, voice), not a code change.
4. **`SessionSettings`** validates its target and feedback languages against
   their roles in the registry.
5. **Default feedback language:** when the learner has not chosen one,
   beginners (A1–A2) get feedback in their native language if it is a
   feedback language; from B1 the feedback is in the target language.
6. **The interface language is a separate concept** from the target and
   feedback languages. Tutor's dev client is English only; localised
   interfaces belong to Parlora Hub.
7. **The API publishes the catalogue** (`GET /api/v1/languages`,
   `GET /api/v1/cefr-levels`) so clients never hard-code it.

## Consequences

- Code that needs a language receives one; there is no implicit English.
- Supporting a new language is mostly content work, and the API exposes it
  without a contract change.
- Codes such as `en-GB` are rejected until region subtags are needed.

## Options ruled out

- **Full BCP 47 tags from the start:** more validation and no benefit while
  accent is a voice setting.
- **English-only now, generalise later:** the assumption would spread into
  prompts, tests and the API and be costly to remove.
- **One language setting per session:** it cannot express "practise English,
  explain in Spanish".
