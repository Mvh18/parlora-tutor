# ADR 0001 — Licence and contributor agreement

- **Status:** Accepted
- **Date:** 2026-10-28
- **Issue:** #1 (project skeleton epic)

> This ADR records an engineering and governance decision. It is not legal
> advice; the CLA text is flagged for legal review before the first external
> contribution is accepted.

## Context

Parlora Tutor is public on GitHub from its first commit. The copyright holder
wants three things at once:

1. anyone may use, study, modify and share it for non-commercial purposes;
2. third parties may not use it commercially without a separate licence;
3. the copyright holder keeps full freedom to use it commercially, including
   embedding it in their own course platform (Parlora Hub), with external
   contributions included.

An OSI-approved licence cannot meet the second goal, and accepting
contributions under the project licence alone would block the third.

## Decision

1. **Licence:** the verbatim PolyForm Noncommercial License 1.0.0 in
   `LICENSE`, with the `Required Notice: Copyright Marlon Vásquez Herrera`
   line. Public documentation describes the project as **source-available,
   free for non-commercial use**, and never as open source. A repository test
   enforces both.
2. **Commercial use** by anyone else needs a separate licence from the
   copyright holder, requested at team.parlora@gmail.com.
3. **The copyright holder is not a licensee.** PolyForm Noncommercial limits
   what others may do with the code; it does not restrict the copyright
   holder. Embedding Parlora Tutor in the holder's own platform, commercial or
   not, needs no exception.
4. **Contributions** are accepted under an individual Contributor Licence
   Agreement (`CLA.md`, modelled on the Apache ICLA). Contributors keep their
   copyright and grant the maintainer a broad licence that includes the right
   to sublicense and relicense, commercially too, and a patent licence.
5. **Enforcement:** the `cla.yml` workflow (an `actions/github-script` job on
   `pull_request_target`, without checking out pull request code) sets a
   `cla` commit status. A contributor accepts once by commenting the fixed
   phrase on a pull request; the comment is the record. The maintainer and
   bots are exempt. The status is not a required check, because it cannot
   always be written for Dependabot pull requests; the maintainer does not
   merge an external pull request while it fails.

## Consequences

- GitHub shows the licence as "Other", and some people and organisations
  avoid non-OSI projects. The README explains the licence in plain words.
- Contributions can be reused in the holder's commercial products, so the
  holder can embed Tutor in Parlora Hub with community contributions in it.
- The CLA adds friction: one comment on the first pull request.
- Third-party components keep their own licences (`NOTICE`); the project
  licence does not change them. This is why no GPL code may run in the
  Parlora Tutor process ([ADR 0004](0004-kokoro-g2p-and-gpl-boundary.md)).
- Signatures live in pull request comments in the repository itself, with no
  external service.

## Options ruled out

- **AGPL-3.0 with a commercial dual licence:** OSI-approved, but it permits
  commercial use under copyleft, which does not meet goal 2.
- **CC BY-NC-SA 4.0:** Creative Commons advises against it for software, and
  it has no patent grant.
- **Developer Certificate of Origin instead of a CLA:** gives no relicensing
  right, so contributions could not be reused commercially (goal 3).
- **CLA Assistant (hosted or the GitHub Action):** the hosted service stores
  signatures with a third party; the action was archived in March 2026 and
  runs on Node 20, which GitHub-hosted runners no longer provide.
