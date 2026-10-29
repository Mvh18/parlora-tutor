# CLAUDE.md

@AGENTS.md

## Notes for Claude Code

- Specs live in OpenSpec under `openspec/` and the project skills under
  `.claude/`. Both folders are git-ignored on purpose: the public specification
  of a change is its GitHub issue, so issue bodies must stay self-contained.
- Use the OpenSpec skills to propose, apply and archive changes, and
  `/release-plan --from-spec <change>` to turn a change into issue cards and
  branch plans under `.claude/plans/`.
- Machine-specific notes (hardware, local paths, accounts) go in
  `CLAUDE.local.md`, which is git-ignored. Never copy them into tracked files.
