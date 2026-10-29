# Repository settings

Some GitHub settings cannot live in files. This page lists them with the
commands that apply them, so the repository can be audited and rebuilt. Run
the commands with the [GitHub CLI](https://cli.github.com/) authenticated as a
repository admin (`gh auth login`). They are idempotent unless noted.

```sh
REPO=Mvh18/parlora-tutor
```

## Security and analysis

**Dependency graph** must be enabled first: the `dependency review` job in
`security.yml` fails with "Dependency review is not supported on this
repository" without it, and Dependabot alerts depend on it. Enable it in
Settings → Code security → **Dependency graph** (there is no REST endpoint for
this toggle).

Secret scanning with push protection, Dependabot alerts and security updates,
and private vulnerability reporting (the channel `SECURITY.md` points to):

```sh
gh api -X PATCH "repos/$REPO" --input - <<'EOF'
{
  "security_and_analysis": {
    "secret_scanning": { "status": "enabled" },
    "secret_scanning_push_protection": { "status": "enabled" }
  }
}
EOF
gh api -X PUT "repos/$REPO/vulnerability-alerts"
gh api -X PUT "repos/$REPO/automated-security-fixes"
gh api -X PUT "repos/$REPO/private-vulnerability-reporting"
```

**Code scanning:** CodeQL runs from `.github/workflows/codeql.yml` (advanced
setup). Keep the CodeQL **default setup** disabled in Settings → Code security;
both cannot be active at once.

## Merge settings

Squash merge only, with the pull request title as the commit subject (it is
validated as a Conventional Commit) and the description as the body (so
`Closes #N` closes the issue). Merged branches are deleted.

```sh
gh api -X PATCH "repos/$REPO" --input - <<'EOF'
{
  "allow_squash_merge": true,
  "allow_merge_commit": false,
  "allow_rebase_merge": false,
  "squash_merge_commit_title": "PR_TITLE",
  "squash_merge_commit_message": "PR_BODY",
  "delete_branch_on_merge": true
}
EOF
```

## GitHub Actions

The default `GITHUB_TOKEN` is read-only and Actions cannot approve pull
requests; workflows request extra permissions per job.

```sh
gh api -X PUT "repos/$REPO/actions/permissions/workflow" \
  -f default_workflow_permissions=read \
  -F can_approve_pull_request_reviews=false
```

## Ruleset for `main`

[`.github/rulesets/main.json`](../.github/rulesets/main.json) protects the
default branch:

- changes arrive through pull requests, merged by squash only;
- the `ci-ok` check (which aggregates every CI and security job) must pass on
  an up-to-date branch;
- review conversations must be resolved;
- linear history; no force pushes; the branch cannot be deleted.

No approving review is required because the project has a single maintainer;
nobody can bypass the rules.

Create it once:

```sh
gh api -X POST "repos/$REPO/rulesets" --input .github/rulesets/main.json
```

Update it after editing the file:

```sh
ID=$(gh api "repos/$REPO/rulesets" --jq '.[] | select(.name == "main") | .id')
gh api -X PUT "repos/$REPO/rulesets/$ID" --input .github/rulesets/main.json
```

To disable it in an emergency: Settings → Rules → Rulesets → `main` →
Enforcement status → Disabled.

## Release and contribution workflows

- **Release token.** A pull request opened with the default `GITHUB_TOKEN`
  does not trigger other workflows, so the release PR would never get the
  required `ci-ok` check. Create a fine-grained personal access token limited
  to this repository with _Contents_ and _Pull requests_ read/write, and store
  it as the `RELEASE_PLEASE_TOKEN` Actions secret
  (`gh secret set RELEASE_PLEASE_TOKEN`). Without it, release-please still
  works, but its PR needs CI started by hand (close and reopen it).
- **Labels.** After `.github/labels.yml` first lands on `main`, the labels
  workflow creates them; run it by hand with
  `gh workflow run labels.yml` if needed.
- **CLA.** `.github/workflows/cla.yml` publishes a `cla` commit status. It is
  not a required check, because the runs of Dependabot pull requests cannot
  always write statuses; do not merge an external pull request while `cla`
  is failing.

## Account setting

Enable **Block command line pushes that expose my email** in the
maintainer's GitHub account (Settings → Emails), so a commit authored with a
private address is rejected on push.
