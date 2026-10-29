# Security policy

## Supported versions

Parlora is pre-1.0. Only the latest release and the `main` branch receive
security fixes.

| Version        | Supported |
| -------------- | --------- |
| `main`         | Yes       |
| Latest release | Yes       |
| Older releases | No        |

## Reporting a vulnerability

Please **do not open a public issue, discussion or pull request** for a
security problem.

Report it privately through GitHub's **private vulnerability reporting**:
open the repository's **Security** tab and choose **Report a vulnerability**
(direct link: <https://github.com/Mvh18/parlora-tutor/security/advisories/new>).

Include, where you can:

- the affected component (domain core, server API, web app, Docker setup,
  workflows) and version or commit;
- steps to reproduce or a proof of concept;
- the impact you expect (for example, data exposure or remote access).

Parlora is maintained by one person. You can expect an acknowledgement within
7 days and a status update at least every 14 days until the report is
resolved. Fixes are released as GitHub security advisories with credit to the
reporter unless you prefer otherwise.

## Scope

Parlora runs locally and handles voice audio and transcripts, which are
personal data. In scope:

- anything that sends audio, transcripts or learner data off the machine
  without the operator configuring it;
- network exposure beyond `127.0.0.1` in the default setup;
- authentication or authorisation flaws in the API;
- injection, path traversal or unsafe deserialisation in the server;
- supply-chain issues in this repository: workflows, pinned actions,
  dependency policy, container images built from it.

Out of scope:

- vulnerabilities in third-party models, runtimes or libraries that are not
  caused by how Parlora uses them (report those upstream);
- setups where the operator deliberately exposed ports to a network;
- findings that require an already compromised machine.
