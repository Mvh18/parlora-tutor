# ADR 0003 — WebRTC transport from Docker

- **Status:** Accepted, amended 2026-11-15 (see [Amendment](#amendment-2026-11-15))
- **Date:** 2026-10-17
- **Issue:** #4 (spike, time-boxed to 1–2 days)

## Context

The voice pipeline (Phase 1) will use Pipecat's SmallWebRTC transport between
the browser and the server. The server is planned to run in a container on
Docker Desktop (Windows, WSL2 backend). Inside the container, WebRTC ICE
candidates carry container-internal addresses, which the browser on the
Windows host may not reach. This was the largest risk of the "server in
Docker" decision. If no setup worked, the voice process would have to run
natively.

## Questions

1. Can a browser on the Windows host exchange audio both ways with a Pipecat
   SmallWebRTC echo bot running in a container?
2. Which setup works: a fixed UDP port range published on `127.0.0.1`, Docker
   Desktop host networking, or an explicitly configured host ICE candidate?
3. How much latency does the container add compared with a native process?
4. Does it still work from a phone on the local network (Phase 4) without
   exposing more than needed?

## Setup

| Item           | Value                                                                                               |
| -------------- | --------------------------------------------------------------------------------------------------- |
| Host           | Windows 11 Pro, Docker Desktop (engine 28.2.2, WSL2 backend), network profile _Private_             |
| Image          | `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`, non-root user                                      |
| Packages       | pipecat-ai 0.0.108 (`webrtc` extra), aiortc 1.15.0, aioice 0.10.2                                   |
| Bot            | `transport.input() → echo → transport.output()` at 16 kHz; no speech-to-text, LLM or text-to-speech |
| Desktop client | Chrome 155, headless, fake media device; a 1 kHz tone generated with WebAudio                       |
| Phone client   | iPhone, Chrome for iOS (WebKit engine), same Wi-Fi network                                          |
| Signalling     | Plain HTTP `POST /api/offer` (non-trickle ICE), no STUN or TURN servers                             |

The test page sends short tones and measures how long each takes to come back
(end-to-end echo latency). It also reads `getStats()` (selected candidate pair,
round-trip time, loss, jitter buffer). No real voice was used or recorded. The
spike code stayed on a throwaway branch and is not merged.

## Findings

### 1. Desktop browser to container — yes, with no UDP port published

With the default bridge network and only the HTTP port published on
`127.0.0.1`, audio flows both ways (6 of 6 sessions).

It works because of how ICE runs, not because of a port mapping:

- The container's only candidate is its bridge address, which the host cannot
  reach.
- The browser's offer lists the host's real interface addresses, including
  the Hyper-V/WSL virtual adapter.
- The container sends its ICE checks _out_ to those addresses through Docker's
  NAT. The browser learns the container as a peer-reflexive candidate, and the
  pair succeeds.

When the browser's candidates are removed from the offer, so that only the
browser can start the checks, the same setup fails. The outbound path from
the container is what makes it work.

This depends on the browser revealing real host addresses. Browsers do that
only after the page has microphone permission; otherwise they send mDNS names
(`<uuid>.local`), which the container cannot resolve. A voice app always asks
for the microphone, but it **must do so before creating the WebRTC offer**.

### 2. Which setup works

| Setup                                                                                    | Result                                                                                                                                                                                                  |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Default bridge, no UDP published                                                         | ✅ Works on the desktop through container-initiated checks (above). Nothing is exposed.                                                                                                                 |
| Fixed UDP range published on `127.0.0.1`, candidate rewritten to `127.0.0.1`             | ⚠️ Connects, but only through the path above. On its own (browser-initiated only) ICE **fails**: Chrome does not complete checks against a loopback candidate.                                          |
| Fixed UDP range published on the host's LAN address, candidate rewritten to that address | ✅ Works on its own, including browser-initiated only.                                                                                                                                                  |
| Docker Desktop host networking (`--network host`)                                        | ❌ Not usable by default. The container gets the Docker VM's network, and not even its HTTP port is reachable from Windows unless host networking is turned on in Docker Desktop settings. Not pursued. |

Two limits in the current libraries:

- **aioice cannot restrict ICE to a port range.** It always binds port 0
  (random). Publishing a fixed range needs a small patch. The spike wraps the
  event loop's `create_datagram_endpoint` so that host candidates bind inside
  the range. Rewriting the candidate address in the SDP answer is a simple
  text substitution.
- **Pipecat's SmallWebRTC transport always imports OpenCV**, even for audio
  only. The default `opencv-python` wheel needs X11 libraries that the slim
  image does not have, so the container fails at import. Replacing it with
  `opencv-python-headless` fixes this. The image was 540 MB.

### 3. Latency

| Session                                          | Connect    | ICE round trip | Echo latency, median (range) | Loss |
| ------------------------------------------------ | ---------- | -------------- | ---------------------------- | ---- |
| Desktop, bridge, 8 tones                         | 533 ms     | 1 ms           | 490 ms (372–518)             | 0    |
| Desktop, bridge, 25 tones                        | 266 ms     | 2 ms           | 508 ms (416–567)             | 0    |
| Desktop, bridge, 8 tones                         | 295 ms     | 2 ms           | 276 ms                       | 0    |
| Desktop, fixed range (3 sessions, 15 tones)      | 282–288 ms | 2 ms           | 268–321 ms                   | 0    |
| Desktop, LAN address only (2 sessions, 10 tones) | 220–233 ms | 2–3 ms         | 255–291 ms                   | 0    |
| Phone over Wi-Fi, LAN address                    | 1,252 ms   | 5 ms           | 304 ms (264–395)             | 0    |

- The network path through Docker adds very little: ICE round trips are
  1–3 ms on the desktop and 5 ms from the phone, with no packet loss.
- End-to-end echo latency (255–508 ms median) is dominated by audio buffering
  in the pipeline and the browser (Pipecat's input and output chunking,
  resampling, the Opus codec, and a 22–93 ms jitter buffer), not by the
  network. It varies a lot between sessions, but does not keep growing within
  a session.

**Native comparison: not measured.** As in ADR 0002, the project keeps the
toolchain off the host, so there was no native process to compare against.
Given the 1–3 ms network round trip, the container is unlikely to be the main
cost, but this is unverified. The comparison moves to
`add-component-benchmarks`, together with tuning the pipeline's buffering.

### 4. Phone on the local network — yes, with three explicit opt-ins

The phone connected and audio flowed both ways (above). It needed three
things, and each is an exposure the project must make opt-in:

1. **A fixed UDP range published on the host's LAN address, advertised as the
   ICE candidate.** The phone only sends an mDNS candidate, so the
   container-initiated path from finding 1 cannot work. The browser must
   start the checks against a reachable published address.
2. **A Windows firewall rule.** Docker Desktop's own allow rules for
   `com.docker.backend` covered only the _Public_ profile. On a _Private_
   network, inbound LAN traffic to published ports was blocked until a rule
   was added for TCP (HTTP) and the UDP range, scoped to `LocalSubnet` and
   the _Private_ profile. The rule was removed after the test.
3. **HTTPS for the microphone.** Over plain HTTP on a LAN address the page is
   not a secure context, so browsers do not expose `getUserMedia`. The spike
   used a generated tone instead. Real use from a phone needs HTTPS
   (for example, a locally trusted certificate).

## Decision

**Keep the voice server in Docker** (design decision D12). The native
fallback is not needed.

- **Phase 1 (desktop, same machine):** default bridge network. Publish only
  the HTTP port on `127.0.0.1`. **Publish no UDP ports.** ICE works through
  container-initiated checks. The web client must get microphone permission
  before it creates the offer.
- **Phase 4 (phone on the LAN), opt-in only and documented:**
  - a fixed ICE UDP port range, through a small, tested patch around aioice
    (or an upstream option, if one appears);
  - that range published on the host's LAN address and advertised as the
    candidate;
  - a firewall rule scoped to the local subnet and _Private_ profile;
  - HTTPS.
    This is off by default; nothing listens on the LAN unless the user turns it
    on.
- **Image:** replace `opencv-python` with `opencv-python-headless` (or drop
  OpenCV if a later Pipecat version stops importing it for audio-only use).

## Consequences

- The local-runtime work and ROADMAP Phase 1 go ahead with the server in
  Docker. The rule "all published ports bind to `127.0.0.1`" still holds for
  the default stack.
- The web client gets one hard requirement: request the microphone first,
  then negotiate WebRTC. A test should cover the order.
- Desktop connectivity relies on the container reaching the host's interface
  addresses through Docker's NAT. This held on the target machine, but it is
  not a documented Docker guarantee. If it breaks on another setup, the
  Phase 4 configuration with a fixed range is the fallback, and it also
  works locally.
- Phase 4 carries real security work: LAN exposure, a firewall rule and
  certificates. It must be designed as an explicit opt-in, not as a default.
- End-to-end echo latency of about 250–500 ms, before any speech-to-text, LLM
  or text-to-speech, is already a large share of a conversational turn.
  Measuring and reducing pipeline buffering is added to
  `add-component-benchmarks`.

## Options ruled out

- **Voice process native, API and database in Docker.** Not needed: the
  container path works, and a native process would break the Dev Container
  decision.
- **Advertising `127.0.0.1` as the ICE candidate.** Chrome does not complete
  checks against it.
- **Docker Desktop host networking.** Off by default, needs a per-machine
  setting, and adds nothing over the bridge setup.
- **A TURN server in the Compose stack.** Would also solve reachability, but
  adds a service and credentials for a problem the bridge setup already
  solves on the desktop. Kept in reserve for Phase 4 if the fixed-range patch
  proves brittle.

## Amendment (2026-11-15)

**Finding 1 no longer held on the target machine.** With the Phase 1 voice
loop running (Docker Desktop 4.42, Docker Engine 28.2), desktop calls did not
connect: the server accepted the offer and the browser's candidates (real host
addresses, since the microphone was granted first), the container's STUN
checks got no answer, and the browser closed the call after about 14 s. Chrome
and Playwright's Chromium behaved the same, and a firewall rule allowing all
inbound traffic to Playwright's binary did not help.

A direct probe isolated it: UDP sent from the container to the host's LAN
address or its Hyper-V adapters never arrives, even at a socket bound to all
interfaces; UDP sent to `host.docker.internal` arrives, but over loopback, which
a socket bound to an interface address (as Chrome binds its candidates) never
sees. The container-initiated path that finding 1 relied on is gone. What
changed since the spike was not established.

**Decision.** The fallback this ADR already measured — a fixed UDP range
published on the host's LAN address, advertised as the ICE candidate — becomes
an opt-in for desktop calls, ahead of Phase 4 (design D11 of `add-voice-loop`):

- `PARLORA_VOICE__ICE_HOST` (unset by default) and
  `PARLORA_VOICE__ICE_PORT_MIN`/`_MAX` (40000–40019).
- When set, aioice's host sockets bind inside the range and the answer's host
  candidates carry `ICE_HOST`. The patch swaps aioice's `asyncio` name for a
  shim, so no other code in the process sees a changed event loop.
- `compose.ice.yaml` publishes the range as UDP on `${PARLORA_VOICE__ICE_HOST}`
  and refuses to render without it; `just up gpu lan` adds it.

**Consequences.**

- The default stack is unchanged and still publishes only the HTTP port on
  `127.0.0.1`. The opt-in is the single documented exception to that rule.
- With the opt-in, 20 UDP ports are reachable from the local network while the
  stack runs. Each answers ICE only for a live call with credentials exchanged
  over the loopback-only API, and calls are capped by `PARLORA_VOICE__MAX_CALLS`.
- Phase 4 (phone on the LAN) builds on this rather than introducing it; it still
  needs the firewall rule and HTTPS from finding 4.
