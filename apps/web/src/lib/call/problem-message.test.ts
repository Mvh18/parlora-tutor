import { describe, expect, it, vi } from "vitest";

import { problemMessage } from "./call-session";
import { browserCallDeps } from "./use-call";

describe("problemMessage", () => {
  it.each([
    [
      { type: "/problems/voice-unavailable", unavailable: ["stt", "tts"] },
      /speech recognition, speech synthesis/,
    ],
    [{ type: "/problems/voice-unavailable", unavailable: ["gpu"] }, /gpu/],
    [{ type: "/problems/voice-unavailable" }, /Calls are unavailable right now\.$/],
    [{ type: "/problems/call-limit" }, /already in progress/],
    [{ status: 422 }, /settings/],
    [{ status: 500 }, /could not be started/],
    [undefined, /could not be started/],
  ])("explains %j", (problem, expected) => {
    expect(problemMessage(problem)).toMatch(expected);
  });
});

describe("browserCallDeps", () => {
  it("uses the browser's microphone and peer connection", () => {
    const connection = vi.fn();
    vi.stubGlobal("RTCPeerConnection", connection);
    vi.stubGlobal("navigator", { mediaDevices: { getUserMedia: vi.fn() } });

    const deps = browserCallDeps();
    deps.createPeerConnection();

    expect(connection).toHaveBeenCalledOnce();
    expect(deps.mediaDevices).toBe(navigator.mediaDevices);
    vi.unstubAllGlobals();
  });
});
