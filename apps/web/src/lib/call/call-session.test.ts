import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";

import { API_BASE_URL, createApiClient } from "../api/client";
import { server } from "../../test/server";
import { CallError, startCall, type CallHandlers } from "./call-session";
import { FakePeerConnection, fakeMediaDevices } from "./fakes";

const OFFER_URL = `${API_BASE_URL}/api/v1/calls/offer`;
const SETTINGS = { level: "B1", topic: "travel" } as const;

const answer = { call_id: "call-1", pc_id: "call-1", sdp: "v=0 answer", type: "answer" };

function problem(status: number, type: string, extra: Record<string, unknown> = {}) {
  return HttpResponse.json(
    { type, title: "Problem", status, detail: "", instance: "/api/v1/calls/offer", ...extra },
    { status, headers: { "Content-Type": "application/problem+json" } },
  );
}

function setup(handlers: Partial<CallHandlers> = {}) {
  const pc = new FakePeerConnection();
  const media = fakeMediaDevices();
  const all: CallHandlers = {
    onConnected: vi.fn(),
    onEnded: vi.fn(),
    onRemoteStream: vi.fn(),
    onLatency: vi.fn(),
    ...handlers,
  };
  const deps = {
    client: createApiClient(),
    mediaDevices: media,
    createPeerConnection: () => pc as unknown as RTCPeerConnection,
  };
  return { pc, media, handlers: all, deps };
}

describe("startCall", () => {
  it("asks for the microphone before it sends any offer", async () => {
    const order: string[] = [];
    const { deps, handlers, media } = setup();
    media.getUserMedia.mockImplementation(() => {
      order.push("microphone");
      return Promise.resolve(media.stream);
    });
    server.use(
      http.post(OFFER_URL, () => {
        order.push("offer");
        return HttpResponse.json(answer);
      }),
    );

    await startCall(deps, SETTINGS, handlers);

    expect(order).toEqual(["microphone", "offer"]);
  });

  it("sends the offer in the transport's shape with the settings under requestData", async () => {
    const { deps, handlers } = setup();
    let body: unknown;
    server.use(
      http.post(OFFER_URL, async ({ request }) => {
        body = await request.json();
        return HttpResponse.json(answer);
      }),
    );

    await startCall(deps, SETTINGS, handlers);

    expect(body).toEqual({
      sdp: "v=0 fake offer",
      type: "offer",
      pc_id: null,
      restart_pc: false,
      requestData: { target_language: "en", level: "B1", topic: "travel" },
    });
  });

  it("applies the answer and reports the connection once it is up", async () => {
    const { deps, handlers, pc } = setup();
    server.use(http.post(OFFER_URL, () => HttpResponse.json(answer)));

    await startCall(deps, SETTINGS, handlers);
    expect(pc.remoteDescription?.sdp).toBe("v=0 answer");
    expect(handlers.onConnected).not.toHaveBeenCalled();

    pc.setConnectionState("connected");
    expect(handlers.onConnected).toHaveBeenCalledOnce();
  });

  it("sends candidates by pc_id: queued ones after the answer, later ones as they come", async () => {
    const { deps, handlers, pc } = setup();
    const sent: unknown[] = [];
    server.use(
      http.post(OFFER_URL, () => HttpResponse.json(answer)),
      http.patch(OFFER_URL, async ({ request }) => {
        sent.push(await request.json());
        return new HttpResponse(null, { status: 204 });
      }),
    );
    const call = startCall(deps, SETTINGS, handlers);
    await vi.waitFor(() => {
      expect(pc.localDescription).not.toBeNull();
    });
    pc.emitCandidate("candidate:early", "0", 0);

    await call;
    pc.emitCandidate("candidate:late", "0", 0);
    pc.emitCandidate(null);

    await vi.waitFor(() => {
      expect(sent).toHaveLength(2);
    });
    // The two requests are concurrent, so their arrival order is not fixed.
    expect(sent).toEqual(
      expect.arrayContaining([
        {
          pc_id: "call-1",
          candidates: [{ candidate: "candidate:early", sdp_mid: "0", sdp_mline_index: 0 }],
        },
        {
          pc_id: "call-1",
          candidates: [{ candidate: "candidate:late", sdp_mid: "0", sdp_mline_index: 0 }],
        },
      ]),
    );
  });

  it("sends RTVI client-ready when the chat channel opens", async () => {
    const { deps, handlers, pc } = setup();
    server.use(http.post(OFFER_URL, () => HttpResponse.json(answer)));

    await startCall(deps, SETTINGS, handlers);
    expect(pc.channel.label).toBe("chat");
    pc.channel.open();

    expect(pc.channel.sent).toHaveLength(1);
    expect(JSON.parse(pc.channel.sent[0] as string)).toMatchObject({
      label: "rtvi-ai",
      type: "client-ready",
    });
  });

  it("reports the latency from a turn-latency server message", async () => {
    const { deps, handlers, pc } = setup();
    server.use(http.post(OFFER_URL, () => HttpResponse.json(answer)));
    await startCall(deps, SETTINGS, handlers);

    pc.channel.receive({
      label: "rtvi-ai",
      type: "server-message",
      data: { type: "turn-latency", voice_to_voice_ms: 900, interrupted: false },
    });
    pc.channel.receive({ label: "rtvi-ai", type: "bot-ready", data: {} });
    pc.channel.receive("not json");
    pc.channel.receive({
      label: "rtvi-ai",
      type: "server-message",
      data: { type: "turn-latency", voice_to_voice_ms: null },
    });

    expect(handlers.onLatency).toHaveBeenCalledOnce();
    expect(handlers.onLatency).toHaveBeenCalledWith(900);
  });

  it("plays the tutor's audio track", async () => {
    const { deps, handlers, pc } = setup();
    server.use(http.post(OFFER_URL, () => HttpResponse.json(answer)));
    await startCall(deps, SETTINGS, handlers);

    const stream = {} as MediaStream;
    pc.emitTrack(stream);

    expect(handlers.onRemoteStream).toHaveBeenCalledWith(stream);
  });

  it("ends the call: DELETE, close the connection, release the microphone, once", async () => {
    const { deps, handlers, pc, media } = setup();
    const deleted = vi.fn();
    server.use(
      http.post(OFFER_URL, () => HttpResponse.json(answer)),
      http.delete(`${API_BASE_URL}/api/v1/calls/call-1`, () => {
        deleted();
        return new HttpResponse(null, { status: 204 });
      }),
    );
    const call = await startCall(deps, SETTINGS, handlers);

    await call.end();
    await call.end();

    expect(deleted).toHaveBeenCalledOnce();
    expect(pc.closed).toBe(true);
    expect(media.track.stop).toHaveBeenCalled();
    expect(handlers.onEnded).toHaveBeenCalledOnce();
  });

  it("reports the end when the connection fails", async () => {
    const { deps, handlers, pc, media } = setup();
    server.use(
      http.post(OFFER_URL, () => HttpResponse.json(answer)),
      http.delete(
        `${API_BASE_URL}/api/v1/calls/call-1`,
        () => new HttpResponse(null, { status: 204 }),
      ),
    );
    await startCall(deps, SETTINGS, handlers);

    pc.setConnectionState("failed");

    await vi.waitFor(() => {
      expect(handlers.onEnded).toHaveBeenCalledOnce();
    });
    expect(media.track.stop).toHaveBeenCalled();
  });

  describe("problems", () => {
    it("names the language model when the voice is unavailable", async () => {
      const { deps, handlers, media, pc } = setup();
      server.use(
        http.post(OFFER_URL, () =>
          problem(503, "/problems/voice-unavailable", { unavailable: ["llm"] }),
        ),
      );

      const error = await startCall(deps, SETTINGS, handlers).catch((e: unknown) => e);

      expect(error).toBeInstanceOf(CallError);
      expect((error as CallError).message).toMatch(/unavailable/i);
      expect((error as CallError).message).toMatch(/language model/i);
      expect(media.track.stop).toHaveBeenCalled();
      expect(pc.closed).toBe(true);
    });

    it("explains the call limit", async () => {
      const { deps, handlers } = setup();
      server.use(http.post(OFFER_URL, () => problem(409, "/problems/call-limit")));

      await expect(startCall(deps, SETTINGS, handlers)).rejects.toThrow(/already in progress/i);
    });

    it("explains invalid settings", async () => {
      const { deps, handlers } = setup();
      server.use(http.post(OFFER_URL, () => problem(422, "about:blank")));

      await expect(startCall(deps, SETTINGS, handlers)).rejects.toThrow(/settings/i);
    });

    it("falls back to a generic message for other problems", async () => {
      const { deps, handlers } = setup();
      server.use(http.post(OFFER_URL, () => problem(500, "about:blank")));

      await expect(startCall(deps, SETTINGS, handlers)).rejects.toThrow(/could not be started/i);
    });

    it("says the server is unreachable on a network error", async () => {
      const { deps, handlers } = setup();
      server.use(http.post(OFFER_URL, () => HttpResponse.error()));

      await expect(startCall(deps, SETTINGS, handlers)).rejects.toThrow(/reach the server/i);
    });

    it("tells the learner when the microphone is refused, and sends no offer", async () => {
      const { deps, handlers, media } = setup();
      const offer = vi.fn();
      server.use(
        http.post(OFFER_URL, () => {
          offer();
          return HttpResponse.json(answer);
        }),
      );
      media.getUserMedia.mockRejectedValue(new DOMException("denied", "NotAllowedError"));

      await expect(startCall(deps, SETTINGS, handlers)).rejects.toThrow(/microphone/i);
      expect(offer).not.toHaveBeenCalled();
    });
  });
});

describe("startCall without a usable microphone", () => {
  it("says no microphone was found when the browser has none", async () => {
    const { deps, handlers, media } = setup();
    media.getUserMedia.mockRejectedValue(new DOMException("none", "NotFoundError"));

    await expect(startCall(deps, SETTINGS, handlers)).rejects.toThrow(/no microphone/i);
  });

  it("fails cleanly when the offer cannot even be created", async () => {
    const { deps, handlers, pc, media } = setup();
    pc.createOffer = () => Promise.reject(new Error("boom"));

    await expect(startCall(deps, SETTINGS, handlers)).rejects.toThrow(/could not be started/i);
    expect(pc.closed).toBe(true);
    expect(media.track.stop).toHaveBeenCalled();
  });
});
