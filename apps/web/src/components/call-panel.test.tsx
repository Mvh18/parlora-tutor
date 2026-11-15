import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";

import { API_BASE_URL, createApiClient } from "../lib/api/client";
import { FakePeerConnection, fakeMediaDevices } from "../lib/call/fakes";
import { server } from "../test/server";
import { CallPanel } from "./call-panel";

const OFFER_URL = `${API_BASE_URL}/api/v1/calls/offer`;
const answer = { call_id: "call-1", pc_id: "call-1", sdp: "v=0 answer", type: "answer" };

function setup() {
  const pc = new FakePeerConnection();
  const media = fakeMediaDevices();
  render(
    <CallPanel
      deps={{
        client: createApiClient(),
        mediaDevices: media,
        createPeerConnection: () => pc as unknown as RTCPeerConnection,
      }}
    />,
  );
  return { pc, media };
}

describe("CallPanel", () => {
  it("goes idle → connecting → in call → ended", async () => {
    server.use(
      http.post(OFFER_URL, () => HttpResponse.json(answer)),
      http.delete(
        `${API_BASE_URL}/api/v1/calls/call-1`,
        () => new HttpResponse(null, { status: 204 }),
      ),
    );
    const { pc } = setup();
    expect(screen.getByRole("status", { name: /call/i })).toHaveTextContent(/ready to start/i);
    expect(screen.queryByRole("button", { name: /end call/i })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /start call/i }));
    await waitFor(() =>
      expect(screen.getByRole("status", { name: /call/i })).toHaveTextContent(/connecting/i),
    );
    expect(screen.getByLabelText(/topic/i)).toBeDisabled();

    act(() => {
      pc.setConnectionState("connected");
    });
    expect(screen.getByRole("status", { name: /call/i })).toHaveTextContent(/in call/i);

    fireEvent.click(screen.getByRole("button", { name: /end call/i }));
    await waitFor(() =>
      expect(screen.getByRole("status", { name: /call/i })).toHaveTextContent(/call ended/i),
    );
    expect(screen.getByRole("button", { name: /start call/i })).toBeEnabled();
  });

  it("sends the chosen level and topic", async () => {
    let body: { requestData?: unknown } = {};
    server.use(
      http.post(OFFER_URL, async ({ request }) => {
        body = (await request.json()) as typeof body;
        return HttpResponse.json(answer);
      }),
    );
    setup();

    fireEvent.change(screen.getByLabelText(/level/i), { target: { value: "C1" } });
    fireEvent.change(screen.getByLabelText(/topic/i), { target: { value: "" } });
    fireEvent.change(screen.getByLabelText(/topic/i), { target: { value: "cooking" } });
    fireEvent.click(screen.getByRole("button", { name: /start call/i }));

    await waitFor(() => {
      expect(body.requestData).toEqual({ target_language: "en", level: "C1", topic: "cooking" });
    });
  });

  it("shows the problem as a readable message and lets the learner retry", async () => {
    server.use(
      http.post(OFFER_URL, () =>
        HttpResponse.json(
          { type: "/problems/voice-unavailable", title: "x", status: 503, unavailable: ["llm"] },
          { status: 503, headers: { "Content-Type": "application/problem+json" } },
        ),
      ),
    );
    setup();

    fireEvent.click(screen.getByRole("button", { name: /start call/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/language model/i);
    expect(screen.getByRole("button", { name: /start call/i })).toBeEnabled();
  });

  it("shows the last turn's latency and plays the tutor's audio", async () => {
    server.use(http.post(OFFER_URL, () => HttpResponse.json(answer)));
    const { pc } = setup();
    fireEvent.click(screen.getByRole("button", { name: /start call/i }));
    await waitFor(() => {
      expect(pc.channel).toBeDefined();
    });
    await waitFor(() => {
      expect(pc.remoteDescription).not.toBeNull();
    });

    act(() => {
      pc.channel.receive({
        label: "rtvi-ai",
        type: "server-message",
        data: { type: "turn-latency", voice_to_voice_ms: 900 },
      });
      pc.emitTrack({});
    });

    expect(screen.getByText(/900 ms/)).toBeInTheDocument();
    expect(document.querySelector("audio")).not.toBeNull();
  });

  it("rejects an empty topic before touching the microphone", async () => {
    const { media } = setup();

    fireEvent.change(screen.getByLabelText(/topic/i), { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: /start call/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/topic/i);
    expect(media.getUserMedia).not.toHaveBeenCalled();
  });
});
