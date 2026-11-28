import { act, renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";

import { API_BASE_URL, createApiClient } from "../api/client";
import { server } from "../../test/server";
import { FakePeerConnection, fakeMediaDevices } from "./fakes";
import { useCall } from "./use-call";

const OFFER_URL = `${API_BASE_URL}/api/v1/calls/offer`;
const answer = { call_id: "call-1", pc_id: "call-1", sdp: "v=0 answer", type: "answer" };

function deps() {
  const pc = new FakePeerConnection();
  const media = fakeMediaDevices();
  const make = () => ({
    client: createApiClient(),
    mediaDevices: media,
    createPeerConnection: () => pc as unknown as RTCPeerConnection,
  });
  return { pc, media, make };
}

describe("useCall", () => {
  it("accepts a factory for its dependencies and hangs up on unmount", async () => {
    const { pc, make } = deps();
    const deleted = vi.fn();
    server.use(
      http.post(OFFER_URL, () => HttpResponse.json(answer)),
      http.delete(`${API_BASE_URL}/api/v1/calls/call-1`, () => {
        deleted();
        return new HttpResponse(null, { status: 204 });
      }),
    );
    const { result, unmount } = renderHook(() => useCall(make));

    await act(() => result.current.start({ level: "B1", topic: "travel" }));
    act(() => {
      pc.setConnectionState("connected");
    });
    expect(result.current.state).toBe("in-call");
    unmount();

    await waitFor(() => {
      expect(deleted).toHaveBeenCalledOnce();
    });
  });

  it("falls back to a generic message for an unexpected failure", async () => {
    const { make } = deps();
    const broken = () => ({
      ...make(),
      createPeerConnection: () => {
        throw new Error("no WebRTC");
      },
    });
    const { result } = renderHook(() => useCall(broken));

    await act(() => result.current.start({ level: "B1", topic: "travel" }));

    expect(result.current.state).toBe("idle");
    expect(result.current.error).toMatch(/could not be started/i);
  });

  it("ends in the ended state when the connection fails", async () => {
    const { pc, make } = deps();
    server.use(
      http.post(OFFER_URL, () => HttpResponse.json(answer)),
      http.delete(
        `${API_BASE_URL}/api/v1/calls/call-1`,
        () => new HttpResponse(null, { status: 204 }),
      ),
    );
    const { result } = renderHook(() => useCall(make));
    await act(() => result.current.start({ level: "B1", topic: "travel" }));

    act(() => {
      pc.setConnectionState("failed");
    });

    await waitFor(() => {
      expect(result.current.state).toBe("ended");
    });
  });
});
