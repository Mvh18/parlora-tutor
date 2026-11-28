import { vi } from "vitest";

/** Test doubles for the browser's WebRTC and media APIs (jsdom has neither). */

export class FakeDataChannel {
  readyState: RTCDataChannelState = "connecting";
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  readonly sent: unknown[] = [];

  constructor(readonly label: string) {}

  send(data: unknown) {
    this.sent.push(data);
  }

  close() {
    this.readyState = "closed";
  }

  open() {
    this.readyState = "open";
    this.onopen?.();
  }

  /** Delivers a message; objects are serialised the way the server sends them. */
  receive(message: unknown) {
    this.onmessage?.({ data: typeof message === "string" ? message : JSON.stringify(message) });
  }
}

export class FakePeerConnection {
  localDescription: RTCSessionDescriptionInit | null = null;
  remoteDescription: RTCSessionDescriptionInit | null = null;
  connectionState: RTCPeerConnectionState = "new";
  closed = false;
  channel!: FakeDataChannel;
  readonly tracks: unknown[] = [];

  onicecandidate: ((event: { candidate: unknown }) => void) | null = null;
  onconnectionstatechange: (() => void) | null = null;
  ontrack: ((event: { streams: unknown[] }) => void) | null = null;

  addTrack(track: unknown) {
    this.tracks.push(track);
  }

  createDataChannel(label: string) {
    this.channel = new FakeDataChannel(label);
    return this.channel;
  }

  createOffer() {
    return Promise.resolve({ type: "offer" as const, sdp: "v=0 fake offer" });
  }

  setLocalDescription(description: RTCSessionDescriptionInit) {
    this.localDescription = description;
    return Promise.resolve();
  }

  setRemoteDescription(description: RTCSessionDescriptionInit) {
    this.remoteDescription = description;
    return Promise.resolve();
  }

  close() {
    this.closed = true;
  }

  setConnectionState(state: RTCPeerConnectionState) {
    this.connectionState = state;
    this.onconnectionstatechange?.();
  }

  /** A gathered candidate, or `null` for the end of gathering. */
  emitCandidate(
    candidate: string | null,
    sdpMid: string | null = null,
    index: number | null = null,
  ) {
    this.onicecandidate?.({
      candidate:
        candidate === null
          ? null
          : {
              candidate,
              sdpMid,
              sdpMLineIndex: index,
              toJSON: () => ({ candidate, sdpMid, sdpMLineIndex: index }),
            },
    });
  }

  emitTrack(stream: unknown) {
    this.ontrack?.({ streams: [stream] });
  }
}

export function fakeMediaDevices() {
  const track = { stop: vi.fn() };
  const stream = { getTracks: () => [track] } as unknown as MediaStream;
  return { track, stream, getUserMedia: vi.fn(() => Promise.resolve(stream)) };
}
