import type { ApiClient } from "../api/client";
import type { components } from "../api/schema";

type CefrLevel = components["schemas"]["CefrLevel"];

export interface CallSettings {
  level: CefrLevel;
  topic: string;
}

export interface CallDeps {
  client: ApiClient;
  mediaDevices: Pick<MediaDevices, "getUserMedia">;
  createPeerConnection: () => RTCPeerConnection;
}

export interface CallHandlers {
  /** The peer connection is up: the learner can speak. */
  onConnected: () => void;
  /** The call is over, whoever ended it; called once. */
  onEnded: () => void;
  onRemoteStream: (stream: MediaStream) => void;
  /** Voice-to-voice latency of the last turn, in milliseconds. */
  onLatency: (milliseconds: number) => void;
}

export interface ActiveCall {
  end: () => Promise<void>;
}

/** A failure with a message that is safe and useful to show to the learner. */
export class CallError extends Error {
  override readonly name = "CallError";
}

const DEPENDENCY_NAMES: Record<string, string> = {
  stt: "speech recognition",
  tts: "speech synthesis",
  llm: "the language model",
  vad: "voice detection",
};

const TARGET_LANGUAGE = "en";
const RTVI_LABEL = "rtvi-ai";
const LATENCY_MESSAGE = "turn-latency";

/** The learner-facing message for a problem document from the offer request. */
export function problemMessage(problem: unknown): string {
  const body = (problem ?? {}) as { type?: unknown; status?: unknown; unavailable?: unknown };
  if (body.type === "/problems/voice-unavailable") {
    const names = (Array.isArray(body.unavailable) ? body.unavailable : [])
      .map((name) => DEPENDENCY_NAMES[String(name)] ?? String(name))
      .join(", ");
    return names
      ? `Calls are unavailable right now: ${names} not ready.`
      : "Calls are unavailable right now.";
  }
  if (body.type === "/problems/call-limit") {
    return "Another call is already in progress. End it and try again.";
  }
  if (body.status === 422) return "The call settings are not valid. Check the level and topic.";
  return "The call could not be started.";
}

function microphoneMessage(error: unknown): string {
  const denied = error instanceof DOMException && error.name === "NotAllowedError";
  return denied
    ? "The microphone is blocked. Allow it in the browser to start a call."
    : "No microphone could be used. Check that one is connected.";
}

function stopTracks(stream: MediaStream) {
  for (const track of stream.getTracks()) track.stop();
}

/**
 * Starts a call with the signalling of Pipecat's small-WebRTC transport:
 * microphone first (ADR 0003), then the offer with the settings under
 * `requestData`, then ICE candidates by `pc_id`. Rejects with a `CallError`
 * whose message can be shown as is.
 */
export async function startCall(
  deps: CallDeps,
  settings: CallSettings,
  handlers: CallHandlers,
): Promise<ActiveCall> {
  let microphone: MediaStream;
  try {
    microphone = await deps.mediaDevices.getUserMedia({ audio: true });
  } catch (error) {
    throw new CallError(microphoneMessage(error));
  }

  const pc = deps.createPeerConnection();
  let callId: string | null = null;
  let ended = false;
  const pending: RTCIceCandidateInit[] = [];

  const release = () => {
    stopTracks(microphone);
    pc.close();
  };

  const sendCandidates = async (candidates: RTCIceCandidateInit[]) => {
    if (callId === null || candidates.length === 0) return;
    await deps.client
      .PATCH("/api/v1/calls/offer", {
        body: {
          pc_id: callId,
          candidates: candidates.map((c) => ({
            candidate: c.candidate ?? "",
            sdp_mid: c.sdpMid ?? null,
            sdp_mline_index: c.sdpMLineIndex ?? null,
          })),
        },
      })
      .catch(() => undefined);
  };

  const end = async () => {
    if (ended) return;
    ended = true;
    const id = callId;
    release();
    if (id !== null) {
      await deps.client
        .DELETE("/api/v1/calls/{call_id}", { params: { path: { call_id: id } } })
        .catch(() => undefined);
    }
    handlers.onEnded();
  };

  try {
    for (const track of microphone.getTracks()) pc.addTrack(track, microphone);

    const channel = pc.createDataChannel("chat", { ordered: true });
    const clientReady = () => {
      channel.send(
        JSON.stringify({
          label: RTVI_LABEL,
          type: "client-ready",
          id: crypto.randomUUID(),
          data: { version: "1.0.0", about: { library: "parlora-web" } },
        }),
      );
    };
    channel.onopen = clientReady;
    channel.onmessage = (event: MessageEvent<unknown>) => {
      const latency = latencyFrom(event.data);
      if (latency !== null) handlers.onLatency(latency);
    };

    pc.ontrack = (event) => {
      const stream = event.streams[0];
      if (stream) handlers.onRemoteStream(stream);
    };
    pc.onicecandidate = (event) => {
      if (!event.candidate) return;
      const candidate = event.candidate.toJSON();
      if (callId === null) pending.push(candidate);
      else void sendCandidates([candidate]);
    };
    pc.onconnectionstatechange = () => {
      if (pc.connectionState === "connected") handlers.onConnected();
      else if (pc.connectionState === "failed" || pc.connectionState === "closed") void end();
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    let result;
    try {
      result = await deps.client.POST("/api/v1/calls/offer", {
        body: {
          sdp: offer.sdp ?? "",
          type: "offer",
          pc_id: null,
          restart_pc: false,
          requestData: {
            target_language: TARGET_LANGUAGE,
            level: settings.level,
            topic: settings.topic,
          },
        },
      });
    } catch {
      throw new CallError("Could not reach the server. Is it running?");
    }
    if (!result.data) throw new CallError(problemMessage(result.error));

    callId = result.data.call_id;
    await pc.setRemoteDescription({ type: "answer", sdp: result.data.sdp });
    await sendCandidates(pending.splice(0));
  } catch (error) {
    ended = true;
    release();
    throw error instanceof CallError ? error : new CallError("The call could not be started.");
  }

  return { end };
}

/** The latency of a `turn-latency` RTVI server message, or null for anything else. */
function latencyFrom(raw: unknown): number | null {
  if (typeof raw !== "string") return null;
  let message: {
    label?: unknown;
    type?: unknown;
    data?: { type?: unknown; voice_to_voice_ms?: unknown };
  };
  try {
    message = JSON.parse(raw) as typeof message;
  } catch {
    return null;
  }
  if (message.label !== RTVI_LABEL || message.type !== "server-message") return null;
  if (message.data?.type !== LATENCY_MESSAGE) return null;
  const ms = message.data.voice_to_voice_ms;
  return typeof ms === "number" ? ms : null;
}
