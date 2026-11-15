import { useCallback, useEffect, useRef, useState } from "react";

import { createApiClient } from "../api/client";
import {
  CallError,
  startCall,
  type ActiveCall,
  type CallDeps,
  type CallSettings,
} from "./call-session";

export type CallState = "idle" | "connecting" | "in-call" | "ended";

/** The browser's own WebRTC and microphone, with the generated API client. */
export function browserCallDeps(): CallDeps {
  return {
    client: createApiClient(),
    mediaDevices: navigator.mediaDevices,
    createPeerConnection: () => new RTCPeerConnection(),
  };
}

export interface UseCall {
  state: CallState;
  /** A readable message for the last failure; cleared by the next start. */
  error: string | null;
  latencyMs: number | null;
  remoteStream: MediaStream | null;
  start: (settings: CallSettings) => Promise<void>;
  end: () => Promise<void>;
}

/** The call state machine: idle → connecting → in call → ended (and back to start). */
export function useCall(deps: CallDeps | (() => CallDeps) = browserCallDeps): UseCall {
  const [state, setState] = useState<CallState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
  const active = useRef<ActiveCall | null>(null);

  const start = useCallback(
    async (settings: CallSettings) => {
      setError(null);
      setLatencyMs(null);
      setRemoteStream(null);
      setState("connecting");
      try {
        active.current = await startCall(typeof deps === "function" ? deps() : deps, settings, {
          onConnected: () => {
            setState("in-call");
          },
          onEnded: () => {
            active.current = null;
            setState("ended");
          },
          onRemoteStream: setRemoteStream,
          onLatency: setLatencyMs,
        });
      } catch (failure) {
        setError(failure instanceof CallError ? failure.message : "The call could not be started.");
        setState("idle");
      }
    },
    [deps],
  );

  const end = useCallback(async () => {
    await active.current?.end();
  }, []);

  // Leaving the page hangs up.
  useEffect(() => () => void active.current?.end(), []);

  return { state, error, latencyMs, remoteStream, start, end };
}
