import { useEffect, useRef, useState, type SyntheticEvent } from "react";

import type { CallDeps } from "../lib/call/call-session";
import { useCall, type CallState } from "../lib/call/use-call";

const LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"] as const;
type Level = (typeof LEVELS)[number];

const STATE_LABELS: Record<CallState, string> = {
  idle: "Ready to start",
  connecting: "Connecting… allow the microphone if asked",
  "in-call": "In call: speak and the tutor will answer",
  ended: "Call ended",
};

/** Call page: level and topic, Start/End, the tutor's audio and the last turn's latency. */
export function CallPanel({ deps }: { deps?: CallDeps }) {
  const call = useCall(deps);
  const [level, setLevel] = useState<Level>("B1");
  const [topic, setTopic] = useState("daily life");
  const [formError, setFormError] = useState<string | null>(null);
  const audio = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    if (audio.current) audio.current.srcObject = call.remoteStream;
  }, [call.remoteStream]);

  const busy = call.state === "connecting" || call.state === "in-call";
  const message = formError ?? call.error;

  const submit = (event: SyntheticEvent) => {
    event.preventDefault();
    if (busy) return;
    const trimmed = topic.trim();
    if (!trimmed) {
      setFormError("Enter a topic for the call.");
      return;
    }
    setFormError(null);
    void call.start({ level, topic: trimmed });
  };

  return (
    <section aria-labelledby="call-heading" className="call">
      <h2 id="call-heading">Call the tutor</h2>
      <form onSubmit={submit} className="call__form">
        <label>
          Level
          <select
            value={level}
            disabled={busy}
            onChange={(e) => {
              setLevel(e.target.value as Level);
            }}
          >
            {LEVELS.map((code) => (
              <option key={code} value={code}>
                {code}
              </option>
            ))}
          </select>
        </label>
        <label>
          Topic
          <input
            type="text"
            value={topic}
            maxLength={200}
            disabled={busy}
            onChange={(e) => {
              setTopic(e.target.value);
            }}
          />
        </label>
        <div className="call__actions">
          {busy ? (
            <button type="button" onClick={() => void call.end()}>
              End call
            </button>
          ) : (
            <button type="submit">Start call</button>
          )}
        </div>
      </form>
      <p role="status" aria-label="Call status" data-state={call.state} className="call__state">
        {STATE_LABELS[call.state]}
      </p>
      {message ? (
        <p role="alert" className="call__error">
          {message}
        </p>
      ) : null}
      {call.latencyMs !== null ? (
        <p className="call__latency">Last turn: {call.latencyMs} ms</p>
      ) : null}
      <audio ref={audio} autoPlay />
    </section>
  );
}
