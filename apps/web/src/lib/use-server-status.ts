import { useEffect, useState } from "react";

import { createApiClient, type ApiClient } from "./api/client";

export type ServerStatus = "checking" | "healthy" | "degraded" | "unreachable";

const defaultClient = createApiClient();

/**
 * Healthy when live and ready both answer 200, degraded when the server
 * answers but is not ready (readiness 503), unreachable on a network error.
 */
export async function probeServer(client: ApiClient): Promise<Exclude<ServerStatus, "checking">> {
  try {
    const live = await client.GET("/api/v1/health/live");
    if (!live.response.ok) return "degraded";
    const ready = await client.GET("/api/v1/health/ready");
    return ready.response.ok ? "healthy" : "degraded";
  } catch {
    return "unreachable";
  }
}

/** Probes the server on mount and then every `intervalMs`. */
export function useServerStatus(
  client: ApiClient = defaultClient,
  intervalMs = 10_000,
): ServerStatus {
  const [status, setStatus] = useState<ServerStatus>("checking");

  useEffect(() => {
    let active = true;
    const probe = () => {
      void probeServer(client).then((next) => {
        if (active) setStatus(next);
      });
    };
    probe();
    const timer = setInterval(probe, intervalMs);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [client, intervalMs]);

  return status;
}
