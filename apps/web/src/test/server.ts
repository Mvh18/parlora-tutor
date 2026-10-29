import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";

import { API_BASE_URL } from "../lib/api/client";

const live = `${API_BASE_URL}/api/v1/health/live`;
const ready = `${API_BASE_URL}/api/v1/health/ready`;

const problem = (status: number, title: string) =>
  HttpResponse.json(
    { type: "about:blank", title, status },
    { status, headers: { "Content-Type": "application/problem+json" } },
  );

/** Handlers for each server state the status badge distinguishes. */
export const scenarios = {
  healthy: [
    http.get(live, () => HttpResponse.json({ status: "ok" })),
    http.get(ready, () => HttpResponse.json({ status: "ok", checks: { database: "ok" } })),
  ],
  degraded: [
    http.get(live, () => HttpResponse.json({ status: "ok" })),
    http.get(ready, () => problem(503, "Service Unavailable")),
  ],
  liveFailing: [http.get(live, () => problem(500, "Internal Server Error"))],
  unreachable: [http.get(live, () => HttpResponse.error())],
};

export const server = setupServer(...scenarios.healthy);
