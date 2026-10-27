import createClient from "openapi-fetch";

import type { paths } from "./schema";

/** The local Compose stack publishes the server on 127.0.0.1:8700. */
export const API_BASE_URL = import.meta.env.VITE_PARLORA_API_URL ?? "http://127.0.0.1:8700";

export type ApiClient = ReturnType<typeof createApiClient>;

/** Typed client for the Parlora Tutor API, generated from contracts/openapi.json. */
export function createApiClient(baseUrl: string = API_BASE_URL) {
  // Look `fetch` up on every call rather than once, so a replacement installed
  // later (MSW in tests) is honoured.
  return createClient<paths>({ baseUrl, fetch: (request) => globalThis.fetch(request) });
}
