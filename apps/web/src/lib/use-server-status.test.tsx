import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { scenarios, server } from "../test/server";
import { createApiClient } from "./api/client";
import { probeServer, useServerStatus } from "./use-server-status";

const client = createApiClient();

describe("probeServer", () => {
  it.each([
    ["healthy", "healthy"],
    ["degraded", "degraded"],
    ["liveFailing", "degraded"],
    ["unreachable", "unreachable"],
  ] as const)("maps the %s scenario to %s", async (scenario, expected) => {
    server.use(...scenarios[scenario]);

    await expect(probeServer(client)).resolves.toBe(expected);
  });
});

describe("useServerStatus", () => {
  it("starts as checking, then reports the probe result", async () => {
    const { result } = renderHook(() => useServerStatus(client));

    expect(result.current).toBe("checking");
    await waitFor(() => {
      expect(result.current).toBe("healthy");
    });
  });

  it("polls, so a change on the server shows up without a reload", async () => {
    const { result } = renderHook(() => useServerStatus(client, 20));
    await waitFor(() => {
      expect(result.current).toBe("healthy");
    });

    server.use(...scenarios.degraded);

    await waitFor(() => {
      expect(result.current).toBe("degraded");
    });
  });
});
