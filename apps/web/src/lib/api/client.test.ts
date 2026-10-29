import { describe, expect, it } from "vitest";

import { API_BASE_URL, createApiClient } from "./client";

describe("createApiClient", () => {
  it("defaults to the local Compose stack", () => {
    expect(API_BASE_URL).toBe("http://127.0.0.1:8700");
  });

  it("calls typed endpoints under the base URL", async () => {
    const { data, response } = await createApiClient().GET("/api/v1/health/ready");

    expect(response.status).toBe(200);
    expect(data?.checks).toEqual({ database: "ok" });
  });
});
