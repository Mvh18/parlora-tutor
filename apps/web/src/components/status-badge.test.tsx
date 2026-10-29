import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { StatusBadge } from "./status-badge";

describe("StatusBadge", () => {
  it.each([
    ["checking", /checking/i],
    ["healthy", /healthy/i],
    ["degraded", /degraded/i],
    ["unreachable", /unreachable/i],
  ] as const)("describes the %s state in words", (status, text) => {
    render(<StatusBadge status={status} />);

    const badge = screen.getByRole("status");
    expect(badge).toHaveTextContent(text);
    expect(badge).toHaveAttribute("data-status", status);
  });

  it("is announced through a polite live region", () => {
    render(<StatusBadge status="healthy" />);

    expect(screen.getByRole("status")).toHaveAttribute("aria-live", "polite");
  });
});
