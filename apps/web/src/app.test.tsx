import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { App } from "./app";
import { scenarios, server } from "./test/server";

describe("App", () => {
  it("names the product and the dev client in a single h1 page", async () => {
    render(<App />);

    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Parlora Tutor");
    expect(screen.getByText(/Parlora Tutor dev client/)).toBeInTheDocument();
    expect(await screen.findByText(/healthy/i)).toBeInTheDocument();
  });

  it.each(["degraded", "unreachable"] as const)(
    "still renders when the server is %s",
    async (scenario) => {
      server.use(...scenarios[scenario]);

      render(<App />);

      expect(screen.getByRole("heading", { level: 1 })).toBeInTheDocument();
      expect(await screen.findByText(new RegExp(scenario, "i"))).toBeInTheDocument();
    },
  );
});
