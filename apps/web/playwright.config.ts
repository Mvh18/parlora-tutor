import { defineConfig, devices } from "@playwright/test";

const CI = Boolean(process.env.CI);

// The smoke test runs against the production build, served by `vite preview`,
// and a real server started beforehand (`just up`).
export default defineConfig({
  testDir: "e2e",
  forbidOnly: CI,
  retries: CI ? 1 : 0,
  reporter: CI ? [["github"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: "http://localhost:3000",
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "pnpm build && pnpm preview",
    url: "http://localhost:3000",
    reuseExistingServer: !CI,
    timeout: 120_000,
  },
});
