import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const API = "http://127.0.0.1:8700";

// Needs the stack up (`just up`, or `just up nogpu`); see playwright.config.ts.
// Chromium answers the microphone prompt and feeds a synthetic tone.
test.use({
  permissions: ["microphone"],
  launchOptions: {
    args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"],
  },
});

test("the call page starts a call, or says calls are unavailable, and stays local", async ({
  page,
  request,
}) => {
  const hosts = new Set<string>();
  page.on("request", (req) => {
    const url = new URL(req.url());
    if (url.protocol === "http:" || url.protocol === "https:") hosts.add(url.host);
  });

  const ready = await request.get(`${API}/api/v1/health/ready`);
  const voiceReady = ready.ok();

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Call the tutor" })).toBeVisible();
  const status = page.getByRole("status", { name: "Call status" });
  await expect(status).toHaveText(/ready to start/i);

  const start = page.getByRole("button", { name: "Start call" });
  await start.focus();
  await expect(start).toBeFocused();
  await page.keyboard.press("Enter");

  if (voiceReady) {
    await expect(status).toHaveText(/in call/i, { timeout: 30_000 });
    await page.getByRole("button", { name: "End call" }).click();
    await expect(status).toHaveText(/call ended/i);
  } else {
    await expect(page.getByRole("alert")).toContainText(/unavailable/i);
  }

  const { violations } = await new AxeBuilder({ page }).analyze();
  const blocking = violations.filter((v) => v.impact === "serious" || v.impact === "critical");
  expect(blocking.map((v) => `${v.id}: ${v.help}`)).toEqual([]);

  // Only the dev server and the Parlora API: nothing from a third-party host.
  expect([...hosts].filter((h) => h !== "localhost:3000" && h !== "127.0.0.1:8700")).toEqual([]);
});
