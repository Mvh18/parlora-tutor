import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

// Needs the server and database up (`just up`); see playwright.config.ts.
test("the dev client shows a healthy server and passes axe", async ({ page }) => {
  await page.goto("/");

  await expect(page.locator("html")).toHaveAttribute("lang", "en");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Parlora Tutor");
  await expect(page.getByRole("status")).toHaveAttribute("data-status", "healthy");

  const { violations } = await new AxeBuilder({ page }).analyze();
  const blocking = violations.filter((v) => v.impact === "serious" || v.impact === "critical");
  expect(blocking.map((v) => `${v.id}: ${v.help}`)).toEqual([]);
});
