import { expect, test } from "@playwright/test";

import { seedWorkspace } from "./helpers";

test("landing page stays in light mode when browser prefers dark mode", async ({ page, baseURL }) => {
  await page.emulateMedia({ colorScheme: "dark" });
  await page.goto(`${baseURL}/`);

  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
});

test("public apply stays in light mode when browser prefers dark mode", async ({
  page,
  request,
  baseURL,
}) => {
  const setup = await seedWorkspace(request, "Public Light Mode", []);

  await page.emulateMedia({ colorScheme: "dark" });
  await page.goto(`${baseURL}/apply/${setup.publicApplyToken}`);

  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
});
