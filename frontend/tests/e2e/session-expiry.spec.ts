import { expect, test } from "@playwright/test";

import { authenticatePage, seedWorkspace } from "./helpers";

test("expired authenticated session redirects back to login instead of surfacing token errors", async ({
  page,
  request,
  baseURL,
}) => {
  const setup = await seedWorkspace(request, "Session Expiry", [
    {
      fullName: "Session Tester",
      email: "session.tester@example.com",
      lines: [
        "Session Tester",
        "QA Engineer",
        "Authentication regression coverage",
      ],
    },
  ]);

  await authenticatePage(page, setup);
  await page.goto(`${baseURL}/settings`);
  await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible();

  await page.evaluate(() => {
    localStorage.setItem("recruitai.token", "expired.token.value");
  });

  await page.getByRole("button", { name: "Save changes" }).click();

  await expect(page).toHaveURL(/\/login\?redirect=%2Fsettings/, { timeout: 15_000 });
});
