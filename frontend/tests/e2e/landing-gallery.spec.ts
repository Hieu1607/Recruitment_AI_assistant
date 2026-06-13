import { expect, test } from "@playwright/test";

test("landing page renders the three showcase screenshots in order", async ({ page, baseURL }) => {
  await page.goto(`${baseURL}/`);

  const screenshots = [
    page.getByAltText("EasyHR dashboard overview"),
    page.getByAltText("EasyHR scoring interface"),
    page.getByAltText("EasyHR AI assistant interface"),
  ];

  for (const screenshot of screenshots) {
    await expect(screenshot).toBeVisible();
  }

  await expect(page.getByText("[Scoring Interface Screenshot]")).toHaveCount(0);
  await expect(page.getByText("[AI Chat Interface Screenshot]")).toHaveCount(0);
});
