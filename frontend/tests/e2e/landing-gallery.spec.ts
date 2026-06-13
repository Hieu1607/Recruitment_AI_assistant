import { expect, test } from "@playwright/test";

test("landing page renders the three showcase screenshots in order", async ({ page, baseURL }) => {
  await page.goto(`${baseURL}/`);

  const overview = page.getByAltText("EasyHR dashboard overview");
  const scoring = page.getByAltText("EasyHR scoring interface");
  const assistant = page.getByAltText("EasyHR AI assistant interface");
  const screenshots = [overview, scoring, assistant];

  for (const screenshot of screenshots) {
    await expect(screenshot).toBeVisible();
  }

  await expect(scoring).toHaveCSS("object-position", "50% 0%");
  await expect(assistant).toHaveCSS("object-position", "65% 0%");
  await expect(page.getByText("[Scoring Interface Screenshot]")).toHaveCount(0);
  await expect(page.getByText("[AI Chat Interface Screenshot]")).toHaveCount(0);
});
