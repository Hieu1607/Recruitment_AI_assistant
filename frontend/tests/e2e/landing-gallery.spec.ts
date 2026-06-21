import { expect, test } from "@playwright/test";

test("landing page renders autoplay story slides for overview, scoring, and assistant", async ({ page, baseURL }) => {
  await page.goto(`${baseURL}/`);

  const overview = page.getByTestId("landing-story-overview");
  const scoring = page.getByTestId("landing-story-scoring");
  const assistant = page.getByTestId("landing-story-assistant");

  await expect(overview).toBeVisible();
  await expect(scoring).toBeVisible();
  await expect(assistant).toBeVisible();
  await expect(overview).toContainText("Upload CVs");
  await expect(scoring).toContainText("Adjust score weights");
  await expect(assistant).toContainText("Ask the candidate pool");
  await expect(page.getByAltText("EasyHR dashboard overview")).toHaveCount(0);
  await expect(page.getByAltText("EasyHR scoring interface")).toHaveCount(0);
  await expect(page.getByAltText("EasyHR AI assistant interface")).toHaveCount(0);
  await expect(page.getByText("[Scoring Interface Screenshot]")).toHaveCount(0);
  await expect(page.getByText("[AI Chat Interface Screenshot]")).toHaveCount(0);
});
