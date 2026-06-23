import { expect, test } from "@playwright/test";

test("landing page renders the storyboard hero and static supporting product snapshots", async ({
  page,
  baseURL,
}) => {
  await page.goto(`${baseURL}/`);

  const overview = page.getByTestId("landing-story-overview");
  const scoring = page.getByTestId("landing-story-scoring");
  const assistant = page.getByTestId("landing-story-assistant");

  await expect(overview).toBeVisible();
  await expect(overview).toContainText("Tạo job");
  await expect(overview).toContainText("Upload CV");
  await expect(overview).toContainText("AI parse CV");
  await expect(overview).toContainText("Shortlist");

  await expect(scoring).toBeVisible();
  await expect(scoring).toContainText("Scoring Engine");
  await expect(scoring).toContainText("Candidate scorecard");

  await expect(assistant).toBeVisible();
  await expect(assistant).toContainText("AI Assistant");
  await expect(assistant).toContainText("Talent search");

  await expect(overview).toHaveAttribute("data-motion", "storyboard-shell");
  await expect(scoring).toHaveAttribute("data-motion", "reveal");
  await expect(assistant).toHaveAttribute("data-motion", "reveal");
  await expect(page.getByTestId("landing-uploading-badge")).toHaveAttribute("data-motion", "pulse");
  await expect(page.getByTestId("landing-shortlist-top")).toHaveAttribute("data-motion", "hero-highlight");

  await expect(overview).not.toContainText("Upload CVs");
  await expect(page.getByText("Adjust score weights")).toHaveCount(0);
  await expect(page.getByText("Ask the candidate pool")).toHaveCount(0);
});
