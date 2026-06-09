import { expect, test } from "@playwright/test";

import { authenticatePage, seedEmptyWorkspace } from "./helpers";

test("first-run dashboard shows three intro motion cards for an empty workspace", async ({
  page,
  request,
  baseURL,
}) => {
  const setup = await seedEmptyWorkspace(request, "Intro Motion Empty State");
  await authenticatePage(page, setup);

  await page.goto(`${baseURL}/dashboard`);

  await expect(page.getByText("Workspace Ready")).toBeVisible();
  await expect(page.getByTestId("dashboard-intro-gallery")).toBeVisible();
  await expect(page.getByTestId("intro-card-workspace")).toBeVisible();
  await expect(page.getByTestId("intro-card-scoring")).toBeVisible();
  await expect(page.getByTestId("intro-card-chat")).toBeVisible();
});

test("dashboard intro cards fall back to non-looping reduced-motion state", async ({
  page,
  request,
  baseURL,
}) => {
  const setup = await seedEmptyWorkspace(request, "Intro Motion Reduced");
  await authenticatePage(page, setup);
  await page.emulateMedia({ reducedMotion: "reduce" });

  await page.goto(`${baseURL}/dashboard`);

  const scoringCard = page.getByTestId("intro-card-scoring");
  await expect(scoringCard).toHaveAttribute("data-motion-mode", "reduced");
  await expect(scoringCard).toHaveAttribute("data-scene-state", "final");
});
