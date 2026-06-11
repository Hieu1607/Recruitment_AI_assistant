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

test("scoring intro card ends on a ranked shortlist frame", async ({
  page,
  request,
  baseURL,
}) => {
  const setup = await seedEmptyWorkspace(request, "Intro Motion Scoring");
  await authenticatePage(page, setup);

  await page.goto(`${baseURL}/dashboard`);

  const scoringCard = page.getByTestId("intro-card-scoring");
  await expect(scoringCard.getByText("Rank candidates by fit")).toBeVisible();
  await expect(scoringCard.getByText("Top matches")).toBeVisible({ timeout: 20_000 });
  await expect(scoringCard.getByText("92")).toBeVisible();
});

test("workspace and chat cards communicate setup flow and recruiter query value", async ({
  page,
  request,
  baseURL,
}) => {
  const setup = await seedEmptyWorkspace(request, "Intro Motion Full Set");
  await authenticatePage(page, setup);

  await page.goto(`${baseURL}/dashboard`);

  await expect(page.getByTestId("intro-card-workspace").getByText("One recruiting workspace")).toBeVisible();
  await expect(page.getByTestId("intro-card-workspace").getByText("Scoring ready")).toBeVisible({
    timeout: 20_000,
  });

  await expect(page.getByTestId("intro-card-chat").getByText("Ask the candidate pool")).toBeVisible();
  await expect(page.getByTestId("intro-card-chat").getByText("Python and FastAPI")).toBeVisible({
    timeout: 20_000,
  });
});
