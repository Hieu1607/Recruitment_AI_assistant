import { expect, test } from "@playwright/test";

import { authenticatePage } from "./helpers";

test("chat empty state keeps the supporting copy centered under the headline", async ({ page, baseURL }) => {
  const setup = {
    accessToken: "playwright-token",
    jobId: "job-chat-empty-state-1",
    publicApplyToken: "unused",
  };

  await page.route("**/api/v1/auth/me", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: "user-chat-empty-state-1",
        email: "layout@example.com",
        display_name: "Layout Tester",
      }),
    });
  });

  await page.route("**/api/v1/jobs/", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        items: [
          {
            id: setup.jobId,
            title: "Chat Empty State Workspace",
            status: "active",
            created_at: "2026-06-05T08:00:00Z",
            updated_at: "2026-06-05T08:00:00Z",
          },
        ],
        total: 1,
      }),
    });
  });

  await page.route(`**/api/v1/jobs/${setup.jobId}/chat/sessions**`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        items: [],
        total: 0,
      }),
    });
  });

  await page.route(`**/api/v1/jobs/${setup.jobId}/candidates`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        items: [],
        total: 0,
      }),
    });
  });

  await authenticatePage(page, setup);
  await page.goto(`${baseURL}/chat`);

  const heading = page.getByRole("heading", { name: "Ask anything about your candidates" });
  const description = page.getByText(
    "Search, compare, and analyse your candidate pool using natural language.",
  );

  await expect(heading).toBeVisible();
  await expect(description).toBeVisible();

  const headingBox = await heading.boundingBox();
  const descriptionBox = await description.boundingBox();

  expect(headingBox).toBeTruthy();
  expect(descriptionBox).toBeTruthy();
  if (!headingBox || !descriptionBox) return;

  const headingCenterX = headingBox.x + headingBox.width / 2;
  const descriptionCenterX = descriptionBox.x + descriptionBox.width / 2;

  expect(Math.abs(descriptionCenterX - headingCenterX)).toBeLessThan(3);
});
