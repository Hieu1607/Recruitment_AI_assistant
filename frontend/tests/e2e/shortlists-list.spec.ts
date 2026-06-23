import { expect, test } from "@playwright/test";

import { authenticatePage } from "./helpers";

test("shortlists list shows collections without query history", async ({ page, baseURL }) => {
  const setup = {
    accessToken: "playwright-shortlists-list-token",
    jobId: "job-shortlists-list-1",
    publicApplyToken: "unused",
  };

  await page.route("**/api/v1/auth/me**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: "user-shortlists-list-1",
        email: "shortlists-list@example.com",
        display_name: "Shortlists List Tester",
      }),
    });
  });

  await page.route("**/api/v1/jobs**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        items: [
          {
            id: setup.jobId,
            owner_user_id: "user-shortlists-list-1",
            title: "Shortlists List Workspace",
            status: "active",
            public_apply_enabled: true,
            public_apply_url: "http://127.0.0.1:8000/public/jobs/unused",
            candidate_message: null,
            created_at: "2026-06-22T08:00:00Z",
            updated_at: "2026-06-22T08:00:00Z",
            archived_at: null,
          },
        ],
        total: 1,
      }),
    });
  });

  await page.route("**/api/v1/notifications/**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ items: [], unread_count: 0 }),
    });
  });

  await page.route("**/api/v1/shortlist/collections**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        items: [
          {
            id: "collection-shortlists-list-1",
            name: "Priority accountants",
            created_by_user_id: "user-shortlists-list-1",
            source_query_turn_id: null,
            item_count: 2,
            created_at: "2026-06-22T08:30:00Z",
          },
        ],
        total: 1,
      }),
    });
  });

  await authenticatePage(page, setup);
  await page.goto(`${baseURL}/shortlists`);

  await expect(page.getByRole("heading", { name: "Shortlists" })).toBeVisible();
  await expect(page.getByText("Priority accountants")).toBeVisible();
  await expect(page.getByRole("button", { name: "Query History" })).toHaveCount(0);
  await expect(page.getByText("query session history")).toHaveCount(0);
});
