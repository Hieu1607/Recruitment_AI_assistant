import { expect, test } from "@playwright/test";

import { authenticatePage } from "./helpers";

test("shortlist collection keeps skill chips clear of outreach status badges", async ({
  page,
  baseURL,
}) => {
  const setup = {
    accessToken: "playwright-shortlist-layout-token",
    jobId: "job-shortlist-layout-1",
    publicApplyToken: "unused",
  };
  const collectionId = "collection-layout-1";

  await page.setViewportSize({ width: 1365, height: 900 });

  await page.route("**/api/v1/auth/me", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: "user-shortlist-layout-1",
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
            owner_user_id: "user-shortlist-layout-1",
            title: "Shortlist Layout Workspace",
            status: "active",
            public_apply_enabled: true,
            public_apply_url: "http://127.0.0.1:8000/public/jobs/unused",
            candidate_message: null,
            created_at: "2026-06-08T08:00:00Z",
            updated_at: "2026-06-08T08:00:00Z",
            archived_at: null,
          },
        ],
        total: 1,
      }),
    });
  });

  await page.route(`**/api/v1/shortlist/collections/${collectionId}`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: collectionId,
        name: "Priority teachers",
        created_by_user_id: "user-shortlist-layout-1",
        source_query_turn_id: null,
        item_count: 1,
        created_at: "2026-06-08T08:00:00Z",
      }),
    });
  });

  await page.route(`**/api/v1/shortlist/collections/${collectionId}/dispatch-summary`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        collection: {
          id: collectionId,
          name: "Priority teachers",
          item_count: 1,
        },
        job: {
          id: setup.jobId,
          title: "Shortlist Layout Workspace",
        },
        candidates: [
          {
            candidate_profile_id: "candidate-layout-1",
            full_name: "Le Thi Hoa",
            email: "hoa@example.com",
            current_job_title: "Senior Math Teacher",
            skills_text:
              "Curriculum Development, Classroom Management, Educational Technology, Assessment Design",
            contact_status: "ready",
            outreach: null,
            interview: null,
            blockers: [],
          },
        ],
        capabilities: {
          gmail_connected: false,
          active_interview_templates_count: 0,
        },
      }),
    });
  });

  await authenticatePage(page, setup);
  await page.goto(`${baseURL}/shortlists/collections/${collectionId}`);

  const skillChip = page.getByText("Assessment Design");
  const outreachBadge = page.getByText("not started");

  await expect(skillChip).toBeVisible();
  await expect(outreachBadge).toBeVisible();

  const skillBox = await skillChip.boundingBox();
  const outreachBox = await outreachBadge.boundingBox();

  expect(skillBox).not.toBeNull();
  expect(outreachBox).not.toBeNull();
  expect(skillBox!.x + skillBox!.width).toBeLessThanOrEqual(outreachBox!.x - 8);
});
