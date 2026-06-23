import { expect, test } from "@playwright/test";

const TOKEN_KEY = "easyhr.token";
const SELECTED_JOB_KEY = "recruit_ai_selected_job_id";
const PREFERENCES_KEY = "easyhr.notification-preferences";

test("disabling realtime notification toasts keeps bell notifications without showing toast", async ({
  page,
  baseURL,
}) => {
  let notificationRequestCount = 0;

  await page.route("**/api/v1/**", async (route) => {
    const url = route.request().url();
    if (url.includes("/auth/me")) {
      return route.fulfill({
        json: {
          id: "user-1",
          email: "user@example.com",
          display_name: "Recruiter",
          gmail_connected: false,
        },
      });
    }
    if (url.endsWith("/jobs/") || url.includes("/jobs/?")) {
      return route.fulfill({
        json: {
          items: [
            {
              id: "job-1",
              owner_user_id: "user-1",
              title: "Backend Engineer",
              status: "active",
              candidate_message: null,
              public_apply_enabled: true,
              public_apply_url: "/apply/token",
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString(),
              archived_at: null,
            },
          ],
          total: 1,
        },
      });
    }
    if (url.includes("/setup-status")) {
      return route.fulfill({
        json: {
          job_id: "job-1",
          resume_count: 0,
          processed_candidate_count: 0,
          has_uploaded_resumes: false,
          has_processed_candidates: false,
          has_active_job_description: false,
          has_completed_score_run: false,
          has_chat_turn: false,
          completed_score_run_count: 0,
          chat_session_count: 0,
          chat_turn_count: 0,
          latest_job_description_id: null,
          latest_score_run_id: null,
          latest_score_run_at: null,
          latest_chat_session_id: null,
          latest_chat_turn_at: null,
        },
      });
    }
    if (url.includes("/notifications")) {
      notificationRequestCount += 1;
      const items =
        notificationRequestCount === 1
          ? []
          : [
              {
                id: "note-1",
                user_id: "user-1",
                notification_type: "candidate_applied",
                title: "Realtime candidate",
                body: "A candidate submitted a public JD application.",
                target_url: "/candidates/resume-1",
                payload: {},
                created_at: new Date().toISOString(),
                read_at: null,
              },
            ];
      return route.fulfill({
        json: {
          unread_count: items.length,
          items,
        },
      });
    }
    return route.fulfill({ json: {} });
  });

  await page.addInitScript(
    ([preferencesKey, tokenKey, jobKey]) => {
      localStorage.setItem(tokenKey, "mock-token");
      localStorage.setItem(jobKey, "job-1");
      localStorage.setItem(
        preferencesKey,
        JSON.stringify({
          candidate_applied: true,
          interview_completed: true,
          scoring_completed: true,
          realtime_toasts: false,
        }),
      );
    },
    [PREFERENCES_KEY, TOKEN_KEY, SELECTED_JOB_KEY],
  );

  await page.goto(`${baseURL}/dashboard`);
  await expect(page.getByRole("heading", { name: /Good/ })).toBeVisible();
  await expect
    .poll(() => notificationRequestCount, { timeout: 12_000 })
    .toBeGreaterThanOrEqual(2);
  await expect(
    page.getByRole("region", { name: /Notifications/ }).getByText("Realtime candidate"),
  ).toHaveCount(0);

  await page.locator('header [aria-label="Notifications"]').click();
  await expect(page.getByRole("menu").getByText("Realtime candidate")).toBeVisible();
});
