import { expect, test, type Page } from "@playwright/test";

import { authenticatePage } from "./helpers";

const setup = {
  accessToken: "playwright-token",
  jobId: "job-interview-guard-1",
  publicApplyToken: "unused",
};

async function mockInterviewShell(page: Page) {
  const recruiter = {
    id: "user-interview-guard-1",
    email: "interview-guard@example.com",
    display_name: "Interview Guard Tester",
  };

  const job = {
    id: setup.jobId,
    owner_user_id: recruiter.id,
    title: "Interview Guard Job",
    status: "active",
    public_apply_enabled: true,
    public_apply_url: "http://127.0.0.1:5173/apply/interview-guard-job",
    created_at: "2026-07-07T08:00:00Z",
    updated_at: "2026-07-07T08:00:00Z",
    archived_at: null,
  };

  const candidate = {
    id: "candidate-interview-guard-1",
    resume_document_id: "resume-interview-guard-1",
    full_name: "Nguyen Thi Guard",
    phone: null,
    email: "guard@example.com",
    location_normalized: "Ho Chi Minh City",
    current_job_title: "Recruiter",
    summary_text: null,
    skills_text: null,
    experience_text: null,
    experience_years: null,
    education_text: null,
    languages_text: null,
    projects_text: null,
    achievements_text: null,
    certifications_text: null,
  };

  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const method = route.request().method();
    const path = url.pathname.replace("/api/v1", "");

    const json = async (payload: unknown, status = 200) => {
      await route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify(payload),
      });
    };

    if (path === "/auth/me" && method === "GET") {
      await json(recruiter);
      return;
    }

    if (path === "/jobs/" && method === "GET") {
      await json({ items: [job], total: 1 });
      return;
    }

    if (path === `/jobs/${setup.jobId}/interview-invitations` && method === "GET") {
      await json({ items: [], total: 0 });
      return;
    }

    if (path === `/jobs/${setup.jobId}/interview-templates` && method === "GET") {
      await json({ items: [], total: 0 });
      return;
    }

    if (path === `/jobs/${setup.jobId}/candidates` && method === "GET") {
      await json({ items: [candidate], total: 1 });
      return;
    }

    await route.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({ detail: `Unhandled mock route: ${method} ${path}` }),
    });
  });
}

test("create interview link modal routes recruiters to template creation when no template exists", async ({
  page,
  baseURL,
}) => {
  await mockInterviewShell(page);
  await authenticatePage(page, setup);

  await page.goto(`${baseURL}/interviews`);
  await page.getByRole("button", { name: "New interview link" }).click();

  await expect(page.getByText("No active interview templates")).toBeVisible();

  const createTemplateButton = page.getByRole("button", { name: "Create interview template" });
  await expect(createTemplateButton).toBeVisible();
  await createTemplateButton.click();

  await expect(page).toHaveURL(`${baseURL}/interviews/templates?create=1`);
  await expect(page.getByRole("heading", { name: "Create Interview Template" })).toBeVisible();
});
