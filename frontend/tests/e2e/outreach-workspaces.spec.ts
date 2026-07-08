import { expect, test, type Page } from "@playwright/test";

import { authenticatePage } from "./helpers";

const setup = {
  accessToken: "playwright-outreach-token",
  jobId: "job-outreach-1",
  publicApplyToken: "unused",
};

const recruiter = {
  id: "user-outreach-1",
  email: "outreach@example.com",
  display_name: "Outreach Tester",
  gmail_connected: true,
};

const job = {
  id: setup.jobId,
  owner_user_id: recruiter.id,
  title: "Outreach Job",
  status: "active",
  public_apply_enabled: true,
  public_apply_url: "http://127.0.0.1:5173/apply/outreach-job",
  created_at: "2026-07-08T08:00:00Z",
  updated_at: "2026-07-08T08:00:00Z",
  archived_at: null,
};

const candidate = {
  id: "candidate-outreach-1",
  resume_document_id: "resume-outreach-1",
  full_name: "Taylor Outreach",
  phone: null,
  email: "taylor@example.com",
  location_normalized: "Ho Chi Minh City",
  current_job_title: "Recruiting Coordinator",
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

const template = {
  id: "template-outreach-1",
  created_by_user_id: recruiter.id,
  job_id: setup.jobId,
  name: "Warm intro",
  content_source: "template",
  subject_template: "Warm intro subject",
  body_text_template: "Hi {{candidate_name}}, let's discuss {{job_title}}.",
  body_html_template: "<p>Hi {{candidate_name}}, let's discuss {{job_title}}.</p>",
  editor_json: null,
  variables_used: ["candidate_name", "job_title"],
  created_at: "2026-07-08T08:00:00Z",
  updated_at: "2026-07-08T08:00:00Z",
};

async function mockOutreachShell(page: Page) {
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

    if (path === `/jobs/${setup.jobId}/candidates` && method === "GET") {
      await json({ items: [candidate], total: 1 });
      return;
    }

    if (path === "/outreach/" && method === "GET") {
      await json({ items: [], total: 0 });
      return;
    }

    if (path === "/outreach/templates" && method === "GET") {
      await json({ items: [template], total: 1 });
      return;
    }

    await route.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({ detail: `Unhandled mock route: ${method} ${path}` }),
    });
  });
}

test("outreach messages workspace removes template-authoring and AI controls from new message", async ({
  page,
  baseURL,
}) => {
  await mockOutreachShell(page);
  await authenticatePage(page, setup);

  await page.goto(`${baseURL}/outreach`);
  await page.getByRole("button", { name: "+ New message" }).click();

  await expect(page.getByText("Save as template")).toHaveCount(0);
  await expect(page.getByText("Generate once")).toHaveCount(0);
});

test("outreach messages workspace can preload a selected template into the draft editor", async ({
  page,
  baseURL,
}) => {
  await mockOutreachShell(page);
  await authenticatePage(page, setup);

  await page.goto(`${baseURL}/outreach`);
  await page.getByRole("button", { name: "+ New message" }).click();
  await page.getByRole("button", { name: "Use template" }).click();
  await page.getByRole("combobox").nth(1).selectOption(template.id);

  await expect(page.getByPlaceholder("Subject line…")).toHaveValue("Warm intro subject");
  await expect(page.getByText("Hi {{candidate_name}}, let's discuss {{job_title}}.")).toBeVisible();
});

test("outreach templates workspace exposes dedicated AI draft controls", async ({ page, baseURL }) => {
  await mockOutreachShell(page);
  await authenticatePage(page, setup);

  await page.goto(`${baseURL}/outreach/templates`);

  const newTemplateButton = page.getByRole("button", { name: "New template" });
  await expect(newTemplateButton).toBeVisible();
  await newTemplateButton.click();
  await expect(page.getByText("Generate once")).toBeVisible();
});
