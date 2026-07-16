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

const outreachMessage = {
  id: "message-outreach-1",
  candidate_profile_id: candidate.id,
  candidate_full_name: candidate.full_name,
  created_by_user_id: recruiter.id,
  content_source: "template",
  subject: "Warm intro subject",
  body_text: "Hi Taylor Outreach, let's discuss Outreach Job.",
  body_html: "<p>Hi Taylor Outreach, let's discuss Outreach Job.</p>",
  template_id: template.id,
  render_variables: { candidate_name: candidate.full_name, job_title: job.title },
  sent_status: "not_sent",
  sent_at: null,
  created_at: "2026-07-08T08:00:00Z",
};

async function mockOutreachShell(
  page: Page,
  options: {
    messages?: typeof outreachMessage[];
    onBulkSend?: (messageIds: string[]) => void;
  } = {},
) {
  const messages = options.messages ?? [];

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
      const sentStatus = url.searchParams.get("sent_status");
      const filteredMessages = sentStatus
        ? messages.filter((message) => message.sent_status === sentStatus)
        : messages;
      await json({ items: filteredMessages, total: filteredMessages.length });
      return;
    }

    if (path === "/outreach/bulk-send" && method === "POST") {
      const payload = route.request().postDataJSON() as { message_ids: string[] };
      options.onBulkSend?.(payload.message_ids);
      await json({
        queued_count: payload.message_ids.length,
        skipped_count: 0,
        failed_count: 0,
        results: payload.message_ids.map((messageId) => ({
          message_id: messageId,
          status: "queued",
          reason: null,
        })),
      }, 202);
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

test("outreach not-sent folder can select and bulk send existing drafts", async ({ page, baseURL }) => {
  let requestedMessageIds: string[] = [];
  await mockOutreachShell(page, {
    messages: [outreachMessage],
    onBulkSend: (messageIds) => {
      requestedMessageIds = messageIds;
    },
  });
  await authenticatePage(page, setup);

  await page.goto(`${baseURL}/outreach?folder=not_sent`);
  await page.getByRole("button", { name: "Select", exact: true }).click();
  await page.getByLabel(`Select outreach message for ${candidate.full_name}`).check();
  await page.getByRole("button", { name: "Send 1", exact: true }).click();

  await expect(page.getByText("Send 1 outreach messages?")).toBeVisible();
  await page.getByRole("button", { name: "Send 1 messages", exact: true }).click();

  await expect.poll(() => requestedMessageIds).toEqual([outreachMessage.id]);
  await expect(page.getByText("1 queued, 0 skipped, 0 failed")).toBeVisible();
});
