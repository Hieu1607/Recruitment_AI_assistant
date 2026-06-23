import { expect, test } from "@playwright/test";

import { authenticatePage } from "./helpers";

async function dragHandle(page: Parameters<typeof test>[0]["page"], testId: string, deltaX: number) {
  const handle = page.getByTestId(testId);
  const box = await handle.boundingBox();
  expect(box).toBeTruthy();
  if (!box) return;

  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + deltaX, box.y + box.height / 2, { steps: 10 });
  await page.mouse.up();
}

test("clicking a candidate card keeps chat open and shows a resizable resume panel", async ({
  page,
  baseURL,
}) => {
  const setup = {
    accessToken: "playwright-token",
    jobId: "job-chat-pdf-1",
    publicApplyToken: "unused",
  };
  const sessionId = "session-chat-pdf-1";
  const candidateId = "candidate-chat-pdf-1";
  const resumeId = "resume-chat-pdf-1";

  await page.route("**/api/v1/auth/me", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: "user-chat-pdf-1",
        email: "pdf@example.com",
        display_name: "PDF Tester",
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
            title: "Chat Resume Preview",
            status: "active",
            created_at: "2026-06-07T08:00:00Z",
            updated_at: "2026-06-07T08:00:00Z",
          },
        ],
        total: 1,
      }),
    });
  });

  await page.route(`**/api/v1/jobs/${setup.jobId}/chat/sessions`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        items: [
          {
            id: sessionId,
            job_id: setup.jobId,
            session_title: "Candidate PDF preview",
            created_at: "2026-06-07T08:00:00Z",
            updated_at: "2026-06-07T08:30:00Z",
          },
        ],
        total: 1,
      }),
    });
  });

  await page.route(`**/api/v1/jobs/${setup.jobId}/chat/sessions/${sessionId}/turns**`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([
        {
          id: "turn-chat-pdf-1",
          user_question: "Show me the candidate in scope",
          answer_text: "Here is the strongest match.",
          matched_count: 1,
          matched_candidate_ids: [candidateId],
        },
      ]),
    });
  });

  await page.route(`**/api/v1/jobs/${setup.jobId}/candidates`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        items: [
          {
            id: candidateId,
            resume_document_id: resumeId,
            extraction_mode: "pymupdf",
            full_name: "Nguyen Minh Hieu",
            submitted_full_name: "Nguyen Minh Hieu",
            phone: null,
            email: "hieu@example.com",
            submitted_email: "hieu@example.com",
            location_normalized: null,
            contact: null,
            current_job_title: "Data Scientist",
            graduation_status: "unknown",
            ever_studied_abroad: false,
            major: null,
            cpa: null,
            summary_text: null,
            skills_text: null,
            experience_text: null,
            experience_years: null,
            education_text: null,
            languages_text: null,
            projects_text: null,
            achievements_text: null,
            publications_text: null,
            certifications_text: null,
            references_text: null,
            other_text: null,
            structured_profile: null,
          },
        ],
        total: 1,
      }),
    });
  });

  await page.route(`**/api/v1/upload/${resumeId}/file`, async (route) => {
    const pdf = Buffer.from(
      "%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000063 00000 n \n0000000122 00000 n \ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n192\n%%EOF\n",
      "utf8",
    );
    await route.fulfill({
      status: 200,
      contentType: "application/pdf",
      body: pdf,
    });
  });

  await authenticatePage(page, setup);
  await page.setViewportSize({ width: 1440, height: 960 });
  await page.addInitScript(() => {
    localStorage.removeItem("easyhr.chat-candidate-preview-sidebar");
  });
  await page.goto(`${baseURL}/chat/${sessionId}`);

  const chatMainPanel = page.getByTestId("chat-main-panel");
  const chatPanelWidthBefore = await chatMainPanel.evaluate((node) => node.getBoundingClientRect().width);

  await page.getByRole("button", { name: /Nguyen Minh Hieu/i }).click();

  await expect(page).toHaveURL(new RegExp(`/chat/${sessionId}$`));
  await expect(page.getByTestId("chat-candidate-pdf-panel")).toBeVisible();
  await expect(page.getByText("Nguyen Minh Hieu")).toBeVisible();
  await expect(page.locator('[data-testid="chat-candidate-pdf-panel"] iframe')).toBeVisible();
  await expect(page.getByTestId("chat-candidate-pdf-resize-handle")).toBeVisible();

  const resumePanel = page.getByTestId("chat-candidate-pdf-panel");
  const panelWidthBeforeResize = await resumePanel.evaluate((node) => node.getBoundingClientRect().width);
  const chatPanelWidthAfterOpen = await chatMainPanel.evaluate((node) => node.getBoundingClientRect().width);

  expect(chatPanelWidthAfterOpen).toBeLessThan(chatPanelWidthBefore - 120);

  await dragHandle(page, "chat-candidate-pdf-resize-handle", -160);

  const panelWidthAfterResize = await resumePanel.evaluate((node) => node.getBoundingClientRect().width);
  const chatPanelWidthAfterResize = await chatMainPanel.evaluate((node) => node.getBoundingClientRect().width);

  expect(panelWidthAfterResize).toBeGreaterThan(panelWidthBeforeResize + 120);
  expect(chatPanelWidthAfterResize).toBeLessThan(chatPanelWidthAfterOpen - 120);

  await page.getByRole("button", { name: "Close candidate resume preview" }).click();
  await expect(page.getByTestId("chat-candidate-pdf-panel")).toHaveCount(0);
  await expect(page).toHaveURL(new RegExp(`/chat/${sessionId}$`));
});

test("resume panel falls back to extracted CV text when the PDF cannot be loaded", async ({
  page,
  baseURL,
}) => {
  const setup = {
    accessToken: "playwright-token",
    jobId: "job-chat-pdf-fallback-1",
    publicApplyToken: "unused",
  };
  const sessionId = "session-chat-pdf-fallback-1";
  const candidateId = "candidate-chat-pdf-fallback-1";
  const resumeId = "resume-chat-pdf-fallback-1";

  await page.route("**/api/v1/auth/me", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: "user-chat-pdf-fallback-1",
        email: "fallback@example.com",
        display_name: "Fallback Tester",
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
            title: "Chat Resume Fallback",
            status: "active",
            created_at: "2026-06-07T08:00:00Z",
            updated_at: "2026-06-07T08:00:00Z",
          },
        ],
        total: 1,
      }),
    });
  });

  await page.route(`**/api/v1/jobs/${setup.jobId}/chat/sessions`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        items: [
          {
            id: sessionId,
            job_id: setup.jobId,
            session_title: "Candidate PDF fallback",
            created_at: "2026-06-07T08:00:00Z",
            updated_at: "2026-06-07T08:30:00Z",
          },
        ],
        total: 1,
      }),
    });
  });

  await page.route(`**/api/v1/jobs/${setup.jobId}/chat/sessions/${sessionId}/turns**`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([
        {
          id: "turn-chat-pdf-fallback-1",
          user_question: "Show me the candidate in scope",
          answer_text: "The extracted resume text is still available.",
          matched_count: 1,
          matched_candidate_ids: [candidateId],
        },
      ]),
    });
  });

  await page.route(`**/api/v1/jobs/${setup.jobId}/candidates`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        items: [
          {
            id: candidateId,
            resume_document_id: resumeId,
            extraction_mode: "pymupdf",
            full_name: "Nguyen Van An",
            submitted_full_name: "Nguyen Van An",
            phone: null,
            email: "an@example.com",
            submitted_email: "an@example.com",
            location_normalized: "Ha Noi",
            contact: null,
            current_job_title: "Senior Software Engineer",
            graduation_status: "graduated",
            ever_studied_abroad: false,
            major: "Computer Science",
            cpa: null,
            summary_text: "Senior backend engineer focused on Python and platform reliability.",
            skills_text: "Python, FastAPI, PostgreSQL, Redis",
            experience_text: "FPT Software (2019-2026)\nBuilt hiring workflow automation and scoring pipelines.",
            experience_years: 7,
            education_text: "Hanoi University of Science and Technology",
            languages_text: "Vietnamese, English",
            projects_text: "Recruitment AI Assistant",
            achievements_text: null,
            publications_text: null,
            certifications_text: null,
            references_text: null,
            other_text: null,
            structured_profile: null,
          },
        ],
        total: 1,
      }),
    });
  });

  await page.route(`**/api/v1/upload/${resumeId}/file`, async (route) => {
    await route.fulfill({
      status: 502,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Failed to load resume file" }),
    });
  });

  await authenticatePage(page, setup);
  await page.goto(`${baseURL}/chat/${sessionId}`);

  await page.getByRole("button", { name: /Nguyen Van An/i }).click();

  await expect(page.getByTestId("chat-candidate-pdf-panel")).toBeVisible();
  await expect(page.getByText("Showing extracted CV text because the PDF preview is unavailable.")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Summary" })).toBeVisible();
  await expect(page.getByText("Senior backend engineer focused on Python and platform reliability.")).toBeVisible();
  await expect(page.getByText("Python, FastAPI, PostgreSQL, Redis")).toBeVisible();
  await expect(page.getByText("FPT Software (2019-2026)")).toBeVisible();
});
