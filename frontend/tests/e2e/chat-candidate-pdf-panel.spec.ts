import { expect, test } from "@playwright/test";

import { authenticatePage } from "./helpers";

test("clicking a candidate card keeps chat open and shows the resume PDF in a right panel", async ({
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
  await page.goto(`${baseURL}/chat/${sessionId}`);

  await page.getByRole("button", { name: /Nguyen Minh Hieu/i }).click();

  await expect(page).toHaveURL(new RegExp(`/chat/${sessionId}$`));
  await expect(page.getByTestId("chat-candidate-pdf-panel")).toBeVisible();
  await expect(page.getByText("Nguyen Minh Hieu")).toBeVisible();
  await expect(page.locator('[data-testid="chat-candidate-pdf-panel"] iframe')).toBeVisible();

  await page.getByRole("button", { name: "Close candidate resume preview" }).click();
  await expect(page.getByTestId("chat-candidate-pdf-panel")).toHaveCount(0);
  await expect(page).toHaveURL(new RegExp(`/chat/${sessionId}$`));
});
