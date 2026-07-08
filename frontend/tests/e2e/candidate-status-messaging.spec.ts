import { expect, test, type Page } from "@playwright/test";

import { authenticatePage } from "./helpers";

const setup = {
  accessToken: "playwright-token",
  jobId: "job-candidate-status-1",
  publicApplyToken: "unused",
};

async function mockAuthenticatedShell(page: Page) {
  await page.route("**/api/v1/auth/me", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: "user-candidate-status-1",
        email: "candidate-status@example.com",
        display_name: "Candidate Status Tester",
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
            title: "Candidate Status",
            status: "active",
            created_at: "2026-07-07T08:00:00Z",
            updated_at: "2026-07-07T08:00:00Z",
          },
        ],
        total: 1,
      }),
    });
  });
}

test("candidate list shows processing and failed CV guidance in table and grid views", async ({
  page,
  baseURL,
}) => {
  await mockAuthenticatedShell(page);

  await page.route(`**/api/v1/jobs/${setup.jobId}/resumes**`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        items: [
          {
            id: "resume-processing-1",
            original_file_name: "tran_van_a.pdf",
            candidate_display_name: "Tran Van A",
            storage_uri: "minio://resumes/tran_van_a.pdf",
            upload_status: "processing",
            duplicate_group_key: null,
            uploaded_by_user_id: "user-candidate-status-1",
            uploader_display_name: "Recruiter A",
            uploaded_at: "2026-07-07T08:00:00Z",
            processed_at: null,
            retention_expires_at: null,
          },
          {
            id: "resume-failed-1",
            original_file_name: "le_thi_b.pdf",
            candidate_display_name: "Le Thi B",
            storage_uri: "minio://resumes/le_thi_b.pdf",
            upload_status: "failed",
            duplicate_group_key: null,
            uploaded_by_user_id: "user-candidate-status-1",
            uploader_display_name: "Recruiter B",
            uploaded_at: "2026-07-07T09:00:00Z",
            processed_at: null,
            retention_expires_at: null,
          },
        ],
        total: 2,
      }),
    });
  });

  await authenticatePage(page, setup);
  await page.goto(`${baseURL}/candidates`);

  await expect(page.getByRole("heading", { name: "Candidates" })).toBeVisible();
  await expect(page.getByText("CV is being processed")).toBeVisible();
  await expect(page.getByText("CV processing failed")).toBeVisible();

  await page.getByRole("button", { name: "Grid view" }).click();

  await expect(page.getByText("CV is being processed")).toBeVisible();
  await expect(page.getByText("CV processing failed")).toBeVisible();
});

test("candidate detail shows processing guidance inside overview fields", async ({
  page,
  baseURL,
}) => {
  const resumeId = "resume-processing-detail-1";

  await mockAuthenticatedShell(page);

  await page.route(`**/api/v1/upload/${resumeId}`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: resumeId,
        original_file_name: "pham_thi_c.pdf",
        storage_uri: "minio://resumes/pham_thi_c.pdf",
        upload_status: "processing",
        duplicate_group_key: null,
        uploaded_by_user_id: "user-candidate-status-1",
        uploaded_at: "2026-07-07T10:00:00Z",
        processed_at: null,
        retention_expires_at: "2026-08-07T10:00:00Z",
      }),
    });
  });

  await page.route(`**/api/v1/jobs/${setup.jobId}/interview-invitations`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ items: [], total: 0 }),
    });
  });

  await authenticatePage(page, setup);
  await page.goto(`${baseURL}/candidates/${resumeId}`);

  const nameSection = page.locator("section").filter({ has: page.getByRole("heading", { name: "Name" }) });
  const skillsSection = page.locator("section").filter({ has: page.getByRole("heading", { name: "Skills" }) });
  const resumeInfo = page.locator("aside").filter({ has: page.getByRole("heading", { name: "Resume Info" }) });

  await expect(nameSection.getByText("CV is being processed")).toBeVisible();
  await expect(skillsSection.getByText("CV is being processed")).toBeVisible();
  await expect(resumeInfo.getByText("CV is being processed").first()).toBeVisible();

  await page.getByRole("button", { name: "Resume PDF" }).click();
  await expect(page.getByText("CV is being processed")).toBeVisible();
});

test("candidate detail shows failed guidance inside overview fields", async ({
  page,
  baseURL,
}) => {
  const resumeId = "resume-failed-detail-1";

  await mockAuthenticatedShell(page);

  await page.route(`**/api/v1/upload/${resumeId}`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: resumeId,
        original_file_name: "ngo_van_d.pdf",
        storage_uri: "minio://resumes/ngo_van_d.pdf",
        upload_status: "failed",
        duplicate_group_key: null,
        uploaded_by_user_id: "user-candidate-status-1",
        uploaded_at: "2026-07-07T11:00:00Z",
        processed_at: null,
        retention_expires_at: "2026-08-07T11:00:00Z",
      }),
    });
  });

  await page.route(`**/api/v1/jobs/${setup.jobId}/interview-invitations`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ items: [], total: 0 }),
    });
  });

  await authenticatePage(page, setup);
  await page.goto(`${baseURL}/candidates/${resumeId}`);

  const nameSection = page.locator("section").filter({ has: page.getByRole("heading", { name: "Name" }) });
  const skillsSection = page.locator("section").filter({ has: page.getByRole("heading", { name: "Skills" }) });
  const resumeInfo = page.locator("aside").filter({ has: page.getByRole("heading", { name: "Resume Info" }) });

  await expect(nameSection.getByText("CV processing failed")).toBeVisible();
  await expect(skillsSection.getByText("CV processing failed")).toBeVisible();
  await expect(resumeInfo.getByText("CV processing failed").first()).toBeVisible();

  await page.getByRole("button", { name: "Resume PDF" }).click();
  await expect(page.getByText("CV processing failed")).toBeVisible();
});
