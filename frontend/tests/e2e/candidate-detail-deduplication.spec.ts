import { expect, test } from "@playwright/test";

import { authenticatePage } from "./helpers";

test("candidate detail overview does not repeat the same personal-info value inside a section card", async ({
  page,
  baseURL,
}) => {
  const setup = {
    accessToken: "playwright-token",
    jobId: "job-candidate-detail-1",
    publicApplyToken: "unused",
  };
  const resumeId = "resume-candidate-detail-1";
  const profileId = "profile-candidate-detail-1";

  await page.route("**/api/v1/auth/me", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: "user-candidate-detail-1",
        email: "candidate-detail@example.com",
        display_name: "Candidate Detail Tester",
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
            title: "Candidate Detail",
            status: "active",
            created_at: "2026-06-12T08:00:00Z",
            updated_at: "2026-06-12T08:00:00Z",
          },
        ],
        total: 1,
      }),
    });
  });

  await page.route(`**/api/v1/upload/${resumeId}`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: resumeId,
        original_file_name: "tran_thi_bich_lan.pdf",
        storage_uri: "minio://resumes/tran_thi_bich_lan.pdf",
        upload_status: "processed",
        extraction_mode: "text",
        duplicate_group_key: null,
        uploaded_by_user_id: "user-candidate-detail-1",
        uploaded_at: "2026-06-12T08:00:00Z",
        processed_at: "2026-06-12T08:05:00Z",
        retention_expires_at: "2026-07-12T08:00:00Z",
      }),
    });
  });

  await page.route(`**/api/v1/upload/${resumeId}/profile`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: profileId,
        resume_document_id: resumeId,
        extraction_mode: "text",
        full_name: "Trần Thị Bích Lan",
        submitted_full_name: "Trần Thị Bích Lan",
        phone: "+84 92 3456 789",
        email: "lan.bich@example.com",
        submitted_email: "lan.bich@example.com",
        location_normalized: "Hồ Chí Minh",
        contact: null,
        current_job_title: "Marketing Director at VinGroup",
        graduation_status: "graduated",
        ever_studied_abroad: false,
        major: "Business Administration",
        cpa: null,
        summary_text:
          "Strategic marketing leader with a decade of experience driving growth for a major conglomerate.",
        skills_text: "Digital Marketing\nBrand Management",
        experience_text: "Marketing Director at VinGroup",
        experience_years: 10,
        education_text: "Master of Business Administration",
        languages_text: "Tiếng Việt\nTiếng Anh",
        projects_text: "VinMart e-commerce platform launch",
        achievements_text:
          "VinMart ranked #1 in online retail sales in 2021\nReceived VinGroup's Top Executive Award 2022",
        publications_text: null,
        certifications_text: "Certified Digital Marketing Professional (CDMP)",
        references_text: "Ms. Phạm Đình Thúy",
        other_text: null,
        structured_profile: {
          summary: {
            text:
              "Strategic marketing leader with a decade of experience driving growth for a major conglomerate.",
            links: [],
          },
          skills: {
            rawText: "Digital Marketing\nBrand Management",
            entries: [
              {
                title: "Digital Marketing",
                subtitle: null,
                role: null,
                location: null,
                dateRange: null,
                description: null,
                bullets: ["Digital Marketing"],
                links: [],
                metadata: [],
              },
              {
                title: "Brand Management",
                subtitle: null,
                role: null,
                location: null,
                dateRange: null,
                description: null,
                bullets: ["Brand Management"],
                links: [],
                metadata: [],
              },
            ],
          },
          languages: {
            rawText: "Tiếng Việt\nTiếng Anh",
            entries: [
              {
                title: "Tiếng Việt",
                subtitle: null,
                role: null,
                location: null,
                dateRange: null,
                description: null,
                bullets: ["Tiếng Việt"],
                links: [],
                metadata: [],
              },
              {
                title: "Tiếng Anh",
                subtitle: null,
                role: null,
                location: null,
                dateRange: null,
                description: null,
                bullets: ["Tiếng Anh"],
                links: [],
                metadata: [],
              },
            ],
          },
          achievements: {
            rawText: null,
            entries: [
              {
                title: "VinMart ranked #1 in online retail sales in 2021",
                subtitle: null,
                role: null,
                location: null,
                dateRange: null,
                description: null,
                bullets: [
                  "VinMart ranked #1 in online retail sales in 2021",
                ],
                links: [],
                metadata: [],
              },
            ],
          },
        },
      }),
    });
  });

  await page.route("**/api/v1/outreach/**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ items: [], total: 0 }),
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

  await expect(page.getByRole("heading", { name: "Trần Thị Bích Lan" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Skills" })).toBeVisible();

  const digitalMarketingCard = page
    .locator("article")
    .filter({ has: page.getByRole("heading", { name: "Digital Marketing" }) });
  await expect(digitalMarketingCard).toHaveCount(1);
  await expect(digitalMarketingCard.locator("li")).toHaveCount(0);

  const vietnameseCard = page
    .locator("article")
    .filter({ has: page.getByRole("heading", { name: "Tiếng Việt" }) });
  await expect(vietnameseCard).toHaveCount(1);
  await expect(vietnameseCard.locator("li")).toHaveCount(0);

  const achievementCard = page
    .locator("article")
    .filter({ has: page.getByRole("heading", { name: "VinMart ranked #1 in online retail sales in 2021" }) });
  await expect(achievementCard).toHaveCount(1);
  await expect(achievementCard.locator("li")).toHaveCount(0);
});
