import { expect, test } from "@playwright/test";

import { seedWorkspace } from "./helpers";

test("public apply submits a resume and shows success state", async ({ page, request, baseURL }) => {
  const setup = await seedWorkspace(request, "Public Apply Smoke", []);
  const markdownJd = [
    "# Job Description - AI Engineer",
    "",
    "## Position",
    "",
    "**AI Engineer (Fresher / Junior)**",
    "",
    "## Responsibilities",
    "",
    "- Design and develop AI-powered applications using Large Language Models (LLMs).",
  ].join("\n");

  const updateJdResponse = await request.patch(
    `${process.env.E2E_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1"}/jobs/${setup.jobId}/job-description`,
    {
      headers: { Authorization: `Bearer ${setup.accessToken}` },
      data: {
        title: "AI Engineer JD",
        jd_text: markdownJd,
      },
    },
  );
  expect(updateJdResponse.ok()).toBeTruthy();

  await page.goto(`${baseURL}/apply/${setup.publicApplyToken}`);

  await expect(page.getByRole("heading", { name: "Public Apply Smoke" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Job description", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Job Description - AI Engineer" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Position" })).toBeVisible();
  await expect(page.getByText("AI Engineer (Fresher / Junior)")).toBeVisible();
  await expect(page.getByText("# Job Description - AI Engineer")).toHaveCount(0);
  await expect(page.getByText("**AI Engineer (Fresher / Junior)**")).toHaveCount(0);
  await expect(page.getByText("- Design and develop AI-powered applications", { exact: false })).toHaveCount(0);
  await page.getByRole("textbox", { name: "Full name" }).fill("Public Browser Candidate");
  await page.getByRole("textbox", { name: "Email" }).fill("public.browser@example.com");
  await page.locator("#resume-file").setInputFiles({
    name: "public-browser.pdf",
    mimeType: "application/pdf",
    buffer: Buffer.from(
      "%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n4 0 obj\n<< /Length 68 >>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Public Browser Candidate) Tj\nET\nendstream\nendobj\n5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000241 00000 n \n0000000359 00000 n \ntrailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n429\n%%EOF\n",
      "utf8",
    ),
  });
  await page.getByRole("button", { name: "Submit resume" }).click();

  await expect(page.getByRole("heading", { name: "Resume submitted" })).toBeVisible();
});
