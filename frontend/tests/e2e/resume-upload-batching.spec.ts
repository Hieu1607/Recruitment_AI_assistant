import { expect, test } from "@playwright/test";

import { authenticatePage, seedWorkspace } from "./helpers";

function buildPdf(name: string) {
  return {
    name,
    mimeType: "application/pdf",
    buffer: Buffer.from(
      "%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n4 0 obj\n<< /Length 59 >>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Batch upload test) Tj\nET\nendstream\nendobj\n5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000241 00000 n \n0000000349 00000 n \ntrailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n419\n%%EOF\n",
      "utf8",
    ),
  };
}

test("uploading multiple resumes sends one batch request", async ({ page, request, baseURL }) => {
  const setup = await seedWorkspace(request, "Resume Upload Batching", []);
  await authenticatePage(page, setup);

  let uploadRequestCount = 0;
  let uploadedFileNames: string[] = [];

  await page.route(`**/api/v1/jobs/${setup.jobId}/resumes`, async (route) => {
    if (route.request().method() !== "POST") {
      await route.continue();
      return;
    }

    const postData = route.request().postDataBuffer() ?? Buffer.alloc(0);
    const matches = postData.toString("utf8").match(/filename="([^"]+)"/g) ?? [];
    uploadedFileNames = uploadedFileNames.concat(
      matches.map((match) => match.replace(/^filename="/, "").replace(/"$/, "")),
    );
    uploadRequestCount += 1;

    await route.fulfill({
      status: 202,
      contentType: "application/json",
      body: JSON.stringify({
        total_files: 5,
        queued_files: 5,
        items: Array.from({ length: 5 }, (_, index) => ({
          file_name: `resume-${index + 1}.pdf`,
          resume_document_id: `resume-${index + 1}`,
          status: "queued",
          task_id: `task-${index + 1}`,
        })),
      }),
    });
  });

  await page.goto(`${baseURL}/candidates`);
  await page.getByRole("button", { name: "Upload resumes" }).first().click();

  await page.locator('input[type="file"]').setInputFiles([
    buildPdf("resume-1.pdf"),
    buildPdf("resume-2.pdf"),
    buildPdf("resume-3.pdf"),
    buildPdf("resume-4.pdf"),
    buildPdf("resume-5.pdf"),
  ]);
  await page.getByRole("button", { name: "Upload 5 resumes" }).click();

  await expect(page.getByRole("heading", { name: "5 resumes queued" })).toBeVisible();
  await expect.poll(() => uploadRequestCount).toBe(1);
  await expect(uploadedFileNames).toEqual([
    "resume-1.pdf",
    "resume-2.pdf",
    "resume-3.pdf",
    "resume-4.pdf",
    "resume-5.pdf",
  ]);
});
