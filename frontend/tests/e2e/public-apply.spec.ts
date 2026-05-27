import { expect, test } from "@playwright/test";

import { seedWorkspace } from "./helpers";

test("public apply submits a resume and shows success state", async ({ page, request, baseURL }) => {
  const setup = await seedWorkspace(request, "Public Apply Smoke", []);

  await page.goto(`${baseURL}/apply/${setup.publicApplyToken}`);

  await expect(page.getByRole("heading", { name: "Public Apply Smoke" })).toBeVisible();
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
