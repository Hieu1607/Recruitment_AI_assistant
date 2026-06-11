import { expect, test } from "@playwright/test";

import { authenticatePage, seedWorkspace } from "./helpers";

test("Vietnamese UI is shown on login and recruiter workspace pages", async ({ page, request, baseURL }) => {
  await page.goto(`${baseURL}/login`);
  await expect(page.getByRole("heading", { name: "Chào mừng bạn quay lại" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Đăng nhập với Google" })).toBeVisible();

  const setup = await seedWorkspace(request, "Localization Smoke", [
    {
      fullName: "Linh Nguyen",
      email: "linh.nguyen@example.com",
      lines: [
        "Linh Nguyen",
        "Senior Recruiter",
        "Candidate outreach and talent pipeline management",
      ],
    },
  ]);

  await authenticatePage(page, setup);
  await page.goto(`${baseURL}/candidates`);

  await expect(page.getByRole("link", { name: "Ứng viên" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Ứng viên" })).toBeVisible();
  await expect(page.getByRole("complementary").getByRole("button", { name: "Tải lên hồ sơ" })).toBeVisible();
});
