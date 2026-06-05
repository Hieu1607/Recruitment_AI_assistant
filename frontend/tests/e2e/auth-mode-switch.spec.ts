import { expect, test } from "@playwright/test";

test("auth mode switch preserves entered values", async ({ page, baseURL }) => {
  await page.goto(`${baseURL}/login?mode=signup`);

  await page.getByLabel("Full name").fill("Jane Doe");
  await page.getByLabel("Email address").fill("jane@company.com");
  await page.getByLabel("Password").fill("super-secret");

  await page.getByRole("link", { name: "Sign in", exact: true }).click();

  await expect(page).toHaveURL(`${baseURL}/login`);
  await expect(page.getByLabel("Email address")).toHaveValue("jane@company.com");
  await expect(page.getByLabel("Password")).toHaveValue("super-secret");

  await page.getByRole("link", { name: "Sign up", exact: true }).click();

  await expect(page).toHaveURL(`${baseURL}/login?mode=signup`);
  await expect(page.getByLabel("Full name")).toHaveValue("Jane Doe");
  await expect(page.getByLabel("Email address")).toHaveValue("jane@company.com");
  await expect(page.getByLabel("Password")).toHaveValue("super-secret");
});
