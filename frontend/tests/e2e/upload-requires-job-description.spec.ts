import { expect, test } from "@playwright/test";

import {
  authenticatePage,
  createWorkspaceWithoutJobDescription,
} from "./helpers";

test("upload prompts for a job description before opening resume upload", async ({
  page,
  request,
  baseURL,
}) => {
  const setup = await createWorkspaceWithoutJobDescription(
    request,
    "Upload Gate Workspace",
  );

  await authenticatePage(page, setup);

  await page.goto(`${baseURL}/dashboard`);
  await page.getByRole("button", { name: "Upload resumes" }).click();
  await expect(
    page.getByRole("heading", { name: "Add a job description first" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Go to job description" }),
  ).toBeVisible();

  await page.getByRole("button", { name: "Cancel" }).click();
  await expect(
    page.getByRole("heading", { name: "Add a job description first" }),
  ).toBeHidden();

  await page.goto(`${baseURL}/candidates`);
  await page.getByRole("button", { name: "Upload resumes" }).first().click();
  await expect(
    page.getByRole("heading", { name: "Add a job description first" }),
  ).toBeVisible();

  await page.getByRole("button", { name: "Go to job description" }).click();
  await expect(page).toHaveURL(`${baseURL}/job-descriptions`);

  await page.getByRole("button", { name: "Upload resume" }).click();
  await expect(
    page.getByRole("heading", { name: "Add a job description first" }),
  ).toBeVisible();
});
