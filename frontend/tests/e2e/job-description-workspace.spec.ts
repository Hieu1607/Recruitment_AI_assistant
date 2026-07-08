import { expect, test } from "@playwright/test";

import { authenticatePage, seedWorkspace } from "./helpers";

test("job description management follows the selected workspace", async ({ page, request, baseURL }) => {
  const setup = await seedWorkspace(request, "Workspace JD Flow", [
    {
      fullName: "Jamie Workspace",
      email: "jamie.workspace@example.com",
      lines: [
        "Jamie Workspace",
        "Frontend Engineer",
        "React Typescript Accessibility",
      ],
    },
  ]);

  await authenticatePage(page, setup);

  await page.goto(`${baseURL}/jobs`);
  const editJobDescriptionButton = page.getByRole("button", { name: "Edit job description" }).first();
  await expect(editJobDescriptionButton).toBeVisible();

  await editJobDescriptionButton.click();
  await expect(page).toHaveURL(`${baseURL}/job-descriptions`);
  await expect(page.getByRole("heading", { name: "Workspace job description" })).toBeVisible();

  await page.goto(`${baseURL}/job-descriptions/${setup.jobDescriptionId}/edit`);
  await expect(page).toHaveURL(`${baseURL}/job-descriptions`);
  await expect(page.getByRole("heading", { name: "Workspace job description" })).toBeVisible();

  await page.goto(`${baseURL}/jobs/new`);
  await expect(page.getByRole("heading", { name: "New job" })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "Job title" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Job description" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Bold" })).toBeVisible();
  await expect(
    page.getByText("Write the full JD here while you create the workspace so scoring and AI workflows are ready from the start."),
  ).toBeVisible();

  await page.goto(`${baseURL}/job-descriptions/new`);
  await expect(page).toHaveURL(`${baseURL}/job-descriptions`);
  await expect(page.getByRole("heading", { name: "Workspace job description" })).toBeVisible();

  await page.goto(`${baseURL}/job-descriptions`);
  await expect(page.getByRole("heading", { name: "Workspace job description" })).toBeVisible();
  await expect(page.getByText("This page follows the selected workspace.")).toBeVisible();
});
