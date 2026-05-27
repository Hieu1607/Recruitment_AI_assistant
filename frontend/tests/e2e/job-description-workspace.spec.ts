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
  await expect(page).toHaveURL(`${baseURL}/job-descriptions/new`);
  await expect(page.getByRole("button", { name: "Back to workspace" })).toBeVisible();

  await page.goto(`${baseURL}/job-descriptions/${setup.jobDescriptionId}/edit`);
  await expect(page).toHaveURL(`${baseURL}/job-descriptions/new`);
  await expect(page.getByText("This editor follows the currently selected workspace.")).toBeVisible();

  await page.goto(`${baseURL}/job-descriptions`);
  await expect(page.getByRole("heading", { name: "Workspace job description" })).toBeVisible();
  await expect(page.getByText("This page follows the selected workspace.")).toBeVisible();
});
