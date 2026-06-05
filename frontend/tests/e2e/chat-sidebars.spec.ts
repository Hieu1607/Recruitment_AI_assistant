import { expect, test } from "@playwright/test";

import { authenticatePage } from "./helpers";

async function dragHandle(page: Parameters<typeof test>[0]["page"], testId: string, deltaX: number) {
  const handle = page.getByTestId(testId);
  const box = await handle.boundingBox();
  expect(box).toBeTruthy();
  if (!box) return;

  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + deltaX, box.y + box.height / 2, { steps: 10 });
  await page.mouse.up();
}

test("chat page keeps navigation fixed while conversation history can resize", async ({ page, baseURL }) => {
  const setup = {
    accessToken: "playwright-token",
    jobId: "job-sidebar-1",
    publicApplyToken: "unused",
  };

  await page.route("**/api/v1/auth/me", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: "user-sidebar-1",
        email: "layout@example.com",
        display_name: "Layout Tester",
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
            title: "Chat Sidebar Workspace",
            status: "active",
            created_at: "2026-06-05T08:00:00Z",
            updated_at: "2026-06-05T08:00:00Z",
          },
        ],
        total: 1,
      }),
    });
  });

  await page.route(`**/api/v1/jobs/${setup.jobId}/chat/sessions**`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        items: [
          {
            id: "session-sidebar-1",
            job_id: setup.jobId,
            session_title: "Initial screen",
            created_at: "2026-06-05T08:00:00Z",
            updated_at: "2026-06-05T08:30:00Z",
          },
        ],
        total: 1,
      }),
    });
  });

  await page.route(`**/api/v1/jobs/${setup.jobId}/candidates`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        items: [],
        total: 0,
      }),
    });
  });

  await authenticatePage(page, setup);
  await page.addInitScript(() => {
    if (sessionStorage.getItem("chat-sidebar-test-initialized")) return;
    localStorage.removeItem("recruitai.app-shell-sidebar");
    localStorage.removeItem("recruitai.chat-history-sidebar");
    sessionStorage.setItem("chat-sidebar-test-initialized", "true");
  });
  await page.goto(`${baseURL}/chat`);

  const appSidebar = page.getByTestId("app-sidebar");
  const chatSidebar = page.getByTestId("chat-history-sidebar");

  await expect(appSidebar).toBeVisible();
  await expect(chatSidebar).toBeVisible();
  await expect(page.getByRole("button", { name: "Collapse navigation sidebar" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Collapse conversation history" })).toBeVisible();
  await expect(page.getByTestId("app-sidebar-resize-handle")).toHaveCount(0);
  await expect(page.getByTestId("chat-history-resize-handle")).toBeVisible();

  const appWidthBefore = await appSidebar.evaluate((node) => node.getBoundingClientRect().width);
  const chatWidthBefore = await chatSidebar.evaluate((node) => node.getBoundingClientRect().width);

  await dragHandle(page, "chat-history-resize-handle", 32);

  const appWidthAfterResize = await appSidebar.evaluate((node) => node.getBoundingClientRect().width);
  const chatWidthAfterResize = await chatSidebar.evaluate((node) => node.getBoundingClientRect().width);

  expect(appWidthAfterResize).toBe(appWidthBefore);
  expect(chatWidthAfterResize).toBeGreaterThan(chatWidthBefore + 20);

  const appSidebarSurfaceWidth = await appSidebar
    .locator("aside")
    .evaluate((node) => node.getBoundingClientRect().width);
  expect(Math.abs(appSidebarSurfaceWidth - appWidthAfterResize)).toBeLessThan(1);

  await page.getByRole("button", { name: "Collapse navigation sidebar" }).click();
  await expect(page.getByRole("button", { name: "Expand navigation sidebar" })).toBeVisible();
  await expect(appSidebar).toHaveCSS("width", "0px");

  await page.getByRole("button", { name: "Collapse conversation history" }).click();
  await expect(page.getByRole("button", { name: "Expand conversation history" })).toBeVisible();
  await expect(chatSidebar).toHaveCSS("width", "0px");

  await page.reload();

  await expect(page.getByRole("button", { name: "Expand navigation sidebar" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Expand conversation history" })).toBeVisible();
  await expect(appSidebar).toHaveCSS("width", "0px");
  await expect(chatSidebar).toHaveCSS("width", "0px");
});
