import { expect, test } from "@playwright/test";

import { authenticatePage } from "./helpers";

test("chat answers render HTML line breaks inside markdown tables", async ({ page, baseURL }) => {
  const setup = {
    accessToken: "playwright-token",
    jobId: "job-chat-render-1",
    publicApplyToken: "unused",
  };
  const sessionId = "session-chat-render-1";

  await page.route("**/api/v1/auth/me", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: "user-chat-render-1",
        email: "render@example.com",
        display_name: "Render Tester",
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
            title: "Chat Render Workspace",
            status: "active",
            created_at: "2026-06-07T08:00:00Z",
            updated_at: "2026-06-07T08:00:00Z",
          },
        ],
        total: 1,
      }),
    });
  });

  await page.route(`**/api/v1/jobs/${setup.jobId}/chat/sessions`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        items: [
          {
            id: sessionId,
            job_id: setup.jobId,
            session_title: "Table rendering",
            created_at: "2026-06-07T08:00:00Z",
            updated_at: "2026-06-07T08:30:00Z",
          },
        ],
        total: 1,
      }),
    });
  });

  await page.route(`**/api/v1/jobs/${setup.jobId}/chat/sessions/${sessionId}/turns**`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([
        {
          id: "turn-chat-render-1",
          user_question: "Summarize candidate skill match",
          answer_text: [
            "| Năng lực | Minh chứng |",
            "| --- | --- |",
            "| Kỹ năng chính | LLM, Retrieval-Augmented Generation (RAG)<br>FastAPI, Docker, vector databases<br>Phát triển chatbot AI |",
            "| Dự án tiêu biểu | Xây dựng chatbot AI có thể triển khai quy mô thực tế<br>Tối ưu hóa pipeline dữ liệu và truy xuất thông tin |",
          ].join("\n"),
          matched_count: 0,
          matched_candidate_ids: [],
        },
      ]),
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
  await page.goto(`${baseURL}/chat/${sessionId}`);

  await expect(page.locator("table")).toBeVisible();
  await expect(page.locator("table td br")).toHaveCount(3);
  await expect(page.getByText(/<br>/)).toHaveCount(0);
});
