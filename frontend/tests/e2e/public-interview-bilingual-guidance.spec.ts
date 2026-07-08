import { expect, test } from "@playwright/test";

const APP_URL = "http://localhost:5173";

test("public interview shows bilingual guidance on start, question, and completion screens", async ({ page }) => {
  const now = new Date("2026-07-07T10:00:00.000Z").toISOString();
  const token = "public-bilingual-token";
  const invitation = {
    id: "invitation-1",
    public_token: token,
    status: "pending",
    expires_at: "2026-07-10T10:00:00.000Z",
    max_attempts: 1,
    attempt_count: 0,
    candidate_full_name: "Nguyen Van A",
    completed_at: null as string | null,
  };
  const template = {
    id: "template-1",
    name: "Basic",
    language_code: "en-US",
    intro_script: "",
    closing_script: "",
    question_payload: {
      questions: [{ key: "q1", prompt: "Tell me about your experience with hiring workflows." }],
    },
  };

  await page.addInitScript(() => {
    window.URL.createObjectURL = () => "blob:mock-audio";
    window.URL.revokeObjectURL = () => {};
    window.HTMLMediaElement.prototype.play = async () => {};
    window.HTMLMediaElement.prototype.pause = () => {};
  });

  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const method = route.request().method();
    const path = url.pathname.replace("/api/v1", "");

    const json = async (payload: unknown, status = 200) => {
      await route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify(payload),
      });
    };

    if (path === `/public/interview/${token}` && method === "GET") {
      await json({
        invitation,
        template,
        availability: {
          can_start: true,
          reason: "ready",
          detail: null,
        },
      });
      return;
    }

    if (path === `/public/interview/${token}/start` && method === "POST") {
      invitation.status = "in_progress";
      invitation.attempt_count = 1;
      await json({
        invitation,
        session: {
          id: "session-1",
          provider: "fake",
          provider_session_id: "browser-session-1",
          status: "in_progress",
          started_at: now,
          completed_at: null,
        },
        template,
      });
      return;
    }

    if (path === `/public/interview/${token}/events` && method === "POST") {
      await json({ accepted: true, stored_turns: 1 }, 202);
      return;
    }

    if (path === `/public/interview/${token}/complete` && method === "POST") {
      invitation.status = "completed";
      invitation.completed_at = now;
      await json({
        invitation,
        session: {
          id: "session-1",
          provider: "fake",
          provider_session_id: "browser-session-1",
          status: "completed",
          started_at: now,
          completed_at: now,
        },
      });
      return;
    }

    if (path === `/public/interview/${token}/tts` && method === "POST") {
      await route.fulfill({
        status: 200,
        contentType: "audio/mpeg",
        body: "mock-audio",
      });
      return;
    }

    await route.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({ detail: `Unhandled mock route: ${method} ${path}` }),
    });
  });

  await page.goto(`${APP_URL}/interviews/${token}`);

  const startGuidanceItem = page.getByRole("listitem").filter({
    hasText: "Bạn sẽ được hỏi đúng theo bộ câu hỏi đã được cấu hình cho vị trí này.",
  });
  await expect(startGuidanceItem).toBeVisible();
  await expect(
    startGuidanceItem.getByText("The interviewer will ask only the questions configured for this role."),
  ).toBeVisible();

  await page.getByRole("button", { name: "Start interview" }).click();
  await expect(page.getByText("Tell me about your experience with hiring workflows.")).toBeVisible();

  const questionGuidance = page.getByRole("listitem").filter({
    hasText: "Bạn có thể trả lời bằng giọng nói hoặc nhập trực tiếp vào ô bên dưới.",
  });
  await expect(questionGuidance).toBeVisible();
  await expect(questionGuidance.getByText("You can answer by voice or type directly into the box below.")).toBeVisible();

  await page.getByLabel("Answer transcript").fill("I build repeatable interview processes.");
  await page.getByRole("button", { name: "Finish interview" }).click();

  const completionCard = page.getByRole("listitem").filter({
    hasText: "Câu trả lời của bạn đã được gửi đến nhà tuyển dụng để xem xét.",
  });
  await expect(completionCard).toBeVisible();
  await expect(
    completionCard.getByText(
      "Your responses have been submitted. The recruiter can now review the transcript and structured summary.",
    ),
  ).toBeVisible();
});
