import { expect, test } from "@playwright/test";

const APP_URL = "http://localhost:5173";

test("public interview appends speech transcript to existing draft and preserves earlier text after stop", async ({
  page,
}) => {
  const now = new Date("2026-07-07T10:00:00.000Z").toISOString();
  const token = "public-speech-transcript-token";
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
    class MockSpeechRecognition {
      continuous = false;
      interimResults = false;
      lang = "en-US";
      onresult = null;
      onerror = null;
      onend = null;

      constructor() {
        window.__mockRecognition = this;
      }

      start() {}

      stop() {
        this.onend?.();
      }
    }

    window.URL.createObjectURL = () => "blob:mock-audio";
    window.URL.revokeObjectURL = () => {};
    window.HTMLMediaElement.prototype.play = async () => {};
    window.HTMLMediaElement.prototype.pause = () => {};
    window.SpeechRecognition = MockSpeechRecognition;
    window.webkitSpeechRecognition = MockSpeechRecognition;
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

    if (path === `/public/interview/${token}/tts` && method === "POST") {
      await route.fulfill({
        status: 200,
        contentType: "audio/mpeg",
        body: "mock-audio",
      });
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

    await route.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({ detail: `Unhandled mock route: ${method} ${path}` }),
    });
  });

  await page.goto(`${APP_URL}/interviews/${token}`);
  await page.getByRole("button", { name: "Start interview" }).click();

  const transcript = page.getByLabel("Answer transcript");
  await transcript.fill("Existing note.");
  await page.getByRole("button", { name: "Start listening" }).click();

  await page.evaluate(() => {
    window.__mockRecognition.onresult?.({
      resultIndex: 0,
      results: [{ isFinal: true, 0: { transcript: " I led structured hiring." } }],
    });
  });
  await expect(transcript).toHaveValue("Existing note. I led structured hiring.");

  await page.getByRole("button", { name: "Stop listening" }).click();
  await page.evaluate(() => {
    window.__mockRecognition.onresult?.({
      resultIndex: 1,
      results: [
        { isFinal: true, 0: { transcript: " I led structured hiring." } },
        { isFinal: true, 0: { transcript: " I calibrated interviewers." } },
      ],
    });
  });
  await expect(transcript).toHaveValue("Existing note. I led structured hiring. I calibrated interviewers.");
});
