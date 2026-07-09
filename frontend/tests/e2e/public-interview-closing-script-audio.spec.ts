import { expect, test } from "@playwright/test";

const APP_URL = "http://localhost:5173";

test("public interview waits for closing script audio to finish before completing", async ({ page }) => {
  const now = new Date("2026-07-09T10:00:00.000Z").toISOString();
  const token = "public-closing-script-token";
  const invitation = {
    id: "invitation-1",
    public_token: token,
    status: "pending",
    expires_at: "2026-07-12T10:00:00.000Z",
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
    closing_script: "Thanks for your time.",
    question_payload: {
      questions: [{ key: "q1", prompt: "Tell me about your experience with hiring workflows." }],
    },
  };
  let completeRequestCount = 0;

  await page.addInitScript(() => {
    class MockAudio {
      static latestInstance: MockAudio | null = null;

      onended: (() => void) | null = null;
      private listeners = new Map<string, Set<() => void>>();

      constructor() {
        MockAudio.latestInstance = this;
        window.__mockAudioState.instances += 1;
      }

      addEventListener(type: string, listener: () => void) {
        const listeners = this.listeners.get(type) ?? new Set<() => void>();
        listeners.add(listener);
        this.listeners.set(type, listeners);
      }

      removeEventListener(type: string, listener: () => void) {
        this.listeners.get(type)?.delete(listener);
      }

      pause() {
        window.__mockAudioState.pauseCalls += 1;
      }

      async play() {
        window.__mockAudioState.playCalls += 1;
      }

      emit(type: string) {
        if (type === "ended") {
          this.onended?.();
        }
        for (const listener of this.listeners.get(type) ?? []) {
          listener();
        }
      }
    }

    window.__mockAudioState = {
      instances: 0,
      playCalls: 0,
      pauseCalls: 0,
      finishCurrent() {
        MockAudio.latestInstance?.emit("ended");
      },
    };
    window.URL.createObjectURL = () => "blob:mock-audio";
    window.URL.revokeObjectURL = () => {};
    window.Audio = MockAudio as typeof window.Audio;
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
      completeRequestCount += 1;
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
  await page.getByLabel("Answer transcript").fill("I build repeatable interview processes.");

  await page.getByRole("button", { name: "Finish interview" }).click();
  await page.waitForTimeout(100);

  expect(completeRequestCount).toBe(0);
  await expect(page.getByRole("heading", { name: "Interview completed" })).toHaveCount(0);

  await page.evaluate(() => {
    window.__mockAudioState.finishCurrent();
  });

  await expect.poll(() => completeRequestCount).toBe(1);
  await expect(page.getByRole("heading", { name: "Interview completed" })).toBeVisible();
});
