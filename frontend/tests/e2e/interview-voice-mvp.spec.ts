import { expect, test } from "@playwright/test";

const APP_URL = "http://localhost:5173";

test("recruiter manages interview templates, sends an invitation, and completes public interview flow", async ({
  page,
}) => {
  const now = new Date("2026-05-23T10:00:00.000Z").toISOString();
  const recruiter = {
    id: "user-1",
    email: "recruiter@example.com",
    display_name: "Recruiter Playwright",
  };
  const job = {
    id: "job-1",
    owner_user_id: recruiter.id,
    title: "Interview Voice MVP",
    status: "active",
    candidate_message: null,
    public_apply_enabled: true,
    public_apply_url: `${APP_URL}/apply/job-public-token`,
    created_at: now,
    updated_at: now,
    archived_at: null,
  };
  const resume = {
    id: "resume-1",
    job_id: job.id,
    original_file_name: "alice_interview.pdf",
    candidate_profile_id: "candidate-1",
    candidate_display_name: "Alice Interview",
    storage_uri: "/mock/alice_interview.pdf",
    upload_status: "processed",
    duplicate_group_key: null,
    uploaded_by_user_id: recruiter.id,
    uploaded_at: now,
    processed_at: now,
    retention_expires_at: null,
  };
  const profile = {
    id: "candidate-1",
    resume_document_id: resume.id,
    full_name: "Alice Interview",
    phone: null,
    email: "alice.interview@example.com",
    location_normalized: "Ho Chi Minh City",
    current_job_title: "Recruiting Operations Manager",
    summary_text: "Structured recruiting operator with strong hiring workflow design experience.",
    skills_text: "Structured interviewing, calibration, recruiter ops",
    experience_text: "Led candidate coordination and hiring process design.",
    experience_years: 5,
    education_text: null,
    languages_text: null,
    projects_text: null,
    achievements_text: null,
    certifications_text: null,
  };

  let template: null | {
    id: string;
    job_id: string;
    name: string;
    language_code: string;
    status: string;
    intro_script: string;
    closing_script: string;
    question_payload: Record<string, unknown>;
    report_rubric: Record<string, unknown>;
    version: number;
    created_at: string;
    updated_at: string;
  } = {
    id: "template-1",
    job_id: job.id,
    name: "Structured recruiter screen",
    language_code: "en-US",
    status: "active",
    intro_script: "",
    closing_script: "",
    question_payload: {
      questions: [{ key: "intro", prompt: "Tell me about your hiring approach." }],
    },
    report_rubric: { score_bands: ["strong", "mixed", "weak"] },
    version: 1,
    created_at: now,
    updated_at: now,
  };

  let invitation = null as null | {
    id: string;
    job_id: string;
    candidate_profile_id: string;
    candidate_full_name: string | null;
    interview_template_id: string;
    interview_template_name: string | null;
    public_token: string;
    public_url: string;
    status: string;
    expires_at: string | null;
    max_attempts: number;
    attempt_count: number;
    latest_interview_session_id: string | null;
    sent_by_user_id: string | null;
    sent_at: string | null;
    opened_at: string | null;
    completed_at: string | null;
    cancelled_at: string | null;
    created_at: string;
    updated_at: string;
  };
  let session = {
    id: "session-1",
    provider: "fake",
    provider_session_id: "browser-session-1",
    status: "created",
    started_at: null as string | null,
    completed_at: null as string | null,
  };
  let report = {
    id: "report-1",
    interview_session_id: session.id,
    interview_template_id: template?.id ?? null,
    summary_text: "Candidate demonstrated structured recruiting workflows and clear communication.",
    report_payload: {
      status: "completed",
      recommendation: "advance",
      competencies: [{ name: "Structured hiring", score: "strong" }],
    },
    created_at: now,
    updated_at: now,
  };

  const transcriptEvents: Array<{ speaker: string; text: string; question_key?: string | null }> = [];

  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const method = route.request().method();
    const body = route.request().postDataJSON?.() as Record<string, unknown> | undefined;
    const path = url.pathname.replace("/api/v1", "");

    const json = async (payload: unknown, status = 200) => {
      await route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify(payload),
      });
    };

    if (path === "/auth/me" && method === "GET") {
      await json(recruiter);
      return;
    }

    if (path === "/jobs/" && method === "GET") {
      await json({ items: [job], total: 1 });
      return;
    }

    if (path === `/jobs/${job.id}/interview-templates` && method === "GET") {
      await json({ items: template ? [template] : [], total: template ? 1 : 0 });
      return;
    }

    if (path === `/jobs/${job.id}/interview-templates` && method === "POST") {
      template = {
        id: "template-1",
        job_id: job.id,
        name: String(body?.name ?? "Structured recruiter screen"),
        language_code: String(body?.language_code ?? "en-US"),
        status: String(body?.status ?? "draft"),
        intro_script: typeof body?.intro_script === "string" ? body.intro_script : "",
        closing_script: typeof body?.closing_script === "string" ? body.closing_script : "",
        question_payload: (body?.question_payload as Record<string, unknown>) ?? { questions: [] },
        report_rubric: (body?.report_rubric as Record<string, unknown>) ?? {},
        version: 1,
        created_at: now,
        updated_at: now,
      };
      report.interview_template_id = template.id;
      await json(template, 201);
      return;
    }

    if (path === `/interview-templates/${template?.id}` && method === "GET" && template) {
      await json(template);
      return;
    }

    if (path === `/interview-templates/${template?.id}` && method === "PATCH" && template) {
      template = {
        ...template,
        name: typeof body?.name === "string" ? body.name : template.name,
        language_code: typeof body?.language_code === "string" ? body.language_code : template.language_code,
        status: typeof body?.status === "string" ? body.status : template.status,
        intro_script: typeof body?.intro_script === "string" ? body.intro_script : template.intro_script,
        closing_script: typeof body?.closing_script === "string" ? body.closing_script : template.closing_script,
        question_payload: (body?.question_payload as Record<string, unknown>) ?? template.question_payload,
        report_rubric: (body?.report_rubric as Record<string, unknown>) ?? template.report_rubric,
        version: template.version + 1,
        updated_at: new Date("2026-05-23T10:10:00.000Z").toISOString(),
      };
      await json(template);
      return;
    }

    if (path === `/upload/${resume.id}` && method === "GET") {
      await json(resume);
      return;
    }

    if (path === "/upload/" && method === "GET") {
      await json({ items: [resume], total: 1 });
      return;
    }

    if (path === `/upload/${resume.id}/profile` && method === "GET") {
      await json(profile);
      return;
    }

    if (path === "/outreach/" && method === "GET") {
      await json({ items: [], total: 0 });
      return;
    }

    if (path === `/jobs/${job.id}/interview-invitations` && method === "GET") {
      await json({ items: invitation ? [invitation] : [], total: invitation ? 1 : 0 });
      return;
    }

    if (path === "/interview-invitations" && method === "POST" && template) {
      invitation = {
        id: "invitation-1",
        job_id: job.id,
        candidate_profile_id: profile.id,
        candidate_full_name: profile.full_name,
        interview_template_id: template.id,
        interview_template_name: template.name,
        public_token: "public-invite-token",
        public_url: `${APP_URL}/interviews/public-invite-token`,
        status: "pending",
        expires_at: "2026-05-26T10:00:00.000Z",
        max_attempts: 1,
        attempt_count: 0,
        latest_interview_session_id: null,
        sent_by_user_id: recruiter.id,
        sent_at: now,
        opened_at: null,
        completed_at: null,
        cancelled_at: null,
        created_at: now,
        updated_at: now,
      };
      await json(invitation, 201);
      return;
    }

    if (path === `/public/interview/${invitation?.public_token}` && method === "GET" && invitation && template) {
      await json({
        invitation: {
          id: invitation.id,
          public_token: invitation.public_token,
          status: invitation.status,
          expires_at: invitation.expires_at,
          max_attempts: invitation.max_attempts,
          attempt_count: invitation.attempt_count,
          candidate_full_name: invitation.candidate_full_name,
          completed_at: invitation.completed_at,
        },
        template: {
          id: template.id,
          name: template.name,
          language_code: template.language_code,
          intro_script: template.intro_script,
          closing_script: template.closing_script,
          question_payload: template.question_payload,
        },
        availability: {
          can_start: true,
          reason: "ready",
          detail: null,
        },
      });
      return;
    }

    if (path === `/public/interview/${invitation?.public_token}/start` && method === "POST" && invitation && template) {
      invitation = {
        ...invitation,
        status: "in_progress",
        opened_at: now,
        attempt_count: 1,
        latest_interview_session_id: session.id,
        updated_at: now,
      };
      session = {
        ...session,
        provider_session_id: typeof body?.provider_session_id === "string" ? body.provider_session_id : session.provider_session_id,
        status: "in_progress",
        started_at: now,
      };
      await json({
        invitation: {
          id: invitation.id,
          public_token: invitation.public_token,
          status: invitation.status,
          expires_at: invitation.expires_at,
          max_attempts: invitation.max_attempts,
          attempt_count: invitation.attempt_count,
          candidate_full_name: invitation.candidate_full_name,
          completed_at: invitation.completed_at,
        },
        session,
        template: {
          id: template.id,
          name: template.name,
          language_code: template.language_code,
          intro_script: template.intro_script,
          closing_script: template.closing_script,
          question_payload: template.question_payload,
        },
      });
      return;
    }

    if (path === `/public/interview/${invitation?.public_token}/events` && method === "POST") {
      const events = Array.isArray(body?.events) ? body.events : [];
      transcriptEvents.push(
        ...events.map((event) => ({
          speaker: String((event as Record<string, unknown>).speaker),
          text: String((event as Record<string, unknown>).text),
          question_key:
            typeof (event as Record<string, unknown>).question_key === "string"
              ? String((event as Record<string, unknown>).question_key)
              : null,
        })),
      );
      await json({ accepted: true, stored_turns: events.length }, 202);
      return;
    }

    if (path === `/public/interview/${invitation?.public_token}/complete` && method === "POST" && invitation) {
      invitation = {
        ...invitation,
        status: "completed",
        completed_at: now,
        latest_interview_session_id: session.id,
        updated_at: now,
      };
      session = {
        ...session,
        status: "completed",
        completed_at: now,
      };
      report = {
        ...report,
        interview_session_id: session.id,
        updated_at: now,
      };
      await json({
        invitation: {
          id: invitation.id,
          public_token: invitation.public_token,
          status: invitation.status,
          expires_at: invitation.expires_at,
          max_attempts: invitation.max_attempts,
          attempt_count: invitation.attempt_count,
          candidate_full_name: invitation.candidate_full_name,
          completed_at: invitation.completed_at,
        },
        session,
      });
      return;
    }

    if (path === `/interview-reports/${session.id}` && method === "GET") {
      await json(report);
      return;
    }

    await route.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({ detail: `Unhandled mock route: ${method} ${path}` }),
    });
  });

  await page.goto(APP_URL);
  await page.evaluate(
    ([token, jobId]) => {
      localStorage.setItem("recruitai.token", token);
      localStorage.setItem("recruit_ai_selected_job_id", jobId);
    },
    ["playwright-token", job.id],
  );

  await page.goto(`${APP_URL}/interviews/templates`);

  await page.getByRole("button", { name: "New template" }).click();
  await page.getByLabel("Template name").fill("Structured recruiter screen");
  await page.getByLabel("Interview language").selectOption("en-US");
  await page.getByLabel("Status").selectOption("active");
  await page
    .getByLabel("Question list import")
    .fill("1. Tell me about your hiring approach.\n2. How do you keep interviews consistent?");
  await page.getByRole("button", { name: "Import questions" }).click();
  await expect(page.getByLabel("Question 1")).toHaveValue("Tell me about your hiring approach.");
  await expect(page.getByLabel("Question 2")).toHaveValue("How do you keep interviews consistent?");
  await page
    .getByLabel("Report guidance")
    .fill("Focus the summary on structured interviewing, candidate communication, and follow-up risks.");
  await page.getByRole("button", { name: "Create template" }).click();
  expect(template?.report_rubric).toEqual({
    guidance: "Focus the summary on structured interviewing, candidate communication, and follow-up risks.",
  });

  const createdRow = page.getByRole("link", { name: /Structured recruiter screen/i });
  await expect(createdRow).toBeVisible();
  await createdRow.click();
  await expect(page).toHaveURL(/\/interviews\/templates\/.+/);

  await page.getByLabel("Intro script").fill("Welcome to the structured screen.");
  await page.getByRole("button", { name: "Save changes" }).click();
  await expect(page.getByLabel("Intro script")).toHaveValue("Welcome to the structured screen.");
  expect(template?.report_rubric).toEqual({
    guidance: "Focus the summary on structured interviewing, candidate communication, and follow-up risks.",
  });

  await page.goto(`${APP_URL}/interviews`);
  await page.getByRole("button", { name: "New interview link" }).click();
  await page.getByLabel("Candidate").selectOption({ label: "Alice Interview" });
  await page.getByLabel("Interview template").selectOption({ label: "Structured recruiter screen" });
  await page.getByLabel("Expires in hours").fill("72");
  await page.getByRole("button", { name: "Create link" }).click();

  await expect(page.getByRole("heading", { name: "Interviews" })).toBeVisible();
  await expect(page.getByText("Alice Interview")).toBeVisible();
  await expect(page.getByText("Structured recruiter screen")).toBeVisible();
  await expect(page.getByText(/^0\/1$/)).toBeVisible();

  await page.goto(`${APP_URL}/interviews/public-invite-token`);
  await page.getByRole("button", { name: "Start interview" }).click();
  await expect(page.getByText(/Tell me about your hiring approach/i)).toBeVisible();
  await page.getByLabel("Answer transcript").fill("I build structured recruiting workflows.");
  await page.getByRole("button", { name: "Next question" }).click();
  await expect(page.getByText(/How do you keep interviews consistent/i)).toBeVisible();
  await page.getByLabel("Answer transcript").fill("I use a shared scorecard and calibrated rubrics.");
  await page.getByRole("button", { name: "Finish interview" }).click();

  await expect(page.getByRole("heading", { name: "Interview completed" })).toBeVisible();
  expect(transcriptEvents.some((event) => event.speaker === "agent")).toBeTruthy();
  expect(
    transcriptEvents.some(
      (event) => event.speaker === "user" && /structured recruiting workflows/i.test(event.text),
    ),
  ).toBeTruthy();
  expect(
    transcriptEvents.some(
      (event) => event.speaker === "user" && /shared scorecard and calibrated rubrics/i.test(event.text),
    ),
  ).toBeTruthy();

  await page.goto(`${APP_URL}/interviews`);
  await expect(page.getByRole("link", { name: /Report/i })).toBeVisible();

  await page.goto(`${APP_URL}/interviews/reports/${session.id}`);
  await expect(page.getByRole("heading", { name: "Interview Report" })).toBeVisible();
  await expect(page.getByText(/structured recruiting workflows/i)).toBeVisible();
});

test("recruiter can revoke an active interview link from the interviews hub", async ({ page }) => {
  const now = new Date("2026-05-23T10:00:00.000Z").toISOString();
  const recruiter = {
    id: "user-1",
    email: "recruiter@example.com",
    display_name: "Recruiter Playwright",
  };
  const job = {
    id: "job-1",
    owner_user_id: recruiter.id,
    title: "Interview Voice MVP",
    status: "active",
    candidate_message: null,
    public_apply_enabled: true,
    public_apply_url: `${APP_URL}/apply/job-public-token`,
    created_at: now,
    updated_at: now,
    archived_at: null,
  };
  const resume = {
    id: "resume-1",
    job_id: job.id,
    original_file_name: "alice_interview.pdf",
    candidate_profile_id: "candidate-1",
    candidate_display_name: "Alice Interview",
    storage_uri: "/mock/alice_interview.pdf",
    upload_status: "processed",
    duplicate_group_key: null,
    uploaded_by_user_id: recruiter.id,
    uploaded_at: now,
    processed_at: now,
    retention_expires_at: null,
  };
  const template = {
    id: "template-1",
    job_id: job.id,
    name: "Structured recruiter screen",
    language_code: "en-US",
    status: "active",
    intro_script: "",
    closing_script: "",
    question_payload: { questions: [] },
    report_rubric: {},
    version: 1,
    created_at: now,
    updated_at: now,
  };
  let invitation = {
    id: "invitation-1",
    job_id: job.id,
    candidate_profile_id: "candidate-1",
    candidate_full_name: "Alice Interview",
    interview_template_id: template.id,
    interview_template_name: template.name,
    public_token: "public-invite-token",
    public_url: `${APP_URL}/interviews/public-invite-token`,
    status: "pending",
    expires_at: "2026-05-26T10:00:00.000Z",
    max_attempts: 1,
    attempt_count: 0,
    latest_interview_session_id: null,
    sent_by_user_id: recruiter.id,
    sent_at: now,
    opened_at: null,
    completed_at: null,
    cancelled_at: null as string | null,
    created_at: now,
    updated_at: now,
  };

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

    if (path === "/auth/me" && method === "GET") {
      await json(recruiter);
      return;
    }
    if (path === "/jobs/" && method === "GET") {
      await json({ items: [job], total: 1 });
      return;
    }
    if (path === "/upload/" && method === "GET") {
      await json({ items: [resume], total: 1 });
      return;
    }
    if (path === `/jobs/${job.id}/interview-templates` && method === "GET") {
      await json({ items: [template], total: 1 });
      return;
    }
    if (path === `/jobs/${job.id}/interview-invitations` && method === "GET") {
      await json({ items: [invitation], total: 1 });
      return;
    }
    if (path === `/interview-invitations/${invitation.id}/revoke` && method === "POST") {
      invitation = {
        ...invitation,
        status: "cancelled",
        cancelled_at: now,
        updated_at: now,
      };
      await json(invitation);
      return;
    }

    await route.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({ detail: `Unhandled mock route: ${method} ${path}` }),
    });
  });

  await page.goto(APP_URL);
  await page.evaluate(
    ([token, jobId]) => {
      localStorage.setItem("recruitai.token", token);
      localStorage.setItem("recruit_ai_selected_job_id", jobId);
    },
    ["playwright-token", job.id],
  );

  await page.goto(`${APP_URL}/interviews`);
  await expect(page.getByText("Alice Interview")).toBeVisible();
  await page.getByRole("button", { name: "Revoke" }).click();
  await page.getByRole("button", { name: "Confirm revoke" }).click();
  await expect(page.getByText("cancelled")).toBeVisible();
});

test("recruiter can delete an unused interview template from template detail", async ({ page }) => {
  const now = new Date("2026-05-23T10:00:00.000Z").toISOString();
  const recruiter = {
    id: "user-1",
    email: "recruiter@example.com",
    display_name: "Recruiter Playwright",
  };
  const job = {
    id: "job-1",
    owner_user_id: recruiter.id,
    title: "Interview Voice MVP",
    status: "active",
    candidate_message: null,
    public_apply_enabled: true,
    public_apply_url: `${APP_URL}/apply/job-public-token`,
    created_at: now,
    updated_at: now,
    archived_at: null,
  };
  let template = {
    id: "template-1",
    job_id: job.id,
    name: "Unused recruiter screen",
    language_code: "en-US",
    status: "draft",
    intro_script: "",
    closing_script: "",
    question_payload: { questions: [] },
    report_rubric: {},
    version: 1,
    created_at: now,
    updated_at: now,
  };

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

    if (path === "/auth/me" && method === "GET") {
      await json(recruiter);
      return;
    }
    if (path === "/jobs/" && method === "GET") {
      await json({ items: [job], total: 1 });
      return;
    }
    if (path === `/jobs/${job.id}/interview-templates` && method === "GET") {
      await json({ items: template ? [template] : [], total: template ? 1 : 0 });
      return;
    }
    if (path === `/interview-templates/${template?.id}` && method === "GET" && template) {
      await json(template);
      return;
    }
    if (path === `/interview-templates/${template?.id}` && method === "DELETE" && template) {
      const deletedId = template.id;
      template = null as never;
      await json({ deleted: true, template_id: deletedId });
      return;
    }

    await route.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({ detail: `Unhandled mock route: ${method} ${path}` }),
    });
  });

  await page.goto(APP_URL);
  await page.evaluate(
    ([token, jobId]) => {
      localStorage.setItem("recruitai.token", token);
      localStorage.setItem("recruit_ai_selected_job_id", jobId);
    },
    ["playwright-token", job.id],
  );

  await page.goto(`${APP_URL}/interviews/templates`);
  await page.getByRole("link", { name: "Unused recruiter screen" }).click();
  await page.getByRole("button", { name: "Delete template" }).click();
  await page.getByRole("button", { name: "Confirm delete" }).click();
  await expect(page).toHaveURL(`${APP_URL}/interviews/templates`);
  await expect(page.getByText("Unused recruiter screen")).toHaveCount(0);
});

test("public interview link shows expired state before start is clicked", async ({ page }) => {
  const expiredAt = "2026-05-22T10:00:00.000Z";

  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const method = route.request().method();
    const path = url.pathname.replace("/api/v1", "");

    if (path === "/public/interview/expired-public-token" && method === "GET") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          invitation: {
            id: "invitation-expired",
            public_token: "expired-public-token",
            status: "pending",
            expires_at: expiredAt,
            max_attempts: 1,
            attempt_count: 0,
            candidate_full_name: "Expired Candidate",
            completed_at: null,
          },
          template: {
            id: "template-expired",
            name: "Structured recruiter screen",
            language_code: "en-US",
            intro_script: "",
            closing_script: "",
            question_payload: {
              questions: [{ key: "intro", prompt: "Tell me about your hiring approach." }],
            },
          },
          availability: {
            can_start: false,
            reason: "expired",
            detail: "Interview invitation has expired",
          },
        }),
      });
      return;
    }

    await route.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({ detail: `Unhandled mock route: ${method} ${path}` }),
    });
  });

  await page.goto(`${APP_URL}/interviews/expired-public-token`);

  await expect(page.getByText("Interview invitation has expired")).toBeVisible();
  await expect(page.getByRole("button", { name: "Start interview" })).toHaveCount(0);
});
