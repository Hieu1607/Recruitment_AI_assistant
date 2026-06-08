import { expect, test } from "@playwright/test";

import { authenticatePage } from "./helpers";

test("chat candidate results can create a shortlist from a modal selector", async ({
  page,
  baseURL,
}) => {
  const setup = {
    accessToken: "playwright-token",
    jobId: "job-chat-shortlist-1",
    publicApplyToken: "unused",
  };
  const sessionId = "session-chat-shortlist-1";
  const collectionId = "collection-chat-shortlist-1";
  const addedCandidates: string[] = [];

  await page.route("**/api/v1/auth/me", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: "user-chat-shortlist-1",
        email: "shortlist@example.com",
        display_name: "Shortlist Tester",
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
            title: "Chat Shortlist Workspace",
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
            session_title: "Shortlist from chat",
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
          id: "turn-chat-shortlist-1",
          query_session_id: sessionId,
          user_question: "Who looks strongest for this role?",
          answer_text: "These candidates are in scope.",
          matched_count: 3,
          matched_candidate_ids: ["candidate-1", "candidate-2", "candidate-3"],
          tool_trace_masked: null,
          created_at: "2026-06-07T08:15:00Z",
        },
      ]),
    });
  });

  await page.route(`**/api/v1/jobs/${setup.jobId}/candidates`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        items: [
          {
            id: "candidate-1",
            resume_document_id: "resume-1",
            extraction_mode: "pymupdf",
            full_name: "Tran Thi Bich Lan",
            submitted_full_name: "Tran Thi Bich Lan",
            phone: null,
            email: "lan@example.com",
            submitted_email: "lan@example.com",
            location_normalized: null,
            contact: null,
            current_job_title: "Marketing Director",
            graduation_status: "unknown",
            ever_studied_abroad: false,
            major: null,
            cpa: null,
            summary_text: null,
            skills_text: "Leadership, Growth",
            experience_text: null,
            experience_years: null,
            education_text: null,
            languages_text: null,
            projects_text: null,
            achievements_text: null,
            publications_text: null,
            certifications_text: null,
            references_text: null,
            other_text: null,
            structured_profile: null,
          },
          {
            id: "candidate-2",
            resume_document_id: "resume-2",
            extraction_mode: "pymupdf",
            full_name: "Phan Thi My Linh",
            submitted_full_name: "Phan Thi My Linh",
            phone: null,
            email: "linh@example.com",
            submitted_email: "linh@example.com",
            location_normalized: null,
            contact: null,
            current_job_title: "Data Scientist",
            graduation_status: "unknown",
            ever_studied_abroad: false,
            major: null,
            cpa: null,
            summary_text: null,
            skills_text: "Python, ML",
            experience_text: null,
            experience_years: null,
            education_text: null,
            languages_text: null,
            projects_text: null,
            achievements_text: null,
            publications_text: null,
            certifications_text: null,
            references_text: null,
            other_text: null,
            structured_profile: null,
          },
          {
            id: "candidate-3",
            resume_document_id: "resume-3",
            extraction_mode: "pymupdf",
            full_name: "Le Thi Hoa",
            submitted_full_name: "Le Thi Hoa",
            phone: null,
            email: "hoa@example.com",
            submitted_email: "hoa@example.com",
            location_normalized: null,
            contact: null,
            current_job_title: "Senior Math Teacher",
            graduation_status: "unknown",
            ever_studied_abroad: false,
            major: null,
            cpa: null,
            summary_text: null,
            skills_text: "Teaching, Curriculum",
            experience_text: null,
            experience_years: null,
            education_text: null,
            languages_text: null,
            projects_text: null,
            achievements_text: null,
            publications_text: null,
            certifications_text: null,
            references_text: null,
            other_text: null,
            structured_profile: null,
          },
        ],
        total: 3,
      }),
    });
  });

  await page.route("**/api/v1/shortlist/collections/", async (route) => {
    if (route.request().method() !== "POST") {
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ items: [], total: 0 }) });
      return;
    }

    const body = route.request().postDataJSON() as Record<string, string>;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: collectionId,
        name: body.name,
        created_by_user_id: body.created_by_user_id,
        source_query_turn_id: body.source_query_turn_id,
        item_count: 0,
        created_at: "2026-06-07T08:35:00Z",
      }),
    });
  });

  await page.route(`**/api/v1/shortlist/collections/${collectionId}/items`, async (route) => {
    const body = route.request().postDataJSON() as { candidate_profile_id: string };
    addedCandidates.push(body.candidate_profile_id);
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: `item-${body.candidate_profile_id}`,
        shortlist_collection_id: collectionId,
        candidate_profile_id: body.candidate_profile_id,
        added_at: "2026-06-07T08:36:00Z",
      }),
    });
  });

  await authenticatePage(page, setup);
  await page.goto(`${baseURL}/chat/${sessionId}`);

  await page.getByRole("button", { name: "Create shortlist" }).click();

  await expect(page.getByRole("dialog", { name: "Create shortlist" })).toBeVisible();
  await page.getByRole("button", { name: "Select all" }).click();
  await expect(page.getByText("3 selected")).toBeVisible();
  await page.getByLabel("Shortlist name").fill("Top AI candidates");
  await page.getByRole("button", { name: "Create shortlist" }).last().click();

  await expect(page.getByRole("dialog", { name: "Create shortlist" })).toHaveCount(0);
  await expect.poll(() => [...addedCandidates].sort()).toEqual(["candidate-1", "candidate-2", "candidate-3"]);
});
