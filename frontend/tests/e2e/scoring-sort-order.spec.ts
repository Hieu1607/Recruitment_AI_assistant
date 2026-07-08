import { expect, test } from "@playwright/test";

import { authenticatePage, seedWorkspace, API_BASE_URL } from "./helpers";

test("scoring dashboard lets recruiters switch between natural and score sorting", async ({ page, request, baseURL }) => {
  const setup = await seedWorkspace(request, "Scoring Sort", [
    {
      fullName: "Alpha Candidate",
      email: "alpha.sort@example.com",
      lines: ["Alpha Candidate", "Python engineer"],
    },
  ]);

  await page.route(`**/api/v1/jobs/${setup.jobId}/evaluations`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        job_id: setup.jobId,
        section_weights: { skills: 25, experience: 25, education: 20, projects: 20, summary: 10 },
        score_threshold: 50,
        scoring_preferences_applied: false,
        total_candidates: 3,
        completed_count: 3,
        pending_count: 0,
        running_count: 0,
        failed_count: 0,
        outdated_count: 0,
        average_score: 60,
        highest_score: 80,
        items: [
          {
            id: "eval-a",
            job_id: setup.jobId,
            job_description_id: setup.jobDescriptionId,
            candidate_profile_id: "candidate-a",
            candidateName: "Alpha Candidate",
            resumeFileName: "alpha.pdf",
            candidateDisplayName: "Alpha Candidate",
            scoring_signature: "sig-a",
            status: "completed",
            totalScore: 20,
            passedThreshold: false,
            rationale: "Alpha rationale",
            error_message: null,
            scored_at: "2026-07-08T10:00:00Z",
            componentScores: [],
          },
          {
            id: "eval-b",
            job_id: setup.jobId,
            job_description_id: setup.jobDescriptionId,
            candidate_profile_id: "candidate-b",
            candidateName: "Beta Candidate",
            resumeFileName: "beta.pdf",
            candidateDisplayName: "Beta Candidate",
            scoring_signature: "sig-b",
            status: "completed",
            totalScore: 80,
            passedThreshold: true,
            rationale: "Beta rationale",
            error_message: null,
            scored_at: "2026-07-08T10:00:00Z",
            componentScores: [],
          },
          {
            id: "eval-c",
            job_id: setup.jobId,
            job_description_id: setup.jobDescriptionId,
            candidate_profile_id: "candidate-c",
            candidateName: "Gamma Candidate",
            resumeFileName: "gamma.pdf",
            candidateDisplayName: "Gamma Candidate",
            scoring_signature: "sig-c",
            status: "completed",
            totalScore: 50,
            passedThreshold: true,
            rationale: "Gamma rationale",
            error_message: null,
            scored_at: "2026-07-08T10:00:00Z",
            componentScores: [],
          },
        ],
      }),
    });
  });

  await authenticatePage(page, setup);
  await page.goto(`${baseURL}/scoring`);

  await expect(page.getByRole("heading", { name: "Scoring dashboard" })).toBeVisible();
  await expect(page.getByRole("combobox", { name: "Sort candidates" })).toHaveValue("natural");

  const candidateNames = page.locator('[data-testid="scoring-candidate-name"]');
  await expect(candidateNames).toHaveText(["Alpha Candidate", "Beta Candidate", "Gamma Candidate"]);

  await page.getByRole("combobox", { name: "Sort candidates" }).selectOption("score_desc");
  await expect(candidateNames).toHaveText(["Beta Candidate", "Gamma Candidate", "Alpha Candidate"]);

  await page.getByRole("combobox", { name: "Sort candidates" }).selectOption("score_asc");
  await expect(candidateNames).toHaveText(["Alpha Candidate", "Gamma Candidate", "Beta Candidate"]);

  await page.getByRole("combobox", { name: "Sort candidates" }).selectOption("natural");
  await expect(candidateNames).toHaveText(["Alpha Candidate", "Beta Candidate", "Gamma Candidate"]);
});
