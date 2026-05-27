import { expect, test } from "@playwright/test";

import { authenticatePage, seedWorkspace } from "./helpers";

test("authenticated workspace browser smoke covers key recruiter flows", async ({ page, request, baseURL }) => {
  const setup = await seedWorkspace(request, "Workspace Smoke", [
    {
      fullName: "Alice Smoke",
      email: "alice.smoke@example.com",
      lines: [
        "Alice Smoke",
        "Senior QA Engineer",
        "Python Playwright Testing",
        "Shortlist and outreach experience",
      ],
    },
    {
      fullName: "Bob Smoke",
      email: "bob.smoke@example.com",
      lines: [
        "Bob Smoke",
        "Recruiting Ops Specialist",
        "Interview scheduling outreach coordination",
        "Candidate pipeline analysis",
      ],
    },
  ]);

  await authenticatePage(page, setup);

  await page.goto(`${baseURL}/candidates`);
  await expect(page.getByRole("heading", { name: "Candidates" })).toBeVisible();
  await expect(page.getByText("Showing 1–2 of 2")).toBeVisible();

  await page.goto(`${baseURL}/shortlists`);
  await page.getByRole("button", { name: "New collection" }).first().click();
  await page.getByRole("textbox", { name: "Collection name…" }).fill("Workspace Smoke Collection");
  await page.getByRole("button", { name: "Create" }).click();
  await expect(page.getByText("Workspace Smoke Collection")).toBeVisible();

  const candidateResponse = await request.get(`http://127.0.0.1:8000/api/v1/jobs/${setup.jobId}/candidates`, {
    headers: { Authorization: `Bearer ${setup.accessToken}` },
  });
  expect(candidateResponse.ok()).toBeTruthy();
  const candidatePayload = await candidateResponse.json();
  const bob = candidatePayload.items.find((item: { full_name: string }) => item.full_name === "Bob Smoke");
  const collectionLink = page.getByRole("link", { name: /Workspace Smoke Collection/ });
  const href = await collectionLink.getAttribute("href");
  const collectionId = href?.split("/").pop();
  expect(collectionId).toBeTruthy();

  const shortlistAdd = await request.post(
    `http://127.0.0.1:8000/api/v1/shortlist/collections/${collectionId}/items`,
    {
      data: { candidate_profile_id: bob.id },
      headers: { Authorization: `Bearer ${setup.accessToken}` },
    },
  );
  expect(shortlistAdd.ok()).toBeTruthy();

  await page.goto(`${baseURL}${href}`);
  await expect(page.getByText("Bob Smoke")).toBeVisible();

  await page.goto(`${baseURL}/outreach`);
  await page.getByRole("button", { name: "+ New message" }).click();
  await page.getByRole("combobox").first().selectOption("Bob Smoke");
  await page.getByRole("textbox", { name: "Subject line…" }).fill("Initial outreach from Playwright");
  await page.getByRole("textbox", { name: "Write your message here…" }).fill(
    "Hi Bob, this draft verifies outreach creation and mark-as-sent from Playwright.",
  );
  await page.getByRole("button", { name: "Save draft" }).click();
  const outreachRow = page
    .getByRole("button", { name: /Bob Smoke not sent Initial outreach from Playwright/ })
    .first();
  await expect(outreachRow).toBeVisible();
  await outreachRow.click();
  await page.getByRole("button", { name: "Mark as sent" }).click();
  await expect(
    page.getByRole("button", { name: /Bob Smoke sent Initial outreach from Playwright/ }).first(),
  ).toBeVisible();

  await page.goto(`${baseURL}/interview-questions`);
  await page.getByRole("button", { name: "Generate new set" }).first().click();
  await page.getByRole("combobox").first().selectOption("Alice Smoke");
  await page.getByRole("combobox").nth(1).selectOption("Workspace Smoke JD");
  const generateResponsePromise = page.waitForResponse(
    (response) =>
      response.url().includes("/api/v1/interview-questions/generate") &&
      response.request().method() === "POST",
    { timeout: 90_000 },
  );
  await page.getByRole("button", { name: "Generate" }).click();
  const generateResponse = await generateResponsePromise;
  expect(generateResponse.ok()).toBeTruthy();
  await expect(page).toHaveURL(/\/interview-questions\/.+/, { timeout: 15_000 });
  await expect(page.getByRole("heading", { name: /Interview for Alice Smoke/ })).toBeVisible();

  await page.goto(`${baseURL}/chat`);
  await page.getByRole("textbox", { name: "Message the recruiter assistant…" }).fill("How many candidates are in this job?");
  await page.getByRole("button", { name: "Send message" }).click();
  await expect(page.getByText("Có 2 ứng viên trong job này.")).toBeVisible();

  await page.goto(`${baseURL}/scoring`);
  await expect(page.getByRole("heading", { name: "Hidden Information" })).toBeVisible();
  await expect(page.getByText("Select the current workspace job description…")).toHaveCount(0);
  await page.getByRole("textbox", { name: "Hidden Information" }).fill("Prefer candidates with recruiter workflow experience.");
  await page.getByRole("button", { name: "Start scoring" }).click();
  const totalCandidatesCard = page.getByText("Total candidates").locator("..");
  await expect(totalCandidatesCard).toBeVisible();
  await expect(totalCandidatesCard.getByText(/^2$/)).toBeVisible();
});
