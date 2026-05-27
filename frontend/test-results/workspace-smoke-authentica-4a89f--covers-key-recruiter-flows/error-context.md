# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: workspace-smoke.spec.ts >> authenticated workspace browser smoke covers key recruiter flows
- Location: tests\e2e\workspace-smoke.spec.ts:5:1

# Error details

```
Test timeout of 120000ms exceeded.
```

```
Error: locator.click: Test timeout of 120000ms exceeded.
Call log:
  - waiting for getByRole('button', { name: 'Generate new set' }).first()

```

# Page snapshot

```yaml
- generic [ref=e2]:
  - generic [ref=e3]:
    - complementary [ref=e4]:
      - generic [ref=e5]:
        - paragraph [ref=e6]: RecruitAI
        - paragraph [ref=e7]: Editorial Intelligence
        - paragraph [ref=e8]: "Job: Workspace Smoke"
      - navigation [ref=e9]:
        - list [ref=e10]:
          - listitem [ref=e11]:
            - link "Dashboard" [ref=e12] [cursor=pointer]:
              - /url: /dashboard
              - img [ref=e13]
              - generic [ref=e18]: Dashboard
          - listitem [ref=e19]:
            - link "Jobs" [ref=e20] [cursor=pointer]:
              - /url: /jobs
              - img [ref=e21]
              - generic [ref=e25]: Jobs
          - listitem [ref=e26]:
            - link "Candidates" [ref=e27] [cursor=pointer]:
              - /url: /candidates
              - img [ref=e28]
              - generic [ref=e33]: Candidates
          - listitem [ref=e34]:
            - link "Scoring" [ref=e35] [cursor=pointer]:
              - /url: /scoring
              - img [ref=e36]
              - generic [ref=e38]: Scoring
          - listitem [ref=e39]:
            - link "AI Chat" [ref=e40] [cursor=pointer]:
              - /url: /chat
              - img [ref=e41]
              - generic [ref=e43]: AI Chat
          - listitem [ref=e44]:
            - link "Shortlists" [ref=e45] [cursor=pointer]:
              - /url: /shortlists
              - img [ref=e46]
              - generic [ref=e49]: Shortlists
          - listitem [ref=e50]:
            - link "Outreach" [ref=e51] [cursor=pointer]:
              - /url: /outreach
              - img [ref=e52]
              - generic [ref=e55]: Outreach
          - listitem [ref=e56]:
            - link "Interview Prep" [ref=e57] [cursor=pointer]:
              - /url: /interviews/templates
              - img [ref=e59]
              - generic [ref=e63]: Interview Prep
      - button "Upload resume" [ref=e65]:
        - img [ref=e66]
        - text: Upload resume
      - list [ref=e71]:
        - listitem [ref=e72]:
          - link "Settings" [ref=e73] [cursor=pointer]:
            - /url: /settings
            - img [ref=e74]
            - generic [ref=e77]: Settings
        - listitem [ref=e78]:
          - link "Support" [ref=e79] [cursor=pointer]:
            - /url: /interviews/templates#support
            - img [ref=e80]
            - generic [ref=e83]: Support
    - generic [ref=e84]:
      - banner [ref=e85]:
        - link "RecruitAI — go to dashboard" [ref=e86] [cursor=pointer]:
          - /url: /dashboard
          - generic [ref=e87]: RecruitAI
        - navigation "Breadcrumb" [ref=e88]:
          - img [ref=e89]
          - generic [ref=e91]: Interview Templates
        - generic [ref=e92]:
          - img
          - combobox "Selected job" [ref=e93]:
            - option "Select job" [disabled]
            - option "Workspace Smoke" [selected]
          - img
        - button "Search (Cmd K)" [ref=e95]:
          - img [ref=e96]
          - generic [ref=e99]: Search candidates, JDs…
          - generic [ref=e100]:
            - img [ref=e101]
            - text: K
        - generic [ref=e103]:
          - button "Notifications" [ref=e104]:
            - img [ref=e105]
          - button "Command palette" [ref=e108]:
            - img [ref=e109]
          - group [ref=e111]:
            - generic "User menu" [ref=e112] [cursor=pointer]:
              - generic "User menu" [ref=e113]:
                - generic [ref=e114]: P
      - generic [ref=e118]:
        - generic [ref=e119]:
          - generic [ref=e120]:
            - heading "Interview Templates" [level=1] [ref=e121]
            - paragraph [ref=e122]: Create and maintain job-scoped interview scripts, questions, and report rubrics.
          - button "New template" [ref=e123]:
            - img [ref=e125]
            - generic [ref=e126]: New template
        - table [ref=e129]:
          - rowgroup [ref=e130]:
            - row "Template Status Language Updated" [ref=e131]:
              - columnheader "Template" [ref=e132]:
                - generic [ref=e133]: Template
              - columnheader "Status" [ref=e134]:
                - generic [ref=e135]: Status
              - columnheader "Language" [ref=e136]:
                - generic [ref=e137]: Language
              - columnheader "Updated" [ref=e138]:
                - generic [ref=e139]: Updated
          - rowgroup [ref=e140]:
            - row "No interview templates Create the first template for this job to start sending candidate interview invitations." [ref=e141]:
              - cell "No interview templates Create the first template for this job to start sending candidate interview invitations." [ref=e142]:
                - generic [ref=e143]:
                  - img [ref=e145]
                  - generic [ref=e149]:
                    - paragraph [ref=e150]: No interview templates
                    - paragraph [ref=e151]: Create the first template for this job to start sending candidate interview invitations.
  - region "Notifications alt+T"
```

# Test source

```ts
  1   | import { expect, test } from "@playwright/test";
  2   | 
  3   | import { authenticatePage, seedWorkspace } from "./helpers";
  4   | 
  5   | test("authenticated workspace browser smoke covers key recruiter flows", async ({ page, request, baseURL }) => {
  6   |   const setup = await seedWorkspace(request, "Workspace Smoke", [
  7   |     {
  8   |       fullName: "Alice Smoke",
  9   |       email: "alice.smoke@example.com",
  10  |       lines: [
  11  |         "Alice Smoke",
  12  |         "Senior QA Engineer",
  13  |         "Python Playwright Testing",
  14  |         "Shortlist and outreach experience",
  15  |       ],
  16  |     },
  17  |     {
  18  |       fullName: "Bob Smoke",
  19  |       email: "bob.smoke@example.com",
  20  |       lines: [
  21  |         "Bob Smoke",
  22  |         "Recruiting Ops Specialist",
  23  |         "Interview scheduling outreach coordination",
  24  |         "Candidate pipeline analysis",
  25  |       ],
  26  |     },
  27  |   ]);
  28  | 
  29  |   await authenticatePage(page, setup);
  30  | 
  31  |   await page.goto(`${baseURL}/candidates`);
  32  |   await expect(page.getByRole("heading", { name: "Candidates" })).toBeVisible();
  33  |   await expect(page.getByText("Showing 1–2 of 2")).toBeVisible();
  34  | 
  35  |   await page.goto(`${baseURL}/shortlists`);
  36  |   await page.getByRole("button", { name: "New collection" }).first().click();
  37  |   await page.getByRole("textbox", { name: "Collection name…" }).fill("Workspace Smoke Collection");
  38  |   await page.getByRole("button", { name: "Create" }).click();
  39  |   await expect(page.getByText("Workspace Smoke Collection")).toBeVisible();
  40  | 
  41  |   const candidateResponse = await request.get(`http://127.0.0.1:8000/api/v1/jobs/${setup.jobId}/candidates`, {
  42  |     headers: { Authorization: `Bearer ${setup.accessToken}` },
  43  |   });
  44  |   expect(candidateResponse.ok()).toBeTruthy();
  45  |   const candidatePayload = await candidateResponse.json();
  46  |   const bob = candidatePayload.items.find((item: { full_name: string }) => item.full_name === "Bob Smoke");
  47  |   const collectionLink = page.getByRole("link", { name: /Workspace Smoke Collection/ });
  48  |   const href = await collectionLink.getAttribute("href");
  49  |   const collectionId = href?.split("/").pop();
  50  |   expect(collectionId).toBeTruthy();
  51  | 
  52  |   const shortlistAdd = await request.post(
  53  |     `http://127.0.0.1:8000/api/v1/shortlist/collections/${collectionId}/items`,
  54  |     {
  55  |       data: { candidate_profile_id: bob.id },
  56  |       headers: { Authorization: `Bearer ${setup.accessToken}` },
  57  |     },
  58  |   );
  59  |   expect(shortlistAdd.ok()).toBeTruthy();
  60  | 
  61  |   await page.goto(`${baseURL}${href}`);
  62  |   await expect(page.getByText("Bob Smoke")).toBeVisible();
  63  | 
  64  |   await page.goto(`${baseURL}/outreach`);
  65  |   await page.getByRole("button", { name: "+ New message" }).click();
  66  |   await page.getByRole("combobox").first().selectOption("Bob Smoke");
  67  |   await page.getByRole("textbox", { name: "Subject line…" }).fill("Initial outreach from Playwright");
  68  |   await page.getByRole("textbox", { name: "Write your message here…" }).fill(
  69  |     "Hi Bob, this draft verifies outreach creation and mark-as-sent from Playwright.",
  70  |   );
  71  |   await page.getByRole("button", { name: "Save draft" }).click();
  72  |   const outreachRow = page
  73  |     .getByRole("button", { name: /Bob Smoke not sent Initial outreach from Playwright/ })
  74  |     .first();
  75  |   await expect(outreachRow).toBeVisible();
  76  |   await outreachRow.click();
  77  |   await page.getByRole("button", { name: "Mark as sent" }).click();
  78  |   await expect(
  79  |     page.getByRole("button", { name: /Bob Smoke sent Initial outreach from Playwright/ }).first(),
  80  |   ).toBeVisible();
  81  | 
  82  |   await page.goto(`${baseURL}/interview-questions`);
> 83  |   await page.getByRole("button", { name: "Generate new set" }).first().click();
      |                                                                        ^ Error: locator.click: Test timeout of 120000ms exceeded.
  84  |   await page.getByRole("combobox").first().selectOption("Alice Smoke");
  85  |   await page.getByRole("combobox").nth(1).selectOption("Workspace Smoke JD");
  86  |   const generateResponsePromise = page.waitForResponse(
  87  |     (response) =>
  88  |       response.url().includes("/api/v1/interview-questions/generate") &&
  89  |       response.request().method() === "POST",
  90  |     { timeout: 90_000 },
  91  |   );
  92  |   await page.getByRole("button", { name: "Generate" }).click();
  93  |   const generateResponse = await generateResponsePromise;
  94  |   expect(generateResponse.ok()).toBeTruthy();
  95  |   await expect(page).toHaveURL(/\/interview-questions\/.+/, { timeout: 15_000 });
  96  |   await expect(page.getByRole("heading", { name: /Interview for Alice Smoke/ })).toBeVisible();
  97  | 
  98  |   await page.goto(`${baseURL}/chat`);
  99  |   await page.getByRole("textbox", { name: "Message the recruiter assistant…" }).fill("How many candidates are in this job?");
  100 |   await page.getByRole("button", { name: "Send message" }).click();
  101 |   await expect(page.getByText("Có 2 ứng viên trong job này.")).toBeVisible();
  102 | 
  103 |   await page.goto(`${baseURL}/scoring`);
  104 |   await expect(page.getByRole("heading", { name: "Hidden Information" })).toBeVisible();
  105 |   await expect(page.getByText("Select the current workspace job description…")).toHaveCount(0);
  106 |   await page.getByRole("textbox", { name: "Hidden Information" }).fill("Prefer candidates with recruiter workflow experience.");
  107 |   await page.getByRole("button", { name: "Start scoring" }).click();
  108 |   const totalCandidatesCard = page.getByText("Total candidates").locator("..");
  109 |   await expect(totalCandidatesCard).toBeVisible();
  110 |   await expect(totalCandidatesCard.getByText(/^2$/)).toBeVisible();
  111 | });
  112 | 
```