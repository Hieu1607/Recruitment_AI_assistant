# Outreach Workspaces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split Outreach into dedicated `Messages` and `Templates` workspaces, add one-shot AI template drafting, and keep message creation template-first without AI.

**Architecture:** Add a backend draft-generation endpoint plus a small enum extension for message provenance, then split the frontend route into two focused workspace screens that reuse the existing rich editor and outreach APIs. Keep message sending and Gmail onboarding in the messages workspace, while moving template creation and AI generation into a separate templates workspace.

**Tech Stack:** FastAPI, SQLAlchemy, Pydantic, React, React Router, TanStack Query, Vitest/Playwright, pytest

---

### Task 1: Add backend tests for content source and AI draft generation

**Files:**
- Modify: `backend/tests/test_outreach_endpoints.py`
- Test: `backend/tests/test_outreach_endpoints.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_create_outreach_message_blank_sets_manual_content_source(client, auth_headers, seeded_candidate_profile, seeded_user):
    response = client.post(
        "/api/v1/outreach/",
        headers=auth_headers,
        json={
            "candidate_profile_id": str(seeded_candidate_profile.id),
            "created_by_user_id": str(seeded_user.id),
            "content_source": "manual",
            "subject": "Checking in",
            "body_text": "Hello there",
            "body_html": "<p>Hello there</p>",
            "template_id": None,
        },
    )
    assert response.status_code == 201
    assert response.json()["content_source"] == "manual"


def test_generate_outreach_template_draft_returns_subject_and_body(client, auth_headers, seeded_job, monkeypatch):
    monkeypatch.setattr(
        "src.api.v1.endpoints.outreach._generate_outreach_template_draft",
        lambda **kwargs: {
            "subject": "Opportunity at {{company_name}}",
            "body_text": "Hi {{candidate_name}}",
            "body_html": "<p>Hi {{candidate_name}}</p>",
            "variables_used": ["candidate_name", "company_name"],
        },
    )
    response = client.post(
        "/api/v1/outreach/templates/generate-draft",
        headers=auth_headers,
        json={
            "job_id": str(seeded_job.id),
            "brief": "Write a short recruiter intro email",
            "variables_allowed": ["candidate_name", "company_name", "job_title"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["subject"] == "Opportunity at {{company_name}}"
    assert payload["variables_used"] == ["candidate_name", "company_name"]
```

- [ ] **Step 2: Run the targeted tests and verify they fail**

Run: `pytest backend/tests/test_outreach_endpoints.py -k "manual_content_source or generate_outreach_template_draft" -v`

Expected: FAIL because `manual` is not an accepted enum value and `/api/v1/outreach/templates/generate-draft` does not exist.

- [ ] **Step 3: Add validation edge-case tests**

```python
def test_generate_outreach_template_draft_rejects_blank_brief(client, auth_headers, seeded_job):
    response = client.post(
        "/api/v1/outreach/templates/generate-draft",
        headers=auth_headers,
        json={
            "job_id": str(seeded_job.id),
            "brief": "   ",
            "variables_allowed": ["candidate_name"],
        },
    )
    assert response.status_code == 422


def test_generate_outreach_template_draft_returns_404_for_missing_job(client, auth_headers):
    response = client.post(
        "/api/v1/outreach/templates/generate-draft",
        headers=auth_headers,
        json={
            "job_id": "00000000-0000-0000-0000-000000000001",
            "brief": "Need an email",
            "variables_allowed": ["candidate_name"],
        },
    )
    assert response.status_code == 404
```

- [ ] **Step 4: Re-run the targeted tests and verify they still fail for the expected missing implementation**

Run: `pytest backend/tests/test_outreach_endpoints.py -k "generate_outreach_template_draft or manual_content_source" -v`

Expected: FAIL with route/enum-related failures, not syntax errors.

### Task 2: Implement backend content-source support and draft generation

**Files:**
- Modify: `backend/src/models/enums.py`
- Modify: `backend/src/api/v1/endpoints/outreach.py`
- Modify: `backend/src/prompts/build_prompts.py`
- Modify: `frontend/src/api/types.ts`
- Test: `backend/tests/test_outreach_endpoints.py`

- [ ] **Step 1: Add the `manual` enum value**

```python
class ContentSource(str, Enum):
    AI_DRAFT = "ai_draft"
    TEMPLATE = "template"
    MANUAL = "manual"
```

- [ ] **Step 2: Add prompt builder support for outreach template drafts**

```python
def build_outreach_template_draft_prompt(
    self,
    *,
    brief: str,
    job_title: str | None,
    company_name: str | None,
    variables_allowed: list[str],
) -> str:
    variable_lines = "\n".join(f"- {{{{{name}}}}}" for name in variables_allowed)
    return f"""
Generate a recruiter outreach email template as JSON.

Return valid JSON with keys:
- subject
- body_text
- body_html
- variables_used

Rules:
- Use only these variables when needed:
{variable_lines}
- Keep the tone professional and concise.
- Do not mention variables outside the allowed list.
- body_html should be simple semantic HTML.

Context:
- Job title: {job_title or "Unknown"}
- Company name: {company_name or "Unknown"}
- Recruiter brief: {brief.strip()}
""".strip()
```

- [ ] **Step 3: Add a small backend helper that calls the LLM and parses the JSON**

```python
def _generate_outreach_template_draft(*, brief: str, job, variables_allowed: list[str]) -> dict:
    from src.prompts.build_prompts import build_prompts
    from src.services.llm_service import LLMProvider, LLMProviderError

    prompt = build_prompts.build_outreach_template_draft_prompt(
        brief=brief,
        job_title=getattr(job, "title", None),
        company_name=None,
        variables_allowed=variables_allowed,
    )
    try:
        response = LLMProvider().generate(prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.rsplit("```", 1)[0].strip()
        payload = json.loads(text)
    except (LLMProviderError, json.JSONDecodeError, Exception) as exc:
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {exc}")
    return payload
```

- [ ] **Step 4: Add the new generate endpoint and request/response schemas**

```python
class OutreachTemplateGenerateRequest(BaseModel):
    job_id: uuid.UUID
    brief: str = Field(..., min_length=1)
    variables_allowed: list[str] = Field(default_factory=list)


class OutreachTemplateGenerateResponse(BaseModel):
    subject: str
    body_text: str
    body_html: str
    variables_used: list[str]


@router.post("/templates/generate-draft", response_model=OutreachTemplateGenerateResponse)
def generate_template_draft(body: OutreachTemplateGenerateRequest):
    if not body.brief.strip():
        raise HTTPException(status_code=422, detail="brief must not be empty")
    db = SessionLocal()
    try:
        job = _get_or_404(db, Job, body.job_id, "Job")
        payload = _generate_outreach_template_draft(
            brief=body.brief,
            job=job,
            variables_allowed=body.variables_allowed,
        )
        normalized_text, normalized_html = normalize_rich_message(
            body_text=payload.get("body_text") or "",
            body_html=payload.get("body_html") or "",
        )
        return OutreachTemplateGenerateResponse(
            subject=(payload.get("subject") or "").strip(),
            body_text=normalized_text,
            body_html=normalized_html,
            variables_used=list(payload.get("variables_used") or []),
        )
    finally:
        db.close()
```

- [ ] **Step 5: Run the backend tests and verify they pass**

Run: `pytest backend/tests/test_outreach_endpoints.py -k "generate_outreach_template_draft or manual_content_source" -v`

Expected: PASS for the new targeted cases.

### Task 3: Add frontend API coverage for templates workspace generation

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/endpoints/outreach.ts`
- Test: `frontend/src/api/endpoints/outreach.ts`

- [ ] **Step 1: Write the failing API shape expectations in the route-level tests that will consume them**

```typescript
expectTypeOf(api.outreach.generateTemplateDraft).toBeFunction();
```

- [ ] **Step 2: Run the frontend type-aware test or build step to verify the method is missing**

Run: `npm --prefix frontend exec tsc --noEmit`

Expected: FAIL once the consumer code references `generateTemplateDraft` and `manual`.

- [ ] **Step 3: Add the new API request/response types and endpoint method**

```typescript
export interface OutreachTemplateGenerateRequest {
  job_id: string;
  brief: string;
  variables_allowed: string[];
}

export interface OutreachTemplateGenerateResponse {
  subject: string;
  body_text: string;
  body_html: string;
  variables_used: string[];
}
```

```typescript
async generateTemplateDraft(
  body: OutreachTemplateGenerateRequest,
): Promise<OutreachTemplateGenerateResponse> {
  const { data } = await client.post<OutreachTemplateGenerateResponse>(
    "/outreach/templates/generate-draft",
    body,
  );
  return data;
}
```

- [ ] **Step 4: Re-run the frontend typecheck and verify the API surface compiles**

Run: `npm --prefix frontend exec tsc --noEmit`

Expected: PASS for the new API types, or fail only on unrelated pre-existing repository issues that must be reported separately.

### Task 4: Split the frontend route into messages and templates workspaces

**Files:**
- Modify: `frontend/src/routes/index.ts`
- Modify: `frontend/src/components/layout/Sidebar.tsx`
- Modify: `frontend/src/routes/outreach.tsx`
- Create: `frontend/src/routes/outreach-templates.tsx`
- Test: `frontend/tests/e2e/`

- [ ] **Step 1: Write the failing route-level tests for split workspaces**

```typescript
test("outreach messages workspace hides AI template generation controls", async ({ page }) => {
  await page.goto("/outreach");
  await expect(page.getByText("Generate once")).toHaveCount(0);
});

test("outreach templates workspace shows AI brief controls", async ({ page }) => {
  await page.goto("/outreach/templates");
  await expect(page.getByText("Generate once")).toBeVisible();
});
```

- [ ] **Step 2: Run the targeted frontend tests and verify they fail**

Run: `npm --prefix frontend run test:e2e -- outreach`

Expected: FAIL because `/outreach/templates` does not exist and `/outreach` still mixes both flows.

- [ ] **Step 3: Add the new route and shared outreach sub-navigation**

```typescript
export const routes = {
  outreach: "/outreach",
  outreachTemplates: "/outreach/templates",
};
```

```tsx
<NavLink to={routes.outreach}>Messages</NavLink>
<NavLink to={routes.outreachTemplates}>Templates</NavLink>
```

- [ ] **Step 4: Simplify `frontend/src/routes/outreach.tsx` into a messages-only workspace**

```tsx
const [sourceMode, setSourceMode] = useState<"blank" | "template">("blank");

const createPayload = {
  candidate_profile_id: candidateId,
  created_by_user_id: userId ?? "",
  content_source: sourceMode === "template" ? "template" : "manual",
  subject: subject.trim(),
  body_html: bodyHtml.trim(),
  body_text: bodyText.trim() || htmlToPlainText(bodyHtml),
  template_id: sourceMode === "template" ? templateId || null : null,
};
```

- [ ] **Step 5: Create `frontend/src/routes/outreach-templates.tsx` for listing, creating, and editing templates**

```tsx
const generateMutation = useMutation({
  mutationFn: () =>
    api.outreach.generateTemplateDraft({
      job_id: selectedJobId!,
      brief: aiBrief.trim(),
      variables_allowed: TEMPLATE_VARIABLES.map((item) => item.key),
    }),
  onSuccess: (draft) => {
    setSubject(draft.subject);
    setBodyHtml(draft.body_html);
    setBodyText(draft.body_text);
  },
});
```

- [ ] **Step 6: Re-run the targeted route tests and verify the split behavior passes**

Run: `npm --prefix frontend run test:e2e -- outreach`

Expected: PASS for the workspace split tests, or clear unrelated failures only.

### Task 5: Add focused frontend interaction tests for template-first message creation

**Files:**
- Create: `frontend/tests/e2e/outreach-workspaces.spec.ts`
- Test: `frontend/tests/e2e/outreach-workspaces.spec.ts`

- [ ] **Step 1: Write the failing interaction test for applying a template to a message**

```typescript
test("new message can load an existing template into the editor", async ({ page }) => {
  await page.goto("/outreach");
  await page.getByRole("button", { name: "New message" }).click();
  await page.getByLabel("Use template").click();
  await page.getByLabel("Template").selectOption({ label: "Warm intro" });
  await expect(page.getByDisplayValue("Warm intro subject")).toBeVisible();
});
```

- [ ] **Step 2: Run the targeted test and verify it fails before the final UI wiring**

Run: `npm --prefix frontend run test:e2e -- outreach-workspaces`

Expected: FAIL because the modal does not yet expose the separated source flow cleanly.

- [ ] **Step 3: Finish the modal wiring and template workspace polish**

```tsx
{sourceMode === "template" ? (
  <select value={templateId} onChange={handleTemplateSelect}>...</select>
) : null}
```

```tsx
<textarea
  value={aiBrief}
  onChange={(event) => setAiBrief(event.target.value)}
  placeholder="Describe the tone, audience, and what the recruiter should mention."
/>
```

- [ ] **Step 4: Re-run the targeted interaction test and verify it passes**

Run: `npm --prefix frontend run test:e2e -- outreach-workspaces`

Expected: PASS.

### Task 6: Run final verification for touched backend and frontend surfaces

**Files:**
- Verify: `backend/tests/test_outreach_endpoints.py`
- Verify: `frontend/tests/e2e/outreach-workspaces.spec.ts`
- Verify: touched frontend/backend files from previous tasks

- [ ] **Step 1: Run the backend outreach test file**

Run: `pytest backend/tests/test_outreach_endpoints.py -v`

Expected: PASS with 0 failures in the outreach endpoint suite.

- [ ] **Step 2: Run the focused frontend outreach workspace test suite**

Run: `npm --prefix frontend run test:e2e -- outreach-workspaces`

Expected: PASS with the new workspace and template flow scenarios green.

- [ ] **Step 3: Run a frontend typecheck for touched files**

Run: `npm --prefix frontend exec tsc --noEmit`

Expected: PASS, or only unrelated pre-existing repository issues that must be called out explicitly in the final report.

- [ ] **Step 4: Review the changed files before reporting completion**

Run: `git diff -- backend/src/models/enums.py backend/src/api/v1/endpoints/outreach.py backend/src/prompts/build_prompts.py backend/tests/test_outreach_endpoints.py frontend/src/api/types.ts frontend/src/api/endpoints/outreach.ts frontend/src/routes/index.ts frontend/src/routes/outreach.tsx frontend/src/routes/outreach-templates.tsx frontend/tests/e2e/outreach-workspaces.spec.ts`

Expected: Diff shows only the intended workspace split, AI generation, and test changes.
