# Voice Screening Interview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a job-scoped, public-link, structured AI voice screening workflow that lets recruiters send interview invitations after application, collects transcript evidence, and generates an HR review report.

**Architecture:** Replace candidate-scoped interview question sets with job-scoped interview templates, then layer invitation, session, transcript, and report workflows on top. Keep backend orchestration authoritative, keep public candidate access token-based, and isolate STT/TTS provider logic behind service adapters so MVP can start with low-cost browser voice infrastructure and upgrade later.

**Tech Stack:** FastAPI, SQLAlchemy, Alembic, Celery, Postgres JSONB, React, React Query, Vite, Playwright, pytest

---

## File Structure

### Backend files to create

- `backend/migrations/versions/20260522_0006_add_voice_interview_domain.py`
- `backend/src/models/interview_template.py`
- `backend/src/models/interview_invitation.py`
- `backend/src/models/interview_session.py`
- `backend/src/schemas/interview_template.py`
- `backend/src/schemas/interview_invitation.py`
- `backend/src/schemas/interview_public.py`
- `backend/src/schemas/interview_report.py`
- `backend/src/services/interview_template_service.py`
- `backend/src/services/interview_invitation_service.py`
- `backend/src/services/interview_session_service.py`
- `backend/src/services/interview_report_service.py`
- `backend/src/services/voice_provider.py`
- `backend/src/api/v1/endpoints/interview_templates.py`
- `backend/src/api/v1/endpoints/interview_public.py`
- `backend/src/api/v1/endpoints/interview_reports.py`
- `backend/tests/test_interview_template_endpoints.py`
- `backend/tests/test_interview_public_endpoints.py`
- `backend/tests/test_interview_report_service.py`
- `backend/tests/test_voice_provider.py`

### Backend files to modify

- `backend/src/models/entities.py`
- `backend/src/models/job.py`
- `backend/src/models/candidate_profile.py`
- `backend/src/models/job_matching.py`
- `backend/src/api/v1/api.py`
- `backend/src/core/config.py`
- `backend/src/services/llm_service.py`
- `backend/worker/tasks.py`
- `backend/tests/conftest.py`

### Frontend files to create

- `frontend/src/api/endpoints/interviewTemplates.ts`
- `frontend/src/api/endpoints/interviewInvitations.ts`
- `frontend/src/api/endpoints/interviewPublic.ts`
- `frontend/src/api/endpoints/interviewReports.ts`
- `frontend/src/routes/interviews/templates.tsx`
- `frontend/src/routes/interviews/template-detail.tsx`
- `frontend/src/routes/interviews/report.tsx`
- `frontend/src/routes/public-interview.tsx`
- `frontend/src/components/interviews/TemplateEditor.tsx`
- `frontend/src/components/interviews/InvitationSendDialog.tsx`
- `frontend/src/components/interviews/ReportView.tsx`
- `frontend/src/components/interviews/PublicInterviewShell.tsx`
- `frontend/tests/e2e/interview-voice-mvp.spec.ts`

### Frontend files to modify

- `frontend/src/api/index.ts`
- `frontend/src/api/types.ts`
- `frontend/src/router.tsx`
- `frontend/src/routes/index.ts`
- `frontend/src/components/layout/TopBar.tsx`
- `frontend/src/routes/candidates/detail.tsx`
- `frontend/src/routes/interview-questions/list.tsx`
- `frontend/src/routes/interview-questions/detail.tsx`

### Documentation files to create or modify

- `docs/FEATURE_TEST_PLAN.md`
- `docs/superpowers/specs/2026-05-22-voice-screening-interview-design.md`

## Task 1: Establish the interview domain and database schema

**Files:**
- Create: `backend/migrations/versions/20260522_0006_add_voice_interview_domain.py`
- Create: `backend/src/models/interview_template.py`
- Create: `backend/src/models/interview_invitation.py`
- Create: `backend/src/models/interview_session.py`
- Modify: `backend/src/models/entities.py`
- Modify: `backend/src/models/job.py`
- Modify: `backend/src/models/candidate_profile.py`
- Modify: `backend/src/models/job_matching.py`
- Test: `backend/tests/test_interview_template_endpoints.py`

- [ ] **Step 1: Write the failing model smoke test**

```python
def test_interview_template_and_invitation_tables_exist(db_session):
    from sqlalchemy import inspect

    inspector = inspect(db_session.bind)
    table_names = set(inspector.get_table_names())

    assert "interview_templates" in table_names
    assert "interview_invitations" in table_names
    assert "interview_sessions" in table_names
    assert "interview_response_items" in table_names
    assert "interview_transcript_turns" in table_names
    assert "interview_reports" in table_names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_interview_template_endpoints.py::test_interview_template_and_invitation_tables_exist -v`
Expected: FAIL with missing tables or missing test module.

- [ ] **Step 3: Add the schema and SQLAlchemy models**

```python
# backend/src/models/interview_template.py
class InterviewTemplate(Base):
    __tablename__ = "interview_templates"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    language_code: Mapped[str] = mapped_column(String(16), nullable=False, default="vi-VN")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="draft")
    intro_script: Mapped[str] = mapped_column(Text, nullable=False, default="")
    closing_script: Mapped[str] = mapped_column(Text, nullable=False, default="")
    question_payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    report_rubric: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
```

```python
# backend/src/models/interview_invitation.py
class InterviewInvitation(Base):
    __tablename__ = "interview_invitations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    candidate_profile_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("candidate_profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    interview_template_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("interview_templates.id", ondelete="CASCADE"), nullable=False, index=True)
    public_token: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
```

```python
# backend/migrations/versions/20260522_0006_add_voice_interview_domain.py
def upgrade():
    op.create_table(...)
    op.create_index("ix_interview_invitations_public_token", "interview_invitations", ["public_token"], unique=True)
```

- [ ] **Step 4: Run targeted tests and migration smoke**

Run: `pytest backend/tests/test_interview_template_endpoints.py::test_interview_template_and_invitation_tables_exist -v`
Expected: PASS

Run: `pytest backend/tests/test_interview_template_endpoints.py -v`
Expected: FAIL later on endpoint imports or missing schemas, but table smoke should pass.

- [ ] **Step 5: Commit**

```bash
git add backend/migrations/versions/20260522_0006_add_voice_interview_domain.py backend/src/models/interview_template.py backend/src/models/interview_invitation.py backend/src/models/interview_session.py backend/src/models/entities.py backend/src/models/job.py backend/src/models/candidate_profile.py backend/src/models/job_matching.py backend/tests/test_interview_template_endpoints.py
git commit -m "feat: add voice interview domain models"
```

## Task 2: Add recruiter-side template and invitation APIs

**Files:**
- Create: `backend/src/schemas/interview_template.py`
- Create: `backend/src/schemas/interview_invitation.py`
- Create: `backend/src/services/interview_template_service.py`
- Create: `backend/src/services/interview_invitation_service.py`
- Create: `backend/src/api/v1/endpoints/interview_templates.py`
- Modify: `backend/src/api/v1/api.py`
- Modify: `backend/src/services/job_scope.py`
- Test: `backend/tests/test_interview_template_endpoints.py`

- [ ] **Step 1: Write failing API tests for recruiter flows**

```python
def test_create_template_for_job(client, auth_headers, seeded_job):
    response = client.post(
        f"/api/v1/jobs/{seeded_job.id}/interview-templates",
        headers=auth_headers,
        json={
            "name": "Phone Screen v1",
            "language_code": "vi-VN",
            "intro_script": "Xin chao, day la AI interviewer.",
            "closing_script": "Cam on ban da tham gia.",
            "question_payload": {
                "questions": [
                    {"id": "q1", "text": "Hay gioi thieu ban than", "max_duration_sec": 120}
                ]
            },
            "report_rubric": {"focus": ["clarity", "relevance"]},
        },
    )
    assert response.status_code == 201
    assert response.json()["name"] == "Phone Screen v1"
```

```python
def test_create_invitation_for_candidate(client, auth_headers, seeded_candidate, seeded_template):
    response = client.post(
        "/api/v1/interview-invitations",
        headers=auth_headers,
        json={
            "candidate_profile_id": str(seeded_candidate.id),
            "interview_template_id": str(seeded_template.id),
            "expires_in_hours": 72,
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "pending"
    assert body["public_url"].startswith("http")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest backend/tests/test_interview_template_endpoints.py::test_create_template_for_job backend/tests/test_interview_template_endpoints.py::test_create_invitation_for_candidate -v`
Expected: FAIL with missing router or schema.

- [ ] **Step 3: Implement recruiter schemas, services, and routers**

```python
# backend/src/schemas/interview_template.py
class InterviewTemplateCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    language_code: str = Field(default="vi-VN")
    intro_script: str
    closing_script: str
    question_payload: dict
    report_rubric: dict = Field(default_factory=dict)

class InterviewTemplateResponse(BaseModel):
    id: str
    job_id: str
    name: str
    language_code: str
    status: str
    question_payload: dict
```

```python
# backend/src/api/v1/endpoints/interview_templates.py
@router.post("/jobs/{job_id}/interview-templates", response_model=InterviewTemplateResponse, status_code=201)
def create_template(job_id: uuid.UUID, body: InterviewTemplateCreateRequest, current_user=Depends(get_current_user)):
    template = interview_template_service.create_template(job_id=job_id, body=body, current_user=current_user)
    return serialize_template(template)

@router.post("/interview-invitations", response_model=InterviewInvitationResponse, status_code=201)
def create_invitation(body: InterviewInvitationCreateRequest, current_user=Depends(get_current_user)):
    invitation = interview_invitation_service.create_invitation(body=body, current_user=current_user)
    return serialize_invitation(invitation)
```

```python
# backend/src/services/interview_invitation_service.py
def build_public_interview_url(token: str) -> str:
    return f"{settings.frontend_base_url.rstrip('/')}/interviews/public/{token}"
```

- [ ] **Step 4: Run recruiter API tests**

Run: `pytest backend/tests/test_interview_template_endpoints.py -v`
Expected: PASS for create/list/update invitation lifecycle tests added in this file.

- [ ] **Step 5: Commit**

```bash
git add backend/src/schemas/interview_template.py backend/src/schemas/interview_invitation.py backend/src/services/interview_template_service.py backend/src/services/interview_invitation_service.py backend/src/api/v1/endpoints/interview_templates.py backend/src/api/v1/api.py backend/src/services/job_scope.py backend/tests/test_interview_template_endpoints.py
git commit -m "feat: add recruiter interview template APIs"
```

## Task 3: Build public interview session APIs and voice-provider abstraction

**Files:**
- Create: `backend/src/schemas/interview_public.py`
- Create: `backend/src/services/interview_session_service.py`
- Create: `backend/src/services/voice_provider.py`
- Create: `backend/src/api/v1/endpoints/interview_public.py`
- Modify: `backend/src/api/v1/api.py`
- Modify: `backend/src/core/config.py`
- Test: `backend/tests/test_interview_public_endpoints.py`
- Test: `backend/tests/test_voice_provider.py`

- [ ] **Step 1: Write failing public-session and provider tests**

```python
def test_public_token_can_start_session(client, seeded_invitation):
    response = client.post(f"/api/v1/public/interview/{seeded_invitation.public_token}/start")
    assert response.status_code == 201
    body = response.json()
    assert body["session"]["status"] == "in_progress"
    assert body["template"]["question_payload"]["questions"][0]["id"] == "q1"
```

```python
def test_public_token_cannot_start_when_expired(client, expired_invitation):
    response = client.post(f"/api/v1/public/interview/{expired_invitation.public_token}/start")
    assert response.status_code == 410
```

```python
def test_fake_voice_provider_normalizes_transcript_event():
    provider = FakeVoiceProvider()
    event = provider.normalize_transcript_event({"text": "xin chao", "is_final": True, "question_id": "q1"})
    assert event.text == "xin chao"
    assert event.is_final is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest backend/tests/test_interview_public_endpoints.py backend/tests/test_voice_provider.py -v`
Expected: FAIL with missing endpoint or provider module.

- [ ] **Step 3: Implement public start/events/complete flow and provider interface**

```python
# backend/src/services/voice_provider.py
class NormalizedTranscriptEvent(BaseModel):
    text: str
    is_final: bool
    question_id: str | None = None
    started_at_ms: int | None = None
    ended_at_ms: int | None = None

class VoiceProvider(Protocol):
    def normalize_transcript_event(self, payload: dict) -> NormalizedTranscriptEvent: ...

class FakeVoiceProvider:
    def normalize_transcript_event(self, payload: dict) -> NormalizedTranscriptEvent:
        return NormalizedTranscriptEvent(**payload)
```

```python
# backend/src/api/v1/endpoints/interview_public.py
@router.post("/public/interview/{token}/start", response_model=PublicInterviewStartResponse, status_code=201)
def start_public_interview(token: str, body: PublicInterviewStartRequest | None = None):
    session, invitation, template = interview_session_service.start_session(token=token, body=body)
    return {
        "invitation": serialize_public_invitation(invitation),
        "session": serialize_session(session),
        "template": serialize_template(template),
    }

@router.post("/public/interview/{token}/events", status_code=202)
def ingest_public_interview_event(token: str, body: PublicInterviewEventRequest):
    interview_session_service.ingest_event(token=token, body=body)
```

```python
# backend/src/services/interview_session_service.py
def ingest_event(*, token: str, body: PublicInterviewEventRequest) -> None:
    normalized = voice_provider.normalize_transcript_event(body.model_dump())
    append_transcript_turn(...)
    update_response_item(...)
```

- [ ] **Step 4: Run public-session tests**

Run: `pytest backend/tests/test_interview_public_endpoints.py backend/tests/test_voice_provider.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/schemas/interview_public.py backend/src/services/interview_session_service.py backend/src/services/voice_provider.py backend/src/api/v1/endpoints/interview_public.py backend/src/api/v1/api.py backend/src/core/config.py backend/tests/test_interview_public_endpoints.py backend/tests/test_voice_provider.py
git commit -m "feat: add public voice interview session APIs"
```

## Task 4: Generate HR reports asynchronously from transcript evidence

**Files:**
- Create: `backend/src/schemas/interview_report.py`
- Create: `backend/src/services/interview_report_service.py`
- Modify: `backend/src/services/llm_service.py`
- Modify: `backend/worker/tasks.py`
- Test: `backend/tests/test_interview_report_service.py`

- [ ] **Step 1: Write failing report generation tests**

```python
def test_generate_markdown_report_from_completed_session(db_session, completed_interview_session):
    report = generate_interview_report(session_id=completed_interview_session.id)
    assert "# Interview Report" in report.summary_markdown
    assert report.structured_summary_json["completion_status"] == "completed"
    assert report.structured_summary_json["question_summaries"][0]["question_id"] == "q1"
```

```python
def test_report_generation_stays_descriptive(db_session, completed_interview_session):
    report = generate_interview_report(session_id=completed_interview_session.id)
    assert "reject" not in report.summary_markdown.lower()
    assert "hire immediately" not in report.summary_markdown.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest backend/tests/test_interview_report_service.py -v`
Expected: FAIL with missing report service.

- [ ] **Step 3: Implement report prompt building and Celery task**

```python
# backend/src/services/interview_report_service.py
def build_interview_report_prompt(*, candidate_name: str, job_title: str, questions: list[dict], transcript_turns: list[dict]) -> str:
    payload = {
        "candidateName": candidate_name,
        "jobTitle": job_title,
        "questions": questions,
        "transcriptTurns": transcript_turns,
        "responseFormat": {
            "completion_status": "completed",
            "overall_summary": "string",
            "question_summaries": [{"question_id": "q1", "summary": "string", "evidence": ["string"]}],
            "strengths": ["string"],
            "concerns": ["string"],
            "follow_up_topics": ["string"],
        },
    }
    return "Return JSON only for an HR interview summary. Stay descriptive, never recommend hiring decisions.\n\n" + json.dumps(payload)
```

```python
# backend/worker/tasks.py
@celery_app.task(name="worker.tasks.generate_interview_report")
def generate_interview_report_task(session_id: str):
    from src.services.interview_report_service import generate_interview_report
    return generate_interview_report(session_id=uuid.UUID(session_id))
```

```python
# backend/src/services/interview_session_service.py
def complete_session(...):
    ...
    generate_interview_report_task.delay(str(session.id))
```

- [ ] **Step 4: Run report tests**

Run: `pytest backend/tests/test_interview_report_service.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/schemas/interview_report.py backend/src/services/interview_report_service.py backend/src/services/llm_service.py backend/worker/tasks.py backend/tests/test_interview_report_service.py
git commit -m "feat: add interview report generation"
```

## Task 5: Replace the recruiter-facing UI with template, invitation, and report flows

**Files:**
- Create: `frontend/src/api/endpoints/interviewTemplates.ts`
- Create: `frontend/src/api/endpoints/interviewInvitations.ts`
- Create: `frontend/src/api/endpoints/interviewReports.ts`
- Create: `frontend/src/routes/interviews/templates.tsx`
- Create: `frontend/src/routes/interviews/template-detail.tsx`
- Create: `frontend/src/routes/interviews/report.tsx`
- Create: `frontend/src/components/interviews/TemplateEditor.tsx`
- Create: `frontend/src/components/interviews/InvitationSendDialog.tsx`
- Create: `frontend/src/components/interviews/ReportView.tsx`
- Modify: `frontend/src/api/index.ts`
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/router.tsx`
- Modify: `frontend/src/routes/index.ts`
- Modify: `frontend/src/components/layout/TopBar.tsx`
- Modify: `frontend/src/routes/candidates/detail.tsx`
- Modify: `frontend/src/routes/interview-questions/list.tsx`
- Modify: `frontend/src/routes/interview-questions/detail.tsx`
- Test: `frontend/tests/e2e/interview-voice-mvp.spec.ts`

- [ ] **Step 1: Write the failing recruiter workflow E2E**

```ts
test("recruiter can create template, send invitation, and open report", async ({ page }) => {
  await page.goto("/interviews/templates");
  await page.getByRole("button", { name: "New template" }).click();
  await page.getByLabel("Template name").fill("Screening v1");
  await page.getByLabel("Question 1").fill("Hay gioi thieu ban than");
  await page.getByRole("button", { name: "Save template" }).click();
  await expect(page.getByText("Screening v1")).toBeVisible();
});
```

- [ ] **Step 2: Run the E2E to verify it fails**

Run: `npm run test:e2e -- interview-voice-mvp.spec.ts`
Expected: FAIL with missing route or missing page elements.

- [ ] **Step 3: Implement recruiter-facing API clients and UI routes**

```ts
// frontend/src/api/endpoints/interviewTemplates.ts
export const interviewTemplatesApi = {
  list(jobId: string) {
    return client.get(`/jobs/${jobId}/interview-templates`).then((r) => r.data);
  },
  create(jobId: string, body: InterviewTemplateCreateRequest) {
    return client.post(`/jobs/${jobId}/interview-templates`, body).then((r) => r.data);
  },
};
```

```tsx
// frontend/src/routes/interviews/templates.tsx
export default function InterviewTemplatesRoute() {
  const activeJobId = useActiveJobId();
  const { data } = useQuery({
    queryKey: ["interview-templates", activeJobId],
    queryFn: () => api.interviewTemplates.list(activeJobId!),
    enabled: !!activeJobId,
  });

  return <TemplateListView templates={data?.items ?? []} />;
}
```

```tsx
// frontend/src/routes/candidates/detail.tsx
<Button onClick={() => setInviteDialogOpen(true)}>Send interview</Button>
<InvitationSendDialog candidateId={candidate.id} />
```

- [ ] **Step 4: Run recruiter UI tests**

Run: `npm run build`
Expected: PASS

Run: `npm run test:e2e -- interview-voice-mvp.spec.ts`
Expected: PASS for recruiter-side template and invitation flow, while public candidate flow can still fail until Task 6 lands.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api/endpoints/interviewTemplates.ts frontend/src/api/endpoints/interviewInvitations.ts frontend/src/api/endpoints/interviewReports.ts frontend/src/routes/interviews/templates.tsx frontend/src/routes/interviews/template-detail.tsx frontend/src/routes/interviews/report.tsx frontend/src/components/interviews/TemplateEditor.tsx frontend/src/components/interviews/InvitationSendDialog.tsx frontend/src/components/interviews/ReportView.tsx frontend/src/api/index.ts frontend/src/api/types.ts frontend/src/router.tsx frontend/src/routes/index.ts frontend/src/components/layout/TopBar.tsx frontend/src/routes/candidates/detail.tsx frontend/src/routes/interview-questions/list.tsx frontend/src/routes/interview-questions/detail.tsx frontend/tests/e2e/interview-voice-mvp.spec.ts
git commit -m "feat: add recruiter interview template UI"
```

## Task 6: Build the public browser interview page and candidate session client

**Files:**
- Create: `frontend/src/api/endpoints/interviewPublic.ts`
- Create: `frontend/src/routes/public-interview.tsx`
- Create: `frontend/src/components/interviews/PublicInterviewShell.tsx`
- Modify: `frontend/src/router.tsx`
- Modify: `frontend/src/routes/index.ts`
- Test: `frontend/tests/e2e/interview-voice-mvp.spec.ts`

- [ ] **Step 1: Extend the E2E with candidate-side completion**

```ts
test("candidate can open public link and complete a structured interview", async ({ page }) => {
  await page.goto("/interviews/public/test-token");
  await expect(page.getByText("You are speaking with an AI interviewer")).toBeVisible();
  await page.getByRole("button", { name: "Start interview" }).click();
  await expect(page.getByText("Question 1")).toBeVisible();
  await page.getByRole("button", { name: "Simulate answer" }).click();
  await page.getByRole("button", { name: "Finish interview" }).click();
  await expect(page.getByText("Interview completed")).toBeVisible();
});
```

- [ ] **Step 2: Run the E2E to verify it fails**

Run: `npm run test:e2e -- interview-voice-mvp.spec.ts`
Expected: FAIL with missing public interview route.

- [ ] **Step 3: Implement the public interview route and event loop**

```ts
// frontend/src/api/endpoints/interviewPublic.ts
export const interviewPublicApi = {
  start(token: string) {
    return client.post(`/public/interview/${token}/start`).then((r) => r.data);
  },
  sendEvent(token: string, body: PublicInterviewEventRequest) {
    return client.post(`/public/interview/${token}/events`, body).then((r) => r.data);
  },
  complete(token: string, body: PublicInterviewCompleteRequest) {
    return client.post(`/public/interview/${token}/complete`, body).then((r) => r.data);
  },
};
```

```tsx
// frontend/src/components/interviews/PublicInterviewShell.tsx
export function PublicInterviewShell({ token }: { token: string }) {
  const [session, setSession] = useState<PublicInterviewSession | null>(null);
  const [questionIndex, setQuestionIndex] = useState(0);

  const start = async () => setSession(await api.interviewPublic.start(token));
  const sendSimulatedAnswer = async () => {
    const question = session!.template.question_payload.questions[questionIndex];
    await api.interviewPublic.sendEvent(token, {
      event_type: "transcript.final",
      question_id: question.id,
      text: "Ung vien tra loi mau",
      is_final: true,
    });
    setQuestionIndex((value) => value + 1);
  };
```

```tsx
// frontend/src/routes/public-interview.tsx
export default function PublicInterviewRoute() {
  const { token } = useParams<{ token: string }>();
  return token ? <PublicInterviewShell token={token} /> : <EmptyState heading="Invalid interview link" />;
}
```

- [ ] **Step 4: Run build and full E2E**

Run: `npm run build`
Expected: PASS

Run: `npm run test:e2e -- interview-voice-mvp.spec.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api/endpoints/interviewPublic.ts frontend/src/routes/public-interview.tsx frontend/src/components/interviews/PublicInterviewShell.tsx frontend/src/router.tsx frontend/src/routes/index.ts frontend/tests/e2e/interview-voice-mvp.spec.ts
git commit -m "feat: add public voice interview flow"
```

## Task 7: Harden configuration, regression coverage, and delivery docs

**Files:**
- Modify: `backend/tests/conftest.py`
- Modify: `docs/FEATURE_TEST_PLAN.md`
- Modify: `docs/superpowers/specs/2026-05-22-voice-screening-interview-design.md`
- Test: `backend/tests/test_interview_template_endpoints.py`
- Test: `backend/tests/test_interview_public_endpoints.py`
- Test: `backend/tests/test_interview_report_service.py`
- Test: `frontend/tests/e2e/interview-voice-mvp.spec.ts`

- [ ] **Step 1: Add failing integration assertions for config and regression safety**

```python
def test_interview_frontend_url_builder_uses_config(settings):
    settings.frontend_base_url = "http://localhost:5173"
    assert build_public_interview_url("abc123") == "http://localhost:5173/interviews/public/abc123"
```

```python
def test_completed_invitation_cannot_restart(client, completed_invitation):
    response = client.post(f"/api/v1/public/interview/{completed_invitation.public_token}/start")
    assert response.status_code == 409
```

- [ ] **Step 2: Run regression tests to verify new edge cases fail**

Run: `pytest backend/tests/test_interview_template_endpoints.py backend/tests/test_interview_public_endpoints.py -v`
Expected: FAIL on missing config edge cases or restart guard.

- [ ] **Step 3: Implement final hardening and update delivery docs**

```python
# backend/src/services/interview_session_service.py
if invitation.status == "completed":
    raise HTTPException(status_code=409, detail="Interview already completed")
```

```md
# docs/FEATURE_TEST_PLAN.md
- Voice screening interview template CRUD
- Public interview invitation lifecycle
- Public browser interview completion
- HR interview report rendering
```

```md
# docs/superpowers/specs/2026-05-22-voice-screening-interview-design.md
- Implementation note: MVP public browser client uses simulated answer controls in E2E and provider abstraction for future realtime microphone streaming.
```

- [ ] **Step 4: Run full verification**

Run: `pytest backend/tests/test_interview_template_endpoints.py backend/tests/test_interview_public_endpoints.py backend/tests/test_interview_report_service.py -v`
Expected: PASS

Run: `npm run build && npm run test:e2e -- interview-voice-mvp.spec.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/tests/conftest.py docs/FEATURE_TEST_PLAN.md docs/superpowers/specs/2026-05-22-voice-screening-interview-design.md backend/tests/test_interview_template_endpoints.py backend/tests/test_interview_public_endpoints.py backend/tests/test_interview_report_service.py frontend/tests/e2e/interview-voice-mvp.spec.ts
git commit -m "test: harden voice interview coverage and docs"
```

## Spec Coverage Check

- Job-scoped interview template support is covered by Tasks 1 and 2.
- Public invitation sent after application is covered by Tasks 2 and 5.
- Structured browser voice session is covered by Tasks 3 and 6.
- Transcript capture and per-question answer storage is covered by Tasks 1 and 3.
- HR-facing markdown and structured report is covered by Task 4 and recruiter UI in Task 5.
- Cost-aware provider abstraction and future upgrade path is covered by Tasks 3 and 7.
- Compliance-oriented descriptive reporting and no auto-decisioning is covered by Tasks 4 and 7.

## Type Consistency Check

- Core names remain `InterviewTemplate`, `InterviewInvitation`, `InterviewSession`, `InterviewResponseItem`, `InterviewTranscriptTurn`, and `InterviewReport` throughout the plan.
- Public route prefix remains `/public/interview/{token}` throughout the plan.
- Frontend public route remains `/interviews/public/:token` throughout the plan.
- Report generation entrypoint remains `generate_interview_report` and `generate_interview_report_task`.

## Delivery Notes

- Do not delete the existing `interview_question_sets` table in the first delivery. Keep migration additive, then add a later cleanup migration only after the recruiter UI fully transitions.
- Keep E2E deterministic by using simulated transcript events instead of live microphone capture in CI.
- Keep the first provider implementation simple and testable. Realtime microphone streaming can be layered onto the same public session API once the domain and review workflow are stable.
