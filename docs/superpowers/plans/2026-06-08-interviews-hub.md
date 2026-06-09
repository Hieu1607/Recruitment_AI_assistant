# Interviews Hub Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a recruiter-facing `/interviews` hub for invitation-link management, add revoke support for invitations, and make `/interviews/templates` a child area with full template CRUD.

**Architecture:** Extend the existing interview template and invitation API surface with two missing recruiter actions: delete template and revoke invitation. On the frontend, add a dedicated `/interviews` route that lists job-scoped invitations and links to `/interviews/templates`, while reusing the existing invitation dialog, template editor, and report route.

**Tech Stack:** FastAPI, SQLAlchemy, Pydantic, pytest, React, React Router, TanStack Query, TypeScript, Playwright, Vite.

---

## File Structure

**Backend**

- Modify: `backend/src/api/v1/endpoints/interview_templates.py`
  Add recruiter endpoints for template delete and invitation revoke.
- Modify: `backend/src/services/interview_template_service.py`
  Add template delete service logic.
- Modify: `backend/src/services/interview_invitation_service.py`
  Add invitation revoke service logic and serialization reuse.
- Modify: `backend/src/schemas/interview_invitation.py`
  Add revoke response shape only if current response model cannot be reused.
- Modify: `backend/tests/test_interview_template_endpoints.py`
  Cover delete and revoke behavior.

**Frontend**

- Modify: `frontend/src/routes/index.ts`
  Add `/interviews` route helper and clean up interview route naming.
- Modify: `frontend/src/router.tsx`
  Register authenticated `/interviews` before the public `/interviews/:token` matcher causes ambiguity.
- Create: `frontend/src/routes/interviews/index.tsx`
  Build the interview hub page.
- Modify: `frontend/src/api/endpoints/interviewInvitations.ts`
  Add revoke client method.
- Modify: `frontend/src/api/endpoints/interviewTemplates.ts`
  Add delete client method.
- Modify: `frontend/src/api/types.ts`
  Add delete response types if needed.
- Modify: `frontend/src/components/layout/Sidebar.tsx`
  Replace `Interview Prep` with `Interviews`.
- Modify: `frontend/src/components/layout/TopBar.tsx`
  Add breadcrumb/title support for `/interviews`.
- Modify: `frontend/src/routes/interviews/templates.tsx`
  Add in-module navigation link back to `/interviews`.
- Modify: `frontend/src/routes/interviews/template-detail.tsx`
  Add delete flow with confirmation and redirect back to templates list.
- Modify: `frontend/tests/e2e/interview-voice-mvp.spec.ts`
  Expand the existing interview flow to visit `/interviews`, revoke an invitation, and delete a template.

## Task 1: Backend Revoke And Delete Tests

**Files:**
- Modify: `backend/tests/test_interview_template_endpoints.py`

- [ ] **Step 1: Write the failing tests for template delete and invitation revoke**

```python
def test_recruiter_can_delete_unused_interview_template(
    api_client: TestClient,
    seeded_interview_domain,
):
    create_response = api_client.post(
        f"/api/v1/jobs/{seeded_interview_domain['primary_job_id']}/interview-templates",
        json={"name": "Delete Me", "status": "draft"},
    )
    template_id = create_response.json()["id"]

    delete_response = api_client.delete(f"/api/v1/interview-templates/{template_id}")

    assert delete_response.status_code == 200
    assert delete_response.json() == {"deleted": True, "template_id": template_id}


def test_recruiter_can_revoke_pending_interview_invitation(
    api_client: TestClient,
    seeded_interview_domain,
    monkeypatch,
):
    import worker.tasks as tasks_module

    monkeypatch.setattr(
        tasks_module,
        "send_interview_invitation_email",
        type("FakeTask", (), {"delay": staticmethod(lambda invitation_id: None)}),
        raising=False,
    )

    create_response = api_client.post(
        "/api/v1/interview-invitations",
        json={
            "job_id": str(seeded_interview_domain["primary_job_id"]),
            "candidate_profile_id": str(seeded_interview_domain["candidate_id"]),
            "interview_template_id": str(seeded_interview_domain["template_id"]),
        },
    )
    invitation_id = create_response.json()["id"]

    revoke_response = api_client.post(f"/api/v1/interview-invitations/{invitation_id}/revoke")

    assert revoke_response.status_code == 200
    assert revoke_response.json()["id"] == invitation_id
    assert revoke_response.json()["status"] == "cancelled"
    assert revoke_response.json()["cancelled_at"] is not None
```

- [ ] **Step 2: Run the targeted backend tests to verify they fail**

Run:

```bash
pytest backend/tests/test_interview_template_endpoints.py -k "delete_unused_interview_template or revoke_pending_interview_invitation" -v
```

Expected:

```text
FAILED ... 404/405 or attribute/route missing for delete/revoke
```

- [ ] **Step 3: Add one negative test for protected behavior**

```python
def test_recruiter_cannot_delete_template_with_existing_invitation(
    api_client: TestClient,
    seeded_interview_domain,
    monkeypatch,
):
    import worker.tasks as tasks_module

    monkeypatch.setattr(
        tasks_module,
        "send_interview_invitation_email",
        type("FakeTask", (), {"delay": staticmethod(lambda invitation_id: None)}),
        raising=False,
    )

    api_client.post(
        "/api/v1/interview-invitations",
        json={
            "job_id": str(seeded_interview_domain["primary_job_id"]),
            "candidate_profile_id": str(seeded_interview_domain["candidate_id"]),
            "interview_template_id": str(seeded_interview_domain["template_id"]),
        },
    )

    delete_response = api_client.delete(
        f"/api/v1/interview-templates/{seeded_interview_domain['template_id']}"
    )

    assert delete_response.status_code == 409
```

- [ ] **Step 4: Run the targeted backend tests again and verify the new negative test also fails for the right reason**

Run:

```bash
pytest backend/tests/test_interview_template_endpoints.py -k "delete_template or revoke_pending_interview_invitation" -v
```

Expected:

```text
FAILED ... route missing or service behavior missing, not fixture/setup errors
```

- [ ] **Step 5: Commit the red test state only if the repository workflow requires it**

```bash
git diff -- backend/tests/test_interview_template_endpoints.py
```

Expected:

```text
Shows only new delete/revoke test coverage
```

## Task 2: Backend Revoke And Delete Implementation

**Files:**
- Modify: `backend/src/services/interview_template_service.py`
- Modify: `backend/src/services/interview_invitation_service.py`
- Modify: `backend/src/api/v1/endpoints/interview_templates.py`
- Modify: `backend/src/schemas/interview_invitation.py` (only if needed)

- [ ] **Step 1: Add minimal service functions for delete and revoke**

```python
def delete_interview_template(db: Session, *, user_id: uuid.UUID, template_id: uuid.UUID) -> None:
    template = get_interview_template(db, user_id=user_id, template_id=template_id)
    if template.invitations:
        raise HTTPException(status_code=409, detail="Interview template is already in use")
    db.delete(template)
    db.commit()


def revoke_interview_invitation(
    db: Session, *, user_id: uuid.UUID, invitation_id: uuid.UUID
) -> InterviewInvitation:
    invitation = get_scoped_interview_invitation(db, user_id=user_id, invitation_id=invitation_id)
    if invitation.status in {"completed", "cancelled"}:
        return invitation
    invitation.status = "cancelled"
    invitation.cancelled_at = datetime.now(timezone.utc)
    db.add(invitation)
    db.commit()
    db.refresh(invitation)
    return invitation
```

- [ ] **Step 2: Wire the new endpoints in the interview endpoint module**

```python
@router.delete("/interview-templates/{template_id}")
def delete_single_interview_template(
    template_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    delete_interview_template(db, user_id=current_user.id, template_id=template_id)
    return {"deleted": True, "template_id": str(template_id)}


@router.post("/interview-invitations/{invitation_id}/revoke", response_model=InterviewInvitationResponse)
def revoke_single_interview_invitation(
    invitation_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    invitation = revoke_interview_invitation(db, user_id=current_user.id, invitation_id=invitation_id)
    return serialize_interview_invitation(invitation)
```

- [ ] **Step 3: Run the targeted backend tests to verify green**

Run:

```bash
pytest backend/tests/test_interview_template_endpoints.py -k "delete_template or revoke_pending_interview_invitation" -v
```

Expected:

```text
PASSED
```

- [ ] **Step 4: Run the full interview endpoint backend test file**

Run:

```bash
pytest backend/tests/test_interview_template_endpoints.py -v
```

Expected:

```text
All interview template endpoint tests pass with 0 failures
```

- [ ] **Step 5: Commit the backend implementation**

```bash
git add backend/src/api/v1/endpoints/interview_templates.py backend/src/services/interview_template_service.py backend/src/services/interview_invitation_service.py backend/tests/test_interview_template_endpoints.py
git commit -m "feat: add interview revoke and template delete endpoints"
```

## Task 3: Frontend API Surface And Routing

**Files:**
- Modify: `frontend/src/api/endpoints/interviewInvitations.ts`
- Modify: `frontend/src/api/endpoints/interviewTemplates.ts`
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/routes/index.ts`
- Modify: `frontend/src/router.tsx`
- Modify: `frontend/src/components/layout/Sidebar.tsx`
- Modify: `frontend/src/components/layout/TopBar.tsx`

- [ ] **Step 1: Add missing frontend API client methods**

```ts
export const interviewInvitationsApi = {
  async revoke(invitationId: string): Promise<InterviewInvitationResponse> {
    const { data } = await client.post<InterviewInvitationResponse>(
      `/interview-invitations/${invitationId}/revoke`,
    );
    return data;
  },
};

export const interviewTemplatesApi = {
  async remove(templateId: string): Promise<{ deleted: boolean; template_id: string }> {
    const { data } = await client.delete<{ deleted: boolean; template_id: string }>(
      `/interview-templates/${templateId}`,
    );
    return data;
  },
};
```

- [ ] **Step 2: Add route constants for the hub and normalize interview naming**

```ts
export const routes = {
  interviews: "/interviews",
  interviewTemplates: "/interviews/templates",
  interviewTemplateDetail: (id: string) => `/interviews/templates/${id}`,
  interviewReport: (interviewSessionId: string) => `/interviews/reports/${interviewSessionId}`,
  publicInterview: (token: string) => `/interviews/${token}`,
} as const;
```

- [ ] **Step 3: Register the authenticated hub route**

```tsx
{ path: routePatterns.interviews, ...lazy(() => import("@/routes/interviews")) },
{ path: routePatterns.interviewTemplates, ...lazy(() => import("@/routes/interviews/templates")) },
{ path: routePatterns.interviewTemplateDetail, ...lazy(() => import("@/routes/interviews/template-detail")) },
{ path: routePatterns.interviewReport, ...lazy(() => import("@/routes/interviews/report")) },
```

- [ ] **Step 4: Update navigation labels**

```tsx
const NAV_ITEMS = [
  { to: routes.interviews, label: "Interviews", icon: Mic2 },
];
```

- [ ] **Step 5: Run the frontend type/build check**

Run:

```bash
cd frontend
npm run build
```

Expected:

```text
vite build exits 0
```

## Task 4: Interviews Hub UI

**Files:**
- Create: `frontend/src/routes/interviews/index.tsx`
- Modify: `frontend/src/routes/interviews/templates.tsx`

- [ ] **Step 1: Write the failing UI expectation in the existing Playwright interview flow**

```ts
await page.goto(`${APP_URL}/interviews`);
await expect(page.getByRole("heading", { name: "Interviews" })).toBeVisible();
await expect(page.getByRole("link", { name: "Templates" })).toBeVisible();
await expect(page.getByText("Alice Interview")).toBeVisible();
```

- [ ] **Step 2: Run the Playwright interview spec to verify the new `/interviews` expectation fails**

Run:

```bash
cd frontend
npx playwright test tests/e2e/interview-voice-mvp.spec.ts --project=chromium
```

Expected:

```text
FAILED at /interviews because route/page does not exist yet
```

- [ ] **Step 3: Implement the hub route with minimal invitation table and module nav**

```tsx
export default function InterviewsRoute() {
  const selectedJobId = useSelectedJobId();
  const { data, isLoading } = useQuery({
    queryKey: ["interview-invitations", selectedJobId],
    queryFn: () => api.interviewInvitations.list(selectedJobId!),
    enabled: !!selectedJobId,
  });

  return (
    <div className="px-8 py-8 min-h-full">
      <div className="mx-auto max-w-6xl space-y-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="font-display text-[2rem] font-medium text-fg">Interviews</h1>
            <p className="mt-1 text-sm text-fg-muted">
              Manage interview links, statuses, and report access for the selected job.
            </p>
          </div>
          <Link to={routes.interviewTemplates} className="text-sm text-accent hover:underline">
            Templates
          </Link>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Add the selected-job empty state and invitation table**

```tsx
{!selectedJobId ? (
  <EmptyState
    heading="Select a job first"
    body="Interview links are scoped to the active job in the top bar."
  />
) : (
  <DataTable
    columns={columns}
    data={data?.items ?? []}
    loading={isLoading}
    emptyState={
      <EmptyState
        heading="No interview links"
        body="Create the first invitation for this job to start candidate interviews."
      />
    }
  />
)}
```

- [ ] **Step 5: Re-run the Playwright spec to verify `/interviews` now renders**

Run:

```bash
cd frontend
npx playwright test tests/e2e/interview-voice-mvp.spec.ts --project=chromium
```

Expected:

```text
The previous /interviews route failure is gone; later assertions may still fail until revoke/delete UI is added
```

## Task 5: Frontend Revoke And Template Delete UX

**Files:**
- Modify: `frontend/src/routes/interviews/index.tsx`
- Modify: `frontend/src/routes/interviews/template-detail.tsx`
- Modify: `frontend/tests/e2e/interview-voice-mvp.spec.ts`

- [ ] **Step 1: Extend the Playwright spec with revoke and delete expectations**

```ts
await page.getByRole("button", { name: /Revoke/i }).click();
await expect(page.getByText(/cancelled/i)).toBeVisible();

await page.goto(`${APP_URL}/interviews/templates/${template?.id}`);
await page.getByRole("button", { name: /Delete template/i }).click();
await page.getByRole("button", { name: /Confirm delete/i }).click();
await expect(page).toHaveURL(`${APP_URL}/interviews/templates`);
```

- [ ] **Step 2: Run the spec and verify it fails on missing revoke/delete actions**

Run:

```bash
cd frontend
npx playwright test tests/e2e/interview-voice-mvp.spec.ts --project=chromium
```

Expected:

```text
FAILED because revoke and delete controls do not exist yet
```

- [ ] **Step 3: Implement revoke mutation in the hub route**

```tsx
const revokeMutation = useMutation({
  mutationFn: (invitationId: string) => api.interviewInvitations.revoke(invitationId),
  onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: ["interview-invitations", selectedJobId] });
    toast.success("Interview link revoked");
  },
  onError: (error: Error) => toast.error(error.message || "Failed to revoke invitation"),
});
```

- [ ] **Step 4: Implement template delete in the detail route**

```tsx
const deleteMutation = useMutation({
  mutationFn: () => api.interviewTemplates.remove(id!),
  onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: ["interview-templates"] });
    toast.success("Interview template deleted");
    navigate(routes.interviewTemplates);
  },
  onError: (error: Error) => toast.error(error.message || "Failed to delete template"),
});
```

- [ ] **Step 5: Re-run the Playwright spec and verify the full recruiter flow passes**

Run:

```bash
cd frontend
npx playwright test tests/e2e/interview-voice-mvp.spec.ts --project=chromium
```

Expected:

```text
PASSED for interview hub, template CRUD, invitation revoke, public interview flow, and report access
```

## Task 6: Final Verification

**Files:**
- All modified interview hub files

- [ ] **Step 1: Run targeted backend verification**

```bash
pytest backend/tests/test_interview_template_endpoints.py -v
```

Expected:

```text
PASS with 0 failures
```

- [ ] **Step 2: Run targeted frontend verification**

```bash
cd frontend
npm run build
npx playwright test tests/e2e/interview-voice-mvp.spec.ts --project=chromium
```

Expected:

```text
Build exits 0 and Playwright passes
```

- [ ] **Step 3: Review final diff**

```bash
git diff --stat
git diff -- backend/src/api/v1/endpoints/interview_templates.py backend/src/services/interview_template_service.py backend/src/services/interview_invitation_service.py frontend/src/routes/interviews/index.tsx frontend/src/routes/interviews/template-detail.tsx frontend/src/routes/interviews/templates.tsx frontend/src/router.tsx frontend/src/routes/index.ts frontend/src/components/layout/Sidebar.tsx frontend/tests/e2e/interview-voice-mvp.spec.ts
```

Expected:

```text
Diff is limited to interviews hub, template CRUD, invitation revoke, and navigation updates
```

- [ ] **Step 4: Commit the finished implementation**

```bash
git add backend/src/api/v1/endpoints/interview_templates.py backend/src/services/interview_template_service.py backend/src/services/interview_invitation_service.py backend/tests/test_interview_template_endpoints.py frontend/src/api/endpoints/interviewInvitations.ts frontend/src/api/endpoints/interviewTemplates.ts frontend/src/api/types.ts frontend/src/routes/index.ts frontend/src/router.tsx frontend/src/routes/interviews/index.tsx frontend/src/routes/interviews/templates.tsx frontend/src/routes/interviews/template-detail.tsx frontend/src/components/layout/Sidebar.tsx frontend/src/components/layout/TopBar.tsx frontend/tests/e2e/interview-voice-mvp.spec.ts docs/superpowers/plans/2026-06-08-interviews-hub.md
git commit -m "feat: add interviews hub management"
```

- [ ] **Step 5: Prepare a short verification summary for handoff**

```text
Include: backend test command, frontend build command, Playwright command, and any residual risk such as backend delete constraints or route precedence assumptions.
```
