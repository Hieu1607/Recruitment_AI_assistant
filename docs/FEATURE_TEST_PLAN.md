# Feature Inventory And Test Plan

Updated: 2026-05-22

## Scope

This inventory is derived from the current codebase entry points:

- Backend router: `backend/src/api/v1/api.py`
- Frontend route map: `frontend/src/routes/index.ts`
- Existing project notes: `QUICKSTART.md`, `docs/JOB_ARCHITECTURE_TEST_PLAN.md`

## Feature Inventory

### Platform and access

- Landing page and app shell
- Local login flow
- Google OAuth login and callback
- Session token bootstrap on frontend
- Settings screen

### Job workspace

- Create, list, edit, archive/delete jobs
- Select active job workspace on frontend
- Job-scoped isolation for resumes, JD, scoring, and chat
- Public application link settings per job
- Public application link rotation

### Public candidate apply flow

- Anonymous public job lookup by token
- Public resume submission with full name and email
- Public link enable/disable handling
- Queueing submitted resumes for background parsing

### Resume and candidate pipeline

- Recruiter resume upload by selected job
- Resume storage URI creation and retention fields
- Background parsing task enqueue
- CV text extraction and OCR fallback integration hooks
- Candidate profile creation from parsed CV
- Fallback candidate profile creation on parse failure
- Candidate list and candidate detail views

### Job description workflow

- Create/replace active JD for a selected job
- Patch JD content and active status
- Frontend authoring/edit flow

### Scoring

- Job-scoped candidate scoring request
- Threshold, batch size, section weights
- Scoring results screen

### AI-assisted workflows

- Chat over selected-job candidates
- Shortlist sessions and collections
- Outreach draft/send-state management
- Interview question generation, editing, ordering, and deletion
- Voice screening interview templates, invitations, public interview session flow, and HR report review

### Dashboard and reporting surfaces

- Dashboard metrics based on selected job
- Candidate list filtering/sorting/pagination
- Public application link card and QR/share UI

## Test Strategy

### Automated checks

- Backend unit/integration tests via `pytest backend/tests -q`
- Frontend typecheck + production build via `npm run build`
- Frontend browser E2E via `npm run test:e2e`

### Feature groups mapped to evidence

| Area | Core flows | Current evidence |
| --- | --- | --- |
| Auth | Local register/login/profile update, Google login redirect, callback validation, token redirect | `backend/tests/test_auth_account_endpoints.py`, `backend/tests/test_auth_endpoints.py`, `backend/tests/test_google_oauth_service.py` |
| Job workspace | create/list/get/update jobs, application link settings, ownership enforcement | `backend/tests/test_job_application_link_endpoints.py` |
| Public apply | token lookup, disabled link handling, form validation, queued submission | `backend/tests/test_public_job_endpoints.py`, `backend/tests/test_public_job_service.py` |
| Queue + worker integration | Celery enqueue/consume, worker import path, object-storage-backed resume processing | Docker smoke on 2026-05-21 using `docker compose` stack; verified `process_resume` reached `status=processed` and created candidate |
| Resume parsing fallback | submitted full name/email fallback, failure profile creation | `backend/tests/test_resume_service_public_fallback.py` |
| Scoring | JD ownership enforcement, zero-weight validation, explicit weight forwarding | `backend/tests/test_score_endpoint.py` |
| Job-scoped chat | Owner-only access, deterministic total-count responses, and candidate scoping for job chat | `backend/tests/test_job_chat_endpoint.py`, `backend/tests/test_ai_agent_nodes.py`, Docker smoke confirmed no cross-job candidate leakage and correct total-count answer |
| Shortlist | Session CRUD, collection duplicate handling, item duplicate handling | `backend/tests/test_shortlist_endpoints.py` |
| Outreach | Create/list/filter/update/delete, sent timestamp behavior | `backend/tests/test_outreach_endpoints.py`, browser smoke on 2026-05-21 verified draft creation and mark-as-sent flow |
| Interview questions | CRUD plus LLM-backed generation success/failure handling | `backend/tests/test_interview_questions_endpoints.py`, browser smoke on 2026-05-21 verified generate flow and detail page render |
| Voice screening interviews | recruiter template CRUD, invitation send, public interview start/event/complete, report fetch | `backend/tests/test_interview_template_endpoints.py`, `backend/tests/test_interview_public_endpoints.py`, `backend/tests/test_voice_provider.py`, `backend/tests/test_interview_report_service.py`, `frontend/tests/e2e/interview-voice-mvp.spec.ts` |
| PDF extraction dependency | PyMuPDF import/extraction smoke | `backend/tests/test_pymupdf.py` |
| Frontend route graph | Type-safe compile of routed screens and API clients | `npm run build` |
| Browser UI smoke | public apply form, shortlist collection/detail, outreach compose/send, interview generation/detail, AI chat, scoring results | Repeatable Playwright suite in `frontend/tests/e2e/` (`public-apply.spec.ts`, `workspace-smoke.spec.ts`); passed on 2026-05-22 against `http://127.0.0.1:5173` + `http://127.0.0.1:8000/api/v1` |

### Manual / external-service smoke checks

These are valid features in the codebase but were not fully automated in the current suite:

- HF OCR round-trip scripts: `test.py`, `test_pipeline.py`, `test_pipeline_tesseract.py`
- Browser-level UI permutations not covered by the two Playwright smokes: settings edits, full job CRUD variants, candidate detail mutations, destructive delete/archive flows, and public-link rotation UX
- Natural-language answer quality for chat/counting questions remains LLM-prompt dependent even when candidate scope is correct

These require one or more of:

- Running Docker services from `docker-compose.yml`
- Valid `GROQ_API_KEY`
- Reachable HF OCR endpoints
- Browser/manual execution environment

## Execution Record

### Completed in current workspace

- `pytest backend/tests -q`
  - Result: `62 passed`
- `npm run build` in `frontend/`
  - Result: passed
- `npm run test:e2e` in `frontend/`
  - Result: `2 passed`
  - Coverage: repeatable browser smoke for public apply plus authenticated recruiter workspace flow
- `pytest backend/tests/test_interview_report_service.py -q`
  - Result: `8 passed`
  - Coverage: completed interview report generation plus recruiter fetch endpoint
- `npx playwright test tests/e2e/interview-voice-mvp.spec.ts --reporter=line` in `frontend/`
  - Result: `1 passed`
  - Coverage: recruiter template management, invitation send, public interview completion, and report review via mocked API contracts
- `docker compose up -d db redis minio backend worker` plus live API smoke
  - Result: public apply upload was queued, worker consumed from the correct queue, `process_resume` completed, and candidate creation became visible in `/jobs/{job_id}/candidates`
- Playwright browser smoke on local frontend
  - Result: verified first-time job setup, candidate list, shortlist collection/detail, outreach draft + mark sent, interview question generation/detail, job chat count response, scoring setup/results, and public apply success state
  - Note: interview generation smoke now waits for the live `/interview-questions/generate` response explicitly before asserting redirect, which avoids false negatives from slow LLM turnaround

### Deferred by external dependency

- OCR + LLM pipeline scripts were not executed as part of this pass.
- Full scripted browser coverage for every route permutation (settings edits, job CRUD variants, candidate detail tab mutations, destructive flows) is still not present.
- The browser E2E still depends on live local services plus a working external LLM path for interview-question generation; it is repeatable, but not hermetic.
- The dedicated voice-screening Playwright spec is hermetic by design and mocks API contracts so UI regressions can be caught even when Docker/local infra is unavailable.

## Recommended next additions

- Add browser E2E for settings edits, job CRUD variants, candidate detail mutations, and destructive flows.
- Add an automated integration profile that boots DB + Redis + MinIO and exercises Celery `process_resume`.
- Add deterministic tests around chat answer formatting/counting so candidate scope correctness and LLM wording regressions are separated.
- Consider a deterministic fallback or test-only provider for interview-question generation so browser E2E no longer depends on live LLM latency.
