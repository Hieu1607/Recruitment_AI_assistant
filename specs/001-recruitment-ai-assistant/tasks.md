---

description: "Task list for implementing Recruitment AI Assistant Website"
---

# Tasks: Recruitment AI Assistant Website

**Input**: Design documents from `/specs/001-recruitment-ai-assistant/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/recruitment-api.yaml, quickstart.md

**Tests**: Tests are not explicitly requested in the specification, so this task list focuses on implementation tasks.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Every task includes a concrete file path

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Initialize repository layout, dependency manifests, and baseline configuration.

- [X] T001 Create backend project scaffold in backend/src/main.py
- [X] T002 Create backend dependency manifest in backend/requirements.txt
- [X] T003 [P] Create frontend project scaffold in frontend/src/main.tsx
- [X] T004 [P] Create frontend dependency manifest in frontend/package.json
- [X] T005 [P] Create backend environment template with MinIO and LLM provider/model variables (`LLM_PROVIDER`, `GROQ_MODEL`, `OLLAMA_MODEL`) in backend/.env.example
- [X] T006 [P] Create frontend environment template in frontend/.env.example
- [X] T007 [P] Configure backend lint and format settings in backend/pyproject.toml
- [X] T008 [P] Configure frontend lint and format settings in frontend/eslint.config.js
- [X] T009 Define initial Docker Compose-first dev workflow commands (up, down, logs, migrate) in scripts/maintenance/dev.ps1

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build core architecture and shared modules required before any user story delivery.

**CRITICAL**: No user story implementation should begin until this phase is complete.

- [X] T010 Create relational schema migration baseline in backend/migrations/versions/0001_initial_schema.py
- [X] T011 [P] Implement SQLAlchemy base and session management in backend/src/repositories/db.py
- [X] T012 [P] Implement RBAC domain models (UserAccount, RoleAssignment) in backend/src/models/auth.py
- [X] T013 [P] Implement resume and candidate core models in backend/src/models/candidate.py
- [X] T014 [P] Implement shortlist and outreach core models in backend/src/models/engagement.py
- [X] T015 [P] Implement matching and query session core models in backend/src/models/matching.py
- [X] T016 Implement MinIO storage client wrapper in backend/src/services/storage/minio_client.py
- [X] T017 [P] Implement LLM provider abstraction supporting Groq (default) and Ollama based on env configuration in backend/src/services/llm/llm_client.py
- [X] T018 [P] Implement masked logging and audit utility in backend/src/services/observability/audit_logger.py
- [X] T019 Implement FastAPI app bootstrap with middleware and router registration in backend/src/api/app.py
- [X] T020 [P] Implement auth and role guard dependencies in backend/src/api/dependencies/auth.py
- [X] T021 [P] Implement shared error handling contract in backend/src/api/errors.py
- [X] T022 Implement background worker bootstrap in backend/worker/main.py
- [X] T023 [P] Implement retention scheduler skeleton in backend/worker/jobs/retention_job.py
- [X] T024 Implement LangGraph orchestration skeleton and state model in backend/src/orchestration/recruitment_graph.py

**Checkpoint**: Foundation ready for independent user story implementation.

---

## Phase 3: User Story 1 - Parse CV and Build Candidate Profile (Priority: P1) MVP

**Goal**: Recruiter uploads resume PDFs, system extracts and normalizes profiles, recruiter reviews/edits, and saves searchable candidate records.

**Independent Test**: Upload a resume batch, view extraction output with source traceability, edit fields, and persist candidate profiles.

- [X] T025 [P] [US1] Implement resume upload endpoint from contract in backend/src/api/routes/resumes.py
- [X] T026 [P] [US1] Implement PyMuPDF extraction service with OCR fallback hooks in backend/src/services/parsing/resume_extractor.py
- [X] T027 [P] [US1] Implement profile normalization mapper for required fields in backend/src/services/parsing/profile_normalizer.py
- [X] T028 [US1] Implement extraction trace persistence service in backend/src/services/parsing/extraction_trace_service.py
- [X] T029 [US1] Implement async ingestion job orchestration (upload -> extract -> normalize) in backend/worker/jobs/resume_ingestion_job.py
- [X] T030 [US1] Implement candidate review/update endpoint in backend/src/api/routes/candidates.py
- [X] T031 [US1] Implement candidate repository queries for searchable retrieval in backend/src/repositories/candidate_repository.py
- [X] T032 [US1] Implement upload and candidate review frontend screens in frontend/src/modules/candidates/pages/CandidateIngestionPage.tsx
- [X] T033 [US1] Implement candidate detail side panel component for source trace display in frontend/src/modules/candidates/components/CandidateDetailPanel.tsx

**Checkpoint**: User Story 1 is independently usable and delivers MVP value.

---

## Phase 4: User Story 2 - Match Candidates Against Job Description (Priority: P1)

**Goal**: Recruiter submits one JD and one shared scoring prompt for many CVs, and receives per-candidate weighted component score lists plus total scores.

**Independent Test**: Submit one JD with multiple selected candidates and one prompt template, then verify each candidate has component score list + total score, pass threshold filtering, and sorted ranking.

- [X] T034 [P] [US2] Implement match run creation endpoint in backend/src/api/routes/matching.py
- [X] T035 [P] [US2] Implement batch scoring request builder that packages one JD, many CV profiles, and one shared prompt in backend/src/services/matching/batch_prompt_builder.py
- [X] T036 [P] [US2] Implement LLM batch scoring executor for Groq/Ollama in backend/src/services/matching/batch_llm_scorer.py
- [X] T037 [US2] Implement component score list and total score parser, plus MatchResult persistence from LLM list output in backend/src/services/matching/score_list_parser.py
- [X] T038 [US2] Implement match result repository with threshold filtering and sorting in backend/src/repositories/match_repository.py
- [X] T039 [US2] Implement matching configuration and result UI in frontend/src/modules/matching/pages/MatchingPage.tsx
- [X] T040 [US2] Implement score breakdown and rationale component in frontend/src/modules/matching/components/ScoreBreakdownTable.tsx

**Checkpoint**: User Story 2 works independently on top of foundational + candidate data.

---

## Phase 5: User Story 3 - Ask Questions and Filter Candidates in Natural Language (Priority: P1)

**Goal**: Recruiter asks natural-language queries, system routes SQL/LLM/hybrid tools, returns counts plus matched candidates, and supports candidate inspection.

**Independent Test**: Ask count and multi-condition queries, validate routing behavior and matched set correctness, and open candidate details from results.

- [X] T041 [P] [US3] Implement query session and ask endpoint in backend/src/api/routes/query.py
- [X] T042 [P] [US3] Implement SQL search tool for deterministic fields in backend/src/agents/tools/sql_search_tool.py
- [X] T043 [P] [US3] Implement LLM semantic search tool for extracted sections in backend/src/agents/tools/llm_semantic_tool.py
- [X] T044 [US3] Implement router/orchestrator node choosing tool sequence in backend/src/agents/router/query_router.py
- [X] T045 [US3] Implement query verifier ensuring count consistency and fallback behavior in backend/src/agents/verifier/query_verifier.py
- [X] T046 [US3] Implement query turn persistence and audit trail writer in backend/src/services/query/query_history_service.py
- [X] T047 [US3] Implement chat and filter interface with result list in frontend/src/modules/query/pages/QueryWorkspacePage.tsx
- [X] T048 [US3] Implement side-by-side chat and candidate widget layout in frontend/src/modules/query/components/QueryAndCandidateLayout.tsx

**Checkpoint**: User Story 3 provides independently testable conversational filtering.

---

## Phase 6: User Story 4 - Follow-up Actions from Filtered Results (Priority: P2)

**Goal**: Recruiter saves shortlists, drafts and sends approved outreach emails, and generates interview questions per candidate + JD pair.

**Independent Test**: Save shortlist from query results, approve/send outreach email, and generate interview questions for a chosen candidate/JD.

- [X] T049 [P] [US4] Implement shortlist create/list endpoints in backend/src/api/routes/shortlists.py
- [X] T050 [P] [US4] Implement shortlist persistence service in backend/src/services/shortlist/shortlist_service.py
- [X] T051 [P] [US4] Implement outreach draft endpoint (AI/template) in backend/src/api/routes/outreach.py
- [X] T052 [US4] Implement outreach approval-and-send guard logic in backend/src/services/outreach/outreach_service.py
- [X] T053 [P] [US4] Implement SMTP adapter and delivery status logging in backend/src/services/outreach/email_sender.py
- [X] T054 [P] [US4] Implement interview question generation endpoint in backend/src/api/routes/interview_questions.py
- [X] T055 [US4] Implement interview question generation service in backend/src/services/interview/interview_question_service.py
- [X] T056 [US4] Implement shortlist management and outreach UI in frontend/src/modules/engagement/pages/EngagementPage.tsx
- [X] T057 [US4] Implement interview question panel UI in frontend/src/modules/engagement/components/InterviewQuestionPanel.tsx

**Checkpoint**: User Story 4 is independently testable for post-filter workflows.

---

## Phase 7: Polish and Cross-Cutting Concerns

**Purpose**: Final hardening, performance tuning, and operational readiness across all stories.

- [X] T058 [P] Add API contract conformance and example payload documentation in specs/001-recruitment-ai-assistant/contracts/recruitment-api.yaml
- [X] T059 [P] Add architecture and operating runbook documentation in README.md
- [X] T060 Implement end-to-end quickstart validation script in scripts/maintenance/validate_quickstart.ps1
- [X] T061 Implement performance instrumentation and latency dashboards wiring in backend/src/services/observability/metrics.py
- [X] T062 Implement privacy and log masking compliance review checklist in specs/001-recruitment-ai-assistant/checklists/requirements.md
- [X] T063 Implement production readiness hardening for RBAC and secret handling in backend/src/api/security/hardening.py

---

## Dependencies and Execution Order

### Phase Dependencies

- Phase 1 (Setup): starts immediately.
- Phase 2 (Foundational): depends on Phase 1 completion and blocks all user stories.
- Phases 3-6 (User Stories): depend on Phase 2 completion.
- Phase 7 (Polish): depends on completion of selected user stories.

### User Story Dependencies

- US1: starts after Foundational and serves as MVP baseline.
- US2: starts after Foundational and depends on candidate data from US1 for meaningful execution.
- US3: starts after Foundational and depends on candidate persistence from US1; can proceed in parallel with US2.
- US4: depends on outputs from US3 (filtered candidate sets) and uses candidate/JD data from US1/US2.

### Dependency Graph

- Setup -> Foundational -> US1 -> (US2 and US3) -> US4 -> Polish

---

## Parallel Execution Examples

### User Story 1

- Run T025 and T026 in parallel (API route and extraction service in different files).
- Run T027 in parallel with T032 (normalizer backend and UI screen front-end).

### User Story 2

- Run T034, T035, and T036 in parallel (route, scoring engine, JD preprocessing).
- Run T039 and T040 in parallel after backend result model stabilizes.

### User Story 3

- Run T042 and T043 in parallel (SQL and LLM tools).
- Run T047 and T048 in parallel (workspace page and layout component).

### User Story 4

- Run T049, T051, and T054 in parallel (independent endpoints).
- Run T056 and T057 in parallel (separate engagement UI components).

---

## Implementation Strategy

### MVP First

1. Complete Phase 1 and Phase 2.
2. Complete Phase 3 (US1) end to end.
3. Validate ingestion, review/edit, and profile persistence before expanding scope.

### Incremental Delivery

1. Deliver US1 as MVP baseline.
2. Add US2 scoring and ranking workflows.
3. Add US3 conversational filtering and candidate inspection.
4. Add US4 shortlist/outreach/interview actions.
5. Finalize with Phase 7 hardening and operational polish.

### Suggested Team Parallelization

1. Backend Platform Track: T010-T024, then T034-T046, then T049-T055.
2. Frontend Experience Track: T032-T033, then T039-T040, then T047-T048, then T056-T057.
3. Reliability and Ops Track: T058-T063 after core flows stabilize.

---

## Format Validation

- All tasks follow `- [ ] T### [P?] [US?] Description with file path`.
- Setup, Foundational, and Polish phases have no story labels.
- User story phase tasks include required story labels `[US1]`, `[US2]`, `[US3]`, `[US4]`.
- Parallel markers are used only where tasks are independently executable.
