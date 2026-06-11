# Voice Screening Interview Design

Date: 2026-05-22
Status: Proposed
Scope: MVP design for a public-link, structured AI voice screening interview flow

## Goal

Add a job-scoped AI interview feature that lets recruiters send a public interview link to an applicant after they have applied. The candidate completes a structured voice screening interview in the browser. The system stores transcript evidence and generates an HR-facing report for review.

This feature is for initial screening only. It is not a final hiring decision system and should not auto-reject candidates in the MVP.

## Product Decisions

- Interview mode: hybrid roadmap, but MVP is browser-based voice interview
- Link model: public link sent by recruiter after the candidate has applied
- Question model: question set is attached to a job, not to an individual candidate
- Interview style: structured screening only
- Candidate experience: real-time voice interview in a public browser page
- Recruiter output: transcript plus a markdown-style summary report per candidate

## Non-Goals

- Full PSTN or SIP phone interview support in the MVP
- Free-form AI-led deep probing beyond the approved script
- Automatic pass/fail hiring decisions
- Full anti-cheating or identity-verification tooling
- Broad multilingual optimization in the first release

## Existing System Context

The current product already includes:

- Public job application flow
- Candidate profiles created from submitted resumes
- Job-scoped workspace isolation
- Interview question generation and editing flows
- LLM-backed service abstractions and worker infrastructure

Relevant existing areas:

- `backend/src/api/v1/endpoints/interview_questions.py`
- `backend/src/prompts/build_prompts.py`
- `frontend/src/routes/interview-questions/`
- `backend/src/services/llm_service.py`
- `backend/worker/tasks.py`

The main design adjustment is changing interview questions from candidate-scoped generation to job-scoped interview templates, then building invitation, session, transcript, and reporting flows on top.

## User Flow

### Recruiter flow

1. Recruiter creates or edits an interview template for a job.
2. Recruiter reviews applicants in that job workspace.
3. Recruiter selects a candidate and sends an interview invitation.
4. System creates a public tokenized interview link tied to that candidate and job.
5. Recruiter later reviews the completed interview report in the app.

### Candidate flow

1. Candidate receives the public interview link.
2. Candidate opens the link, sees AI disclosure and microphone permission prompt.
3. Candidate starts the voice interview in the browser.
4. Agent asks each scripted question in order.
5. Candidate answers by voice.
6. System completes the interview and shows a completion state.

### System flow

1. Backend resolves invitation token and validates status, expiry, and attempt limits.
2. Backend starts an interview session from the job template.
3. Voice pipeline handles TTS output and STT input.
4. Orchestrator advances through the script question by question.
5. Transcript is stored by turn and by question.
6. Worker generates a structured summary and markdown report for HR.

## Core Requirements

### Functional requirements

- Recruiters can manage a reusable interview template per job.
- Recruiters can send a public interview invitation to a candidate who already applied.
- Each invitation is bound to one candidate and one job.
- Candidate can complete the interview from a public browser page without logging in.
- Interview uses a fixed ordered script from the job template.
- System stores transcript evidence for each question.
- System generates a recruiter-facing report after completion.
- Recruiter can view invitation status and completed report from the job workspace.

### Guardrail requirements

- Candidate must be told they are interacting with AI.
- Agent must stay within the configured script.
- Agent must not ask prohibited or sensitive hiring questions outside approved scope.
- Agent must not communicate hiring decisions to candidates.
- Public link must expire and support attempt controls.
- Report must be positioned as decision support for recruiter review.

### Operational requirements

- MVP should prefer free-tier or low-cost STT/TTS where feasible.
- Architecture must support later migration to higher-quality realtime voice providers.
- Architecture must support later addition of phone-call entrypoints without rewriting core interview domain logic.
- System should handle roughly 50 interview sessions per day without architectural change.

## Architecture

The feature should be split into five bounded areas.

### 1. Interview template domain

This domain defines the screening script at the job level.

Responsibilities:

- Store reusable job-scoped interview templates
- Version and update the ordered question list
- Store intro and closing script text
- Store reporting rubric or summary guidance

This replaces the assumption that questions belong to a specific candidate.

### 2. Invitation domain

This domain controls recruiter-issued access to the public interview.

Responsibilities:

- Create invitation records tied to candidate, job, and template
- Generate and validate public tokens
- Enforce expiry, status, and attempt limits
- Track sent, opened, started, completed, and cancelled states

This domain is the control point that allows a public link without exposing internal workspace data.

### 3. Realtime session orchestrator

This domain controls the actual interview execution.

Responsibilities:

- Create a session from an invitation
- Load the template and current question state
- Emit the scripted prompt for the next question
- Accept streaming or chunked transcript events
- Detect completion, skip, retry, and timeout cases
- Finalize the session

The orchestrator should be a deterministic server-side state machine. The model is used as a voice and summarization layer, not as the authority for interview flow control.

### 4. Transcript and evidence pipeline

This domain captures raw evidence and question-aligned answers.

Responsibilities:

- Store turn-level transcript logs
- Store answer text aligned to each question
- Track timing and confidence metadata
- Preserve enough evidence for review and audit

### 5. Reporting domain

This domain turns interview evidence into recruiter-facing output.

Responsibilities:

- Generate structured JSON summary
- Generate markdown summary for human review
- Highlight notable strengths, concerns, and missing evidence
- Suggest follow-up topics for recruiter review

The report should remain descriptive and should not make an autonomous pass/fail recommendation in the MVP.

## Data Model

### InterviewTemplate

Purpose: reusable interview definition per job

Suggested fields:

- `id`
- `job_id`
- `name`
- `language_code`
- `status`
- `intro_script`
- `closing_script`
- `question_payload`
- `report_rubric`
- `version`
- `created_at`
- `updated_at`

### InterviewInvitation

Purpose: recruiter-issued public invitation per candidate

Suggested fields:

- `id`
- `job_id`
- `candidate_profile_id`
- `interview_template_id`
- `public_token`
- `status`
- `expires_at`
- `max_attempts`
- `attempt_count`
- `sent_by_user_id`
- `sent_at`
- `opened_at`
- `completed_at`
- `cancelled_at`

Suggested statuses:

- `pending`
- `opened`
- `in_progress`
- `completed`
- `expired`
- `cancelled`

### InterviewSession

Purpose: one concrete runtime execution of an interview

Suggested fields:

- `id`
- `invitation_id`
- `provider`
- `provider_session_id`
- `status`
- `started_at`
- `ended_at`
- `device_info`
- `browser_info`
- `connection_metrics`

### InterviewResponseItem

Purpose: per-question answer material for review

Suggested fields:

- `id`
- `session_id`
- `question_id`
- `question_text`
- `question_order`
- `answer_transcript`
- `answer_duration_sec`
- `skipped`
- `needs_review`
- `low_confidence_flags`

### InterviewTranscriptTurn

Purpose: raw ordered transcript evidence

Suggested fields:

- `id`
- `session_id`
- `speaker`
- `sequence`
- `question_id`
- `started_at_ms`
- `ended_at_ms`
- `text`

### InterviewReport

Purpose: recruiter-facing post-interview artifact

Suggested fields:

- `id`
- `session_id`
- `candidate_profile_id`
- `job_id`
- `summary_markdown`
- `structured_summary_json`
- `generated_at`
- `generator_version`

## API Design

### Recruiter APIs

- `POST /jobs/{job_id}/interview-templates`
- `GET /jobs/{job_id}/interview-templates`
- `GET /interview-templates/{template_id}`
- `PATCH /interview-templates/{template_id}`
- `POST /interview-invitations`
- `GET /jobs/{job_id}/interview-invitations`
- `GET /candidates/{candidate_id}/interviews`
- `GET /interview-reports/{session_id}`

### Public candidate APIs

- `GET /public/interview/{token}`
- `POST /public/interview/{token}/start`
- `POST /public/interview/{token}/events`
- `POST /public/interview/{token}/complete`

### Internal orchestration APIs

- `POST /internal/interview-sessions/{session_id}/next-question`
- `POST /internal/interview-sessions/{session_id}/finalize-transcript`
- `POST /internal/interview-sessions/{session_id}/generate-report`

API design principles:

- Public endpoints are token-based and do not require candidate login.
- Recruiter endpoints stay fully job-scoped and authenticated.
- Report generation is asynchronous.
- Public event ingestion is isolated from recruiter CRUD APIs.

## Public Link Behavior

The public link is required for candidate access, but it must remain constrained.

Required controls:

- random unguessable token
- expiry window
- attempt limit
- revocation by recruiter
- invitation status checks before start
- no access to recruiter workspace data

Recommended phase-1 behavior:

- one active session per invitation
- one completion per invitation
- recruiter can reissue a fresh invitation if needed

## Voice Runtime Design

### MVP runtime

Phase 1 uses browser audio with backend orchestration.

Candidate browser responsibilities:

- microphone capture
- public interview UI
- websocket or realtime transport to the backend/provider bridge
- local UI for state, connection, and completion

Backend responsibilities:

- validate invitation
- create and resume session state
- send the next question prompt
- receive and store transcript events
- determine when to advance to the next question
- finalize the session and dispatch report generation

### Structured flow behavior

Because the interview style is structured, the agent should only:

- read intro
- ask current scripted question
- repeat or clarify the same question when needed
- move to the next scripted question
- read closing script

The agent should not:

- invent new screening questions
- improvise behavioral probes
- discuss compensation or hiring outcome
- request unrelated personal details

## Provider Strategy

### MVP provider preference

Low-cost or free-tier biased stack:

- streaming STT: Deepgram
- TTS: Deepgram Aura or a similar low-cost provider
- LLM summarization and reporting: existing backend abstraction

Why:

- lower cost entry point
- good fit for realtime browser screening MVP
- less lock-in if transcript and orchestration stay server-owned

### Scale-up path

When interview quality or product scale increases:

- move to a higher-quality realtime voice stack such as OpenAI Realtime where justified
- add a telephony adapter later for phone-call entry
- keep the same template, invitation, session, transcript, and report domains

This separation allows transport and voice-provider changes without redesigning the product workflow.

## Reporting Design

The report should exist in both structured and human-readable forms.

### Structured summary JSON

Use cases:

- in-app rendering
- filtering and analytics
- future recruiter dashboards
- quality monitoring

Suggested content:

- completion status
- total duration
- question-by-question summary
- skipped questions
- low-confidence segments
- notable strengths
- notable concerns
- suggested recruiter follow-up topics

### Markdown report

Use cases:

- direct HR review
- export or internal sharing
- audit-friendly readable artifact

Suggested format:

```md
# Interview Report

## Candidate
- Name:
- Job:
- Invitation sent:
- Interview completed:
- Duration:

## Overall Summary

## Per-Question Review
### Q1
- Candidate answer summary:
- Evidence:
- Initial review note:

## Signals
- Strengths:
- Concerns:
- Missing evidence:
- Follow-up topics:

## Transcript
```

The report should be descriptive, concise, and review-oriented. It should not declare that a candidate should be accepted or rejected.

## Compliance And Guardrails

This feature operates in a hiring context, so compliance and reviewability matter from day one.

Guardrails to enforce:

- disclose clearly that the candidate is interacting with AI
- use only approved recruiter-managed scripts
- avoid prohibited or sensitive hiring questions
- avoid automated hiring decisions in the candidate-facing flow
- keep recruiter review central to evaluation
- preserve transcript evidence for later inspection

The product team should review interview content against hiring-law guidance before broad release.

Reference material consulted:

- EEOC guidance on inappropriate hiring questions
- EEOC overview on AI and employment discrimination

## Cost And Scale Assumptions

The target of approximately 50 users per day is still compatible with a lightweight architecture if:

- transcript capture is streamed and persisted incrementally
- report generation runs asynchronously
- provider integration is abstracted
- retries and timeouts are handled explicitly

This scale does not require a major infrastructure rewrite beyond the current backend, database, object storage, and worker pattern already present in the repository.

## Test Strategy

### Backend tests

- interview template CRUD
- invitation token lifecycle
- expiry and attempt limit enforcement
- session lifecycle
- report generation from transcript fixtures

### Provider adapter tests

- STT event normalization
- TTS request formatting
- malformed provider event handling
- retry and failure paths

### Frontend and E2E tests

- recruiter creates or edits template
- recruiter sends invitation
- candidate opens public link
- simulated voice session completes
- recruiter views generated report

### Prompt and quality tests

- fixed transcript fixtures for deterministic report assertions
- forbidden-output checks for candidate-facing agent prompts
- checks that summaries remain descriptive rather than decisioning

## Delivery Recommendation

Implement this feature in three focused stages.

### Stage 1: job-scoped interview domain

- replace candidate-scoped question-set assumption with job-scoped interview template support
- add invitation and report entities
- expose recruiter-side CRUD and list views

### Stage 2: realtime browser interview MVP

- build public interview page
- add session orchestration
- integrate low-cost STT/TTS providers
- store transcript and per-question answer artifacts
- generate recruiter report

### Stage 3: hardening and scale-readiness

- provider abstraction cleanup
- retries and observability
- stronger guardrail tests
- E2E coverage
- phone-call adapter design preparation

## Open Decisions Deferred

These items are intentionally deferred and should not block MVP design approval:

- exact telephony provider for future phone-call support
- whether to store raw audio files or only transcript and metadata in the first release
- whether recruiter scoring rubrics should be editable in the initial UI
- whether multilingual templates should be supported in the first or second iteration

## Implementation Note

The MVP browser client should keep the domain and public-session APIs compatible with future realtime microphone streaming, but the regression suite should stay deterministic. In practice this means:

- browser E2E can use mocked or simulated transcript submission instead of live microphone capture
- transcript ingestion should remain provider-agnostic
- provider-specific realtime audio transport can be added later without changing invitation, session, or report domain objects

## Recommendation

Proceed with a browser-first hybrid interview design that keeps the interview script job-scoped, invitations candidate-scoped, and reporting recruiter-facing. This is the most practical MVP because it minimizes cost and implementation complexity while preserving a clean upgrade path to higher-quality realtime voice infrastructure and future phone support.
