# Recruiter AI — Sprint Plan for Claude Code

> **Strategy:** Each sprint is scoped to fit within a single Claude Code context window.
> A **✅ Checkpoint** ends every sprint — commit, verify, and start a **fresh Claude Code session** before the next sprint.
> Paste only the relevant sprint block into each new session.

---

## How to Use This Plan

1. Open Claude Code in a **new session**.
2. Paste the **Project Context block** + the **current sprint block** only.
3. Complete all tasks in the sprint.
4. Run the **Checkpoint Checklist** at the end.
5. Commit everything, close the session, and move on.

---

## 📌 Project Context Block

> Paste this at the top of **every** Claude Code session.

```
Stack: React + Vite (port 5173) · FastAPI backend (port 8000) · TypeScript
Runtime: Docker Compose — all services managed via docker-compose.yml
Start all services: docker compose up --build
Frontend hot-reload runs inside the frontend container (Vite dev server)
API base: http://localhost:8000/api/v1
Auth: none (JWT planned — use placeholder header)
All IDs: UUIDs (strings)
All datetimes: UTC ISO 8601
Pagination: ?limit=50&offset=0 on all list endpoints
Error shape: { "detail": "..." }
Scores array uses camelCase (candidateId, totalScore, etc.) — rest of API uses snake_case
```

---

## Sprint 1 — Project Scaffolding & API Client

**Goal:** Lay the foundation. No UI yet — just structure, types, and a working API layer.

### Tasks

- [ ] Verify `docker-compose.yml` defines these services (add/adjust if missing):
  - `frontend` — Vite dev server, port `5173`, volume-mounted `./frontend` for hot-reload
  - `backend` — FastAPI, port `8000`
  - `db` — Postgres (or whatever the backend expects)
  - Ensure `BACKEND_CORS_ORIGINS` env var includes `http://localhost:5173`
- [ ] Verify `frontend/Dockerfile` runs `vite --host 0.0.0.0` so the dev server is reachable from the host
- [ ] Run `docker compose up --build` — confirm all containers start cleanly
  - Frontend accessible at `http://localhost:5173`
  - Backend health check: `GET http://localhost:8000/` returns welcome message
  - Swagger UI accessible at `http://localhost:8000/docs`
- [ ] Set up folder structure inside `frontend/src/`:
  ```
  src/
    api/          ← axios instance + per-resource clients
    types/        ← TypeScript interfaces
    store/        ← global state
    components/   ← shared UI
    pages/        ← route-level pages
    hooks/        ← custom hooks
  ```
- [ ] Create `src/api/client.ts` — axios instance with `baseURL: http://localhost:8000/api/v1`, default headers, error interceptor
- [ ] Create `src/types/index.ts` — paste all interfaces from the **TypeScript Interfaces** reference below
- [ ] Create per-resource API modules (typed function stubs, no logic yet):
  - `src/api/resumes.ts`
  - `src/api/jobDescriptions.ts`
  - `src/api/scoring.ts`
  - `src/api/chat.ts`
  - `src/api/shortlist.ts`
  - `src/api/outreach.ts`
  - `src/api/interviewQuestions.ts`
- [ ] Set up React Router with placeholder page components for each section
- [ ] Confirm hot-reload works: edit any `.tsx` file → browser updates without container restart

### TypeScript Interfaces (paste into `src/types/index.ts`)

```typescript
// --- Pagination ---
export interface PaginatedResponse<T> {
  total: number;
  items: T[];
}

// --- Resumes ---
export interface ResumeResponse {
  id: string;
  original_file_name: string;
  storage_uri: string;
  upload_status: 'uploaded' | 'processing' | 'processed' | 'failed';
  duplicate_group_key: string | null;
  uploaded_by_user_id: string;
  uploaded_at: string | null;
  processed_at: string | null;
  retention_expires_at: string | null;
}

export interface BatchParseResponse {
  total_files: number;
  processed_files: number;
  failed_files: number;
  items: {
    file_name: string;
    resume_document_id: string;
    candidate_profile_id: string;
    status: string;
  }[];
}

// --- Job Descriptions ---
export interface JobDescriptionResponse {
  id: string;
  title: string | null;
  jd_text: string;
  created_by_user_id: string;
  created_at: string;
  is_active: boolean;
}

// --- Scoring ---
export interface ComponentScore {
  criterionKey: string;
  weight: number;
  score: number;
  weightedScore: number;
  evidenceSummary: string;
}

export interface CandidateScore {
  candidateId: string;
  totalScore: number;
  passedThreshold: boolean;
  rationale: string;
  componentScores: ComponentScore[];
}

export interface ScoreResponse {
  match_run_id: string;
  job_description_id: string;
  total_candidates: number;
  total_passed_candidates: number;
  batches: number;
  scores: CandidateScore[];
}

// --- Chat ---
export interface ChatResponse {
  session_id: string;
  answer: string;
  candidates_in_scope: number;
}

export interface ChatMessage {
  role: 'human' | 'ai';
  content: string;
}

export interface ChatHistoryResponse {
  session_id: string;
  messages: ChatMessage[];
}

// --- Shortlist ---
export interface SessionResponse {
  id: string;
  user_id: string;
  session_title: string | null;
  turn_count: number;
  created_at: string;
  updated_at: string;
}

export interface TurnResponse {
  id: string;
  query_session_id: string;
  user_question: string;
  answer_text: string;
  matched_candidate_ids: string[] | null;
  matched_count: number | null;
  tool_trace_masked: Record<string, unknown> | null;
  created_at: string;
}

export interface CollectionResponse {
  id: string;
  name: string;
  created_by_user_id: string;
  source_query_turn_id: string | null;
  item_count: number;
  created_at: string;
}

export interface ShortlistItemResponse {
  id: string;
  shortlist_collection_id: string;
  candidate_profile_id: string;
  added_at: string;
}

// --- Outreach ---
export interface OutreachResponse {
  id: string;
  candidate_profile_id: string;
  candidate_full_name: string | null;
  created_by_user_id: string;
  content_source: 'ai_draft' | 'template';
  subject: string;
  body: string;
  sent_status: 'not_sent' | 'sent' | 'failed';
  sent_at: string | null;
  created_at: string;
}

// --- Interview Questions ---
export interface QuestionSetResponse {
  id: string;
  candidate_profile_id: string;
  candidate_full_name: string | null;
  job_description_id: string;
  job_description_title: string | null;
  generated_by_user_id: string;
  question_payload: Record<string, unknown>;
  created_at: string;
}
```

---

### ✅ Sprint 1 Checkpoint

Run through this before closing the session:

- [ ] `docker compose up --build` starts all containers without errors
- [ ] `http://localhost:5173` loads the React app
- [ ] `http://localhost:8000/` returns backend welcome message
- [ ] `http://localhost:8000/docs` loads Swagger UI
- [ ] CORS allows requests from `http://localhost:5173` (check `BACKEND_CORS_ORIGINS`)
- [ ] Hot-reload works — file save triggers browser update without rebuild
- [ ] All 7 API module stubs exist in `src/api/`
- [ ] `src/types/index.ts` exports all interfaces below
- [ ] React Router renders placeholder pages for each section
- [ ] No TypeScript errors (`docker compose exec frontend npx tsc --noEmit`)
- [ ] **Commit:** `git commit -m "sprint-1: docker compose verified, scaffold, types, api client"`
- [ ] **Close session** — start Sprint 2 fresh

---

## Sprint 2 — Resume Upload & Management UI

**Goal:** Build the full resume upload flow and the resume list page.

### Context Reminder (paste at top of session)

> See **Project Context Block** above. Focus area: `POST /api/v1/upload/batch-parse`, `GET /api/v1/upload/`, `GET /api/v1/upload/{id}`, `PATCH /api/v1/upload/{id}`, `DELETE /api/v1/upload/{id}`.

### Tasks

- [ ] `src/api/resumes.ts` — implement all 5 endpoints:
  - `batchParseResumes(files: File[], uploadedByUserId?: string): Promise<BatchParseResponse>`
  - `listResumes(params?): Promise<PaginatedResponse<ResumeResponse>>`
  - `getResume(id: string): Promise<ResumeResponse>`
  - `updateResume(id: string, data): Promise<ResumeResponse>`
  - `deleteResume(id: string, deleteFile?: boolean): Promise<{ deleted: boolean; resume_id: string }>`
- [ ] `src/pages/ResumesPage.tsx`:
  - Drag-and-drop / file picker — **PDF only** (validate before upload, reject non-PDFs with inline error)
  - Upload button → calls `batchParseResumes`
  - Loading spinner during upload (warn: "This may take 30+ seconds for large batches")
  - Paginated list of resumes with status badges (`uploaded` / `processing` / `processed` / `failed`)
  - Delete button per row → confirm modal → `deleteResume`
- [ ] Error handling: display `detail` from error response in a toast or inline message
- [ ] Hook: `src/hooks/useResumes.ts` — encapsulates list fetch + pagination state

### API Notes for This Sprint

```
POST /api/v1/upload/batch-parse
  Content-Type: multipart/form-data
  Field: files[] (PDF only — validate on frontend before sending)
  Field: uploaded_by_user_id (optional UUID)
  ⚠️ Synchronous — blocks until LLM parses all PDFs. Show loading state.

DELETE /api/v1/upload/{resume_id}?delete_file=false
  Returns 200 { deleted: true, resume_id: "uuid" }  ← NOT 204
```

---

### ✅ Sprint 2 Checkpoint

- [ ] PDF-only validation fires before any network request
- [ ] Loading state shown during `batch-parse` (30s+ warning visible)
- [ ] Resume list renders with correct status badges
- [ ] Delete works and removes item from list
- [ ] API errors surface readable messages to the user
- [ ] No TypeScript errors
- [ ] **Commit:** `git commit -m "sprint-2: resume upload and management UI"`
- [ ] **Close session** — start Sprint 3 fresh

---

## Sprint 3 — Job Descriptions UI

**Goal:** CRUD interface for job descriptions.

### Context Reminder

> See **Project Context Block**. Focus area: `/api/v1/job-descriptions`.

### Tasks

- [ ] `src/api/jobDescriptions.ts` — implement all 5 endpoints:
  - `createJobDescription(data): Promise<JobDescriptionResponse>`
  - `listJobDescriptions(params?): Promise<PaginatedResponse<JobDescriptionResponse>>`
  - `getJobDescription(id: string): Promise<JobDescriptionResponse>`
  - `updateJobDescription(id: string, data): Promise<JobDescriptionResponse>`
  - `deleteJobDescription(id: string): Promise<{ deleted: boolean; job_description_id: string }>`
- [ ] `src/pages/JobDescriptionsPage.tsx`:
  - Paginated list with active/inactive badge
  - "New JD" button → inline form or modal with `title` (optional) and `jd_text` (required, textarea)
  - Edit button → populates form with existing data → calls `updateJobDescription`
  - Toggle active/inactive via PATCH `{ is_active: false }`
  - Delete button → confirm → `deleteJobDescription`
- [ ] `src/hooks/useJobDescriptions.ts`

### API Notes for This Sprint

```
POST /api/v1/job-descriptions/
  Required: jd_text (min 1 char), created_by_user_id (UUID)
  Optional: title (max 255 chars)
  Returns: 201 Created

PATCH /api/v1/job-descriptions/{id}
  All fields optional. Only sent fields are changed.
  422 if jd_text is empty string.

DELETE returns: 200 { deleted: true, job_description_id: "uuid" }  ← NOT 204
```

---

### ✅ Sprint 3 Checkpoint

- [ ] Can create a new JD with required fields validated
- [ ] List shows active/inactive state; toggling works
- [ ] Edit form pre-fills correctly
- [ ] Delete removes the item from the list
- [ ] `created_by_user_id` wired up (even if hardcoded for now)
- [ ] No TypeScript errors
- [ ] **Commit:** `git commit -m "sprint-3: job descriptions CRUD"`
- [ ] **Close session** — start Sprint 4 fresh

---

## Sprint 4 — Candidate Scoring UI

**Goal:** Build the scoring interface — job selector, candidate selector, weight tuner, and results view.

### Context Reminder

> See **Project Context Block**. Focus area: `POST /api/v1/score/`.

### Tasks

- [ ] `src/api/scoring.ts`:
  - `scoreCandidates(data): Promise<ScoreResponse>`
- [ ] `src/pages/ScoringPage.tsx`:
  - Step 1: Select a Job Description (dropdown from JD list)
  - Step 2: Select candidates (optional — leave empty to score all)
  - Step 3: Configure scoring:
    - `score_threshold` slider (0–100, default 50)
    - `section_weights` — sliders for `skills`, `experience`, `projects`, `education`, `summary` (default weights pre-filled)
    - `batch_size` input (1–50, default 10)
  - Submit → loading state (warn: "Scoring may take a while for large candidate sets")
  - Results table: candidate ID, total score, pass/fail badge, rationale, expandable component scores
- [ ] `src/components/ScoreResultsTable.tsx` — reusable results renderer

### API Notes for This Sprint

```
POST /api/v1/score/
  Required: job_description_id (UUID), initiated_by_user_id (UUID)
  Optional: score_threshold (default 50.0), candidate_profile_ids[], section_weights{}, batch_size (default 10)
  ⚠️ Synchronous — blocks while LLM evaluates batches. Show loading feedback.
  ⚠️ scores[] uses camelCase: candidateId, totalScore, passedThreshold, componentScores
      All other API fields use snake_case.
  404 if JD not found or no candidates exist
  422 if all section weights are 0
```

---

### ✅ Sprint 4 Checkpoint

- [ ] Scoring form submits with correct payload structure
- [ ] Loading state displays during synchronous LLM call
- [ ] Results render with pass/fail badges and component score breakdown
- [ ] camelCase fields from scores array mapped correctly (no snake_case confusion)
- [ ] 422 "all weights are 0" error shown clearly
- [ ] No TypeScript errors
- [ ] **Commit:** `git commit -m "sprint-4: candidate scoring UI"`
- [ ] **Close session** — start Sprint 5 fresh

---

## Sprint 5 — Chat / Recruiter Chatbot UI

**Goal:** Conversational chat interface with session memory.

### Context Reminder

> See **Project Context Block**. Focus area: `/api/v1/chat`.

### Tasks

- [ ] `src/api/chat.ts`:
  - `sendMessage(message: string, sessionId?: string, candidateLimit?: number): Promise<ChatResponse>`
  - `getChatHistory(sessionId: string): Promise<ChatHistoryResponse>`
  - `deleteSession(sessionId: string): Promise<{ session_id: string; deleted: boolean }>`
- [ ] `src/pages/ChatPage.tsx`:
  - Chat bubble UI (human / AI messages)
  - Input bar + send button
  - First message: omit `session_id` → backend generates one → store it in state
  - Subsequent messages: pass stored `session_id`
  - Show `candidates_in_scope` count per AI reply
  - "New chat" button → clears session ID from state (starts fresh session on next message)
  - Handle `session_id` gracefully — if backend returns 404 for a stored session (restart scenario), auto-start a new session
- [ ] `src/hooks/useChat.ts` — manages messages array + session ID state

### API Notes for This Sprint

```
POST /api/v1/chat/
  First turn: omit session_id
  Subsequent turns: pass session_id from previous response
  ⚠️ Sessions are IN-MEMORY ONLY — lost on backend restart.
     Handle 404 on session gracefully: clear stored ID and start fresh.
  Backend remembers last 5 messages per session.

GET /api/v1/chat/{session_id}   → full message history
DELETE /api/v1/chat/{session_id} → 200 { session_id, deleted: true }
```

---

### ✅ Sprint 5 Checkpoint

- [ ] First message starts a new session; subsequent messages pass session ID
- [ ] `candidates_in_scope` displayed per reply
- [ ] "New chat" correctly resets session
- [ ] 404 on stale session handled gracefully (auto-restart)
- [ ] No TypeScript errors
- [ ] **Commit:** `git commit -m "sprint-5: recruiter chatbot UI"`
- [ ] **Close session** — start Sprint 6 fresh

---

## Sprint 6 — Shortlists (Sessions, Turns, Collections, Items)

**Goal:** Persistent recruiter query history and saved candidate collections.

### Context Reminder

> See **Project Context Block**. Focus area: `/api/v1/shortlist`.

### Tasks

- [ ] `src/api/shortlist.ts` — implement all endpoints:

  **Sessions**
  - `createSession(userId, title?): Promise<SessionResponse>`
  - `listSessions(userId, params?): Promise<SessionResponse[]>`
  - `getSession(id): Promise<SessionResponse>`
  - `updateSession(id, title): Promise<SessionResponse>`
  - `deleteSession(id): Promise<void>` ← 204 No Content

  **Turns**
  - `createTurn(sessionId, data): Promise<TurnResponse>`
  - `listTurns(sessionId, params?): Promise<TurnResponse[]>`
  - `getTurn(id): Promise<TurnResponse>`
  - `deleteTurn(id): Promise<void>` ← 204 No Content

  **Collections**
  - `createCollection(data): Promise<CollectionResponse>`
  - `listCollections(userId, params?): Promise<CollectionResponse[]>`
  - `getCollection(id): Promise<CollectionResponse>`
  - `renameCollection(id, name): Promise<CollectionResponse>`
  - `deleteCollection(id): Promise<void>` ← 204 No Content

  **Items**
  - `addItem(collectionId, candidateProfileId): Promise<ShortlistItemResponse>`
  - `listItems(collectionId, params?): Promise<ShortlistItemResponse[]>`
  - `removeItem(collectionId, candidateId): Promise<void>` ← 204 No Content

- [ ] `src/pages/ShortlistPage.tsx`:
  - Sidebar: list of saved collections (per user)
  - Collection view: list of candidate items with remove button
  - "New collection" button → name input → `createCollection`
  - Rename collection inline
  - Session history panel: list of query sessions + turn count
  - Expandable session → shows turns (question + answer)

### API Notes for This Sprint

```
⚠️ DELETE endpoints here return 204 No Content (no body).
   This differs from /upload and /job-descriptions which return 200 + JSON.
   Do NOT try to parse the response body on 204.

Collection names are UNIQUE per user — 409 Conflict on duplicate.

GET /api/v1/shortlist/sessions/  requires ?user_id=UUID (mandatory filter)
GET /api/v1/shortlist/collections/ requires ?user_id=UUID (mandatory filter)

Items list: default limit 100 (max 500) — higher than the global default.
```

---

### ✅ Sprint 6 Checkpoint

- [ ] All 204 DELETE calls handled without body parsing
- [ ] 409 Conflict on duplicate collection name shows clear error
- [ ] Sessions list correctly filtered by user ID
- [ ] Collections list correctly filtered by user ID
- [ ] Items can be added and removed from a collection
- [ ] No TypeScript errors
- [ ] **Commit:** `git commit -m "sprint-6: shortlists, sessions, collections, items"`
- [ ] **Close session** — start Sprint 7 fresh

---

## Sprint 7 — Outreach Messages UI

**Goal:** Create, view, and send outreach emails to candidates.

### Context Reminder

> See **Project Context Block**. Focus area: `/api/v1/outreach`.

### Tasks

- [ ] `src/api/outreach.ts`:
  - `createOutreach(data): Promise<OutreachResponse>`
  - `listOutreach(params?): Promise<PaginatedResponse<OutreachResponse>>`
  - `getOutreach(id): Promise<OutreachResponse>`
  - `updateOutreach(id, data): Promise<OutreachResponse>`
  - `deleteOutreach(id): Promise<void>` ← 204 No Content
- [ ] `src/pages/OutreachPage.tsx`:
  - Paginated list of outreach messages, filterable by `sent_status` and candidate
  - Status badges: `not_sent` / `sent` / `failed`
  - "New message" → form with subject, body, candidate selector, `content_source` toggle (`ai_draft` / `template`)
  - "Mark as sent" button → PATCH `{ sent_status: "sent" }` → `sent_at` auto-filled by backend
  - Edit subject/body for unsent messages
  - Delete button → 204, remove from list

### API Notes for This Sprint

```
POST /api/v1/outreach/
  Required: candidate_profile_id, created_by_user_id, content_source, subject, body
  Returns: 201 Created

PATCH /api/v1/outreach/{id}
  When sent_status → "sent", backend auto-sets sent_at. Do NOT send sent_at from frontend.

DELETE returns: 204 No Content (no body)

GET list: total field reflects ALL matching records (not just page) — unlike other list endpoints.
```

---

### ✅ Sprint 7 Checkpoint

- [ ] Create form validates all required fields
- [ ] "Mark as sent" updates status and `sent_at` is displayed after response
- [ ] `sent_at` is never sent in PATCH payload
- [ ] 204 DELETE handled without body parsing
- [ ] Filters by `sent_status` work
- [ ] No TypeScript errors
- [ ] **Commit:** `git commit -m "sprint-7: outreach messages UI"`
- [ ] **Close session** — start Sprint 8 fresh

---

## Sprint 8 — Interview Questions UI

**Goal:** Create and view AI-generated interview question sets per candidate + job description.

### Context Reminder

> See **Project Context Block**. Focus area: `/api/v1/interview-questions`.

### Tasks

- [ ] `src/api/interviewQuestions.ts`:
  - `createQuestionSet(data): Promise<QuestionSetResponse>`
  - `listQuestionSets(params?): Promise<PaginatedResponse<QuestionSetResponse>>`
  - `getQuestionSet(id): Promise<QuestionSetResponse>`
  - `updateQuestionSet(id, payload): Promise<QuestionSetResponse>`
  - `deleteQuestionSet(id): Promise<void>` ← 204 No Content
- [ ] `src/pages/InterviewQuestionsPage.tsx`:
  - Filterable list by candidate and/or job description
  - "Generate question set" → selectors for candidate + JD + `question_payload` JSON input
  - View question set → renders `question_payload` (free-form JSON — display generically or with structure if known)
  - Edit (replace entire `question_payload`) via PATCH
  - Delete → 204

### API Notes for This Sprint

```
question_payload is FREE-FORM JSON.
  The backend does not validate its structure.
  Render it generically (JSON tree or key-value display) unless you define a fixed schema.

PATCH replaces the ENTIRE question_payload — it is not a partial update.

DELETE returns: 204 No Content (no body)

Required fields: candidate_profile_id, job_description_id, generated_by_user_id, question_payload
```

---

### ✅ Sprint 8 Checkpoint

- [ ] Question set creates successfully with all required fields
- [ ] `question_payload` renders without crashing on unexpected shapes
- [ ] PATCH sends the full replacement payload
- [ ] 204 DELETE handled without body parsing
- [ ] Filters by candidate and JD work
- [ ] No TypeScript errors
- [ ] **Commit:** `git commit -m "sprint-8: interview questions UI"`
- [ ] **Close session** — start Sprint 9 fresh

---

## Sprint 9 — Global Polish & Error Handling

**Goal:** Harden error states, unify UX, add final polish. No new features.

### Context Reminder

> See **Project Context Block**. This sprint touches all pages.

### Tasks

- [ ] Global error boundary component
- [ ] Toast / notification system for API errors (`detail` field)
- [ ] Audit all DELETE handlers — ensure 204 calls don't attempt JSON parse
- [ ] Audit all loading states — especially `batch-parse` and `score` (both synchronous + slow)
- [ ] Consistent empty states (no data yet) across all pages
- [ ] Consistent pagination controls across all list pages
- [ ] Handle 409 Conflict (collection duplicate name) with clear inline error
- [ ] Handle stale chat session (404 on session_id) — auto-reset
- [ ] Verify `scores[]` camelCase mapping is correct everywhere
- [ ] Final `docker compose exec frontend npx tsc --noEmit` — zero errors
- [ ] Run `docker compose up` and smoke-test each feature against live backend

---

### ✅ Sprint 9 Checkpoint (Final)

- [ ] Zero TypeScript errors (`docker compose exec frontend npx tsc --noEmit`)
- [ ] All pages have loading + error + empty states
- [ ] 204 endpoints never crash on response body parsing
- [ ] 409 Conflict errors surface cleanly to user
- [ ] Chat session gracefully recovers from backend restart
- [ ] Score results render camelCase fields correctly
- [ ] Smoke test passed against live backend
- [ ] **Commit:** `git commit -m "sprint-9: global polish and error handling"`
- [ ] 🎉 **Done**

---

## Quick Reference — HTTP Status Codes

| Status | Meaning                        | Returned by                            |
|--------|--------------------------------|----------------------------------------|
| 200    | OK                             | Most GET, PATCH, DELETE (upload, JD)   |
| 201    | Created                        | POST job-descriptions, sessions, turns, collections, items, outreach, interview-questions |
| 204    | No Content                     | DELETE sessions, turns, collections, items, outreach, interview-questions |
| 400    | Bad request                    | Non-PDF upload, missing files          |
| 404    | Not found                      | Any resource by ID                     |
| 409    | Conflict                       | Duplicate collection name, duplicate item in collection |
| 422    | Validation error               | Invalid field values                   |
| 500    | Server error                   | LLM failure, unexpected exceptions     |

> ⚠️ **DELETE inconsistency:** `/upload` and `/job-descriptions` DELETE → **200 + JSON body**. All other DELETE endpoints → **204 No Content**. Handle both patterns.

---

## Quick Reference — Enum Values

| Enum           | Values                                            |
|----------------|---------------------------------------------------|
| UploadStatus   | `uploaded` `processing` `processed` `failed`      |
| ProfileStatus  | `draft` `reviewed` `approved` `archived`          |
| MatchRunStatus | `running` `completed` `failed`                    |
| ContentSource  | `ai_draft` `template`                             |
| SentStatus     | `not_sent` `sent` `failed`                        |
| UserStatus     | `active` `suspended`                              |
| RoleName       | `admin` `recruiter` `viewer`                      |
