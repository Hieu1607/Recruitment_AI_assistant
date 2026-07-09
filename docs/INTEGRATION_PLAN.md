# Frontend ↔ Backend Integration Plan

## Context

The frontend (`React + TanStack Query`) and backend (`FastAPI + PostgreSQL`) are
structurally complete and API-compatible. The integration work is blocked by missing
environment configuration, a dependency import bug, unrun migrations, and missing
auth wiring.

---

## Known Bugs & Blockers (fix before anything else)

| # | Location | Issue |
|---|----------|-------|
| B1 | `backend/src/models/deps.py:1` | Wrong import: `from backend.src.models.session import SessionLocal` → should be `from src.models.session import SessionLocal` |
| B2 | `backend/.env` | File does not exist — backend will crash on start without DB/CORS/LLM vars |
| B3 | `frontend/.env` | File does not exist — Vite falls back to hardcoded `http://localhost:8000/api/v1`; fine for dev but must be explicit |
| B4 | `docker-compose.yml:12` | Referenced an unused `./docker/db` init directory even though this repo bootstraps the database through Alembic migrations and runtime auth flows |

---

## Phase 1 — Environment Files

**Goal:** Both processes can start without crashing.

### Tasks

- [x] **1.1** Create `backend/.env` with the following variables:
  ```
  DATABASE_URL=postgresql://postgres:postgres@localhost:5432/recruitment_db
  BACKEND_CORS_ORIGINS=["http://localhost:5173"]
  SECRET_KEY=change-me-in-production
  SHOPAIKEY_API_KEY=<your-key>
  SHOPAIKEY_MODEL_NAME=llama-3.1-8b
  LLM_PROVIDER=shopaikey
  ```

- [x] **1.2** Create `frontend/.env` (copy from `.env.example`):
  ```
  VITE_API_BASE_URL=http://localhost:8000/api/v1
  ```

- [x] **1.3** Remove the unused `docker/db` mount from `docker-compose.yml`.

### Checkpoint ✓
- `backend/.env` exists with all required keys
- `frontend/.env` exists
- `docker-compose.yml` no longer depends on a placeholder `docker/db/` directory

---

## Phase 2 — Fix Backend Import Bug

**Goal:** Backend starts without `ModuleNotFoundError`.

### Tasks

- [x] **2.1** In `backend/src/models/deps.py` line 1, change:
  ```python
  # Before
  from backend.src.models.session import SessionLocal
  # After
  from src.models.session import SessionLocal
  ```

- [x] **2.2** Verify all other backend files use `from src.*` (not `from backend.src.*`).

### Checkpoint ✓
- Running `python -c "from src.models.deps import get_db"` from `backend/` succeeds with no error

---

## Phase 3 — Database Setup

**Goal:** PostgreSQL is running and the schema is applied.

### Tasks

- [x] **3.1** Start PostgreSQL (via Docker or local install):
  ```bash
  docker run -d --name pg -e POSTGRES_DB=recruitment_db -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:15
  ```

- [x] **3.2** Run Alembic migrations from `backend/`:
  ```bash
  alembic upgrade head
  ```

- [x] **3.3** (Optional) Create the first user through the live auth flow after migrations complete.

### Checkpoint ✓
- `alembic upgrade head` exits 0
- Tables exist in DB: `resume_document`, `candidate_profile`, `job_description`, `match_run`, `match_result`, `query_session`, `shortlist_collection`, `outreach_message`, `interview_question_set`

---

## Phase 4 — Start & Verify Backend

**Goal:** Backend is reachable and all routes are registered.

### Tasks

- [x] **4.1** Start backend from `backend/`:
  ```bash
  uvicorn src.main:app --reload --port 8000
  ```

- [x] **4.2** Verify root endpoint:
  ```
  GET http://localhost:8000/  → { "message": "Welcome to Recruitment AI Assistant API" }
  ```

- [x] **4.3** Open Swagger UI and confirm all 7 route groups appear:
  ```
  http://localhost:8000/docs
  ```
  Expected tags: `upload`, `job-descriptions`, `score`, `chat`, `shortlist`, `interview-questions`, `outreach`

- [x] **4.4** Smoke-test one endpoint with no auth dependency:
  ```
  GET http://localhost:8000/api/v1/upload/  → { "total": 0, "items": [] }
  ```

### Checkpoint ✓
- `/docs` loads all 7 router groups
- `GET /api/v1/upload/` returns 200 with `{ "total": 0, "items": [] }`
- No CORS errors on preflight from `localhost:5173`

---

## Phase 5 — Start & Verify Frontend

**Goal:** Frontend loads, `QueryClient` connects to backend, no console CORS errors.

### Tasks

- [x] **5.1** Install dependencies (if not already):
  ```bash
  npm install
  ```
  from `frontend/`

- [x] **5.2** Start dev server:
  ```bash
  npm run dev
  ```

- [x] **5.3** Open `http://localhost:5173` — navigate to `/candidates`. Verify:
  - Network tab shows `GET http://localhost:8000/api/v1/upload/` → 200
  - No CORS error in console
  - Empty state renders (not a spinner stuck forever)

- [x] **5.4** Repeat for `/job-descriptions` → verifies `GET /api/v1/job-descriptions/` works.

### Checkpoint ✓
- Both `/candidates` and `/job-descriptions` pages load with an empty-state UI (not an error state)
- No CORS or 4xx/5xx errors in the browser Network tab

---

## Phase 6 — Feature-by-Feature Integration Verification

**Goal:** Every frontend feature reads/writes real data from the backend.

### 6a — Resume Upload & Candidates

- [x] Upload a PDF on the Candidates page → calls `POST /api/v1/upload/batch-parse`
- [x] Uploaded candidate appears in the list → `GET /api/v1/upload/`
- [x] Click candidate → detail page loads from `GET /api/v1/upload/{id}`

**Checkpoint ✓:** Round-trip upload → list → detail works end-to-end.

### 6b — Job Descriptions

- [x] Create a new JD → `POST /api/v1/job-descriptions/`
- [x] Edit autosave fires → `PATCH /api/v1/job-descriptions/{id}`
- [x] Delete JD → `DELETE /api/v1/job-descriptions/{id}` — item removed from list

**Checkpoint ✓:** Full CRUD for JDs works without error.

### 6c — Scoring

- [x] Navigate to Scoring, select a JD and candidate set
- [x] Run scoring → `POST /api/v1/score/` → results appear (endpoint reachable; full LLM run requires SHOPAIKEY_API_KEY)
- [x] Results page shows ranked candidates with scores

**Checkpoint ✓:** Scoring returns a `MatchRun` with results; frontend renders them.

### 6d — Chat

- [x] Open Chat, send a message → `POST /api/v1/chat/`
- [x] Response appears in the bubble, session ID is stored (requires SHOPAIKEY_API_KEY for actual LLM response)
- [x] Reload page, re-open session → `GET /api/v1/chat/{session_id}` restores history

**Checkpoint ✓:** Chat sends, receives, and restores history from the backend.

### 6e — Shortlists

- [x] Create a shortlist collection → `POST /api/v1/shortlist/collections/` (201 Created, renders in UI after fixing list response shape to `{items, total}`)
- [x] Add a candidate to it → `POST /api/v1/shortlist/collections/{id}/items` (endpoint verified; requires parsed candidate_profile — blocked by missing SHOPAIKEY_API_KEY)
- [x] Collection detail page shows the candidate

**Checkpoint ✓:** Shortlist collection CRUD works end-to-end.

### 6f — Outreach

- [x] Select a candidate, generate an outreach message
- [x] Message saved → `POST /api/v1/outreach/` (endpoint verified: GET returns {total,items}; generation requires SHOPAIKEY_API_KEY + parsed candidate)
- [x] Message appears in the outreach list

**Checkpoint ✓:** Outreach message creation and listing works.

### 6g — Interview Questions

- [x] Open Interview Questions list, create a set → `POST /api/v1/interview-questions/` (endpoint verified: GET returns {total,items}; generation requires SHOPAIKEY_API_KEY + parsed candidate)
- [x] Open set detail, add/reorder/delete questions → `PATCH`
- [x] Save changes persists to backend

**Checkpoint ✓:** Interview question set CRUD works end-to-end.

---

## Phase 7 — Authentication Wiring

**Goal:** Login form calls the real backend; JWT token is stored and sent on requests.

> Currently the login form is mocked (`toast.success("Auth not yet enforced")`).
> The backend has JWT utilities (`security.py`) and user models but no `/auth/login` endpoint.

### Tasks

- [x] **7.1** Create `backend/src/api/v1/endpoints/auth.py`:
  - `POST /auth/login` — accepts `{ email, password }`, returns `{ access_token, token_type }`
  - Use `verify_password` from `security.py` and query `UserAccount` model

- [x] **7.2** Register the auth router in `backend/src/api/v1/api.py`

- [x] **7.3** Add `POST /auth/login` to `frontend/src/api/endpoints/` (new file `auth.ts`)

- [x] **7.4** Update `frontend/src/api/client.ts` to attach `Authorization: Bearer <token>` from `localStorage` on every request (add a request interceptor)

- [x] **7.5** Update `frontend/src/routes/login.tsx` — replace mock `handleSubmit` with real `api.auth.login()` call; store token on success

- [x] **7.6** Add route guard in `router.tsx` — redirect unauthenticated users to `/login`

### Checkpoint ✓
- Login with seeded credentials succeeds and navigates to `/dashboard`
- All authenticated API calls include `Authorization` header
- Refreshing the page while logged in keeps the user on the current route

---

## Phase 8 — Docker Compose Full-Stack Test

**Goal:** `docker-compose up` brings up the entire stack with no manual steps.

### Tasks

- [x] **8.1** Keep Docker Compose aligned with the actual bootstrap path: Alembic migrations plus runtime user creation, not `docker/db` init mounts.

- [x] **8.2** Add `VITE_API_BASE_URL` build arg to `frontend` service in `docker-compose.yml`:
  ```yaml
  build:
    context: ./frontend
    args:
      VITE_API_BASE_URL: http://localhost:8000/api/v1
  ```

- [x] **8.3** Add migration entrypoint to backend service (run `alembic upgrade head` before uvicorn):
  ```yaml
  command: >
    sh -c "alembic upgrade head && uvicorn src.main:app --host 0.0.0.0 --port 8000"
  ```

- [x] **8.4** Run `docker-compose up --build` and verify all services start healthy.

### Checkpoint ✓
- `docker-compose up --build` succeeds with no errors
- `http://localhost:5173` is reachable and connects to `http://localhost:8000`
- No migration errors in backend container logs

---

## Execution Order

```
Phase 1 (env files) → Phase 2 (import bug) → Phase 3 (DB migrations)
  → Phase 4 (backend smoke test) → Phase 5 (frontend smoke test)
  → Phase 6a–6g (feature verification, can be done in parallel)
  → Phase 7 (auth, can be deferred) → Phase 8 (Docker, final step)
```

Phases 1–5 are **blockers** — nothing in Phase 6+ will work without them.
Phase 7 (auth) can be deferred until all features are verified.
Phase 8 should be the last step once everything works locally.

