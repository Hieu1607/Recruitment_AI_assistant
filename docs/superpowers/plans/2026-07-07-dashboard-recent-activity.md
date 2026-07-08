# Dashboard Recent Activity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dashboard's thin client-side recent activity list with a dedicated activity feed that surfaces meaningful recruiter activity without duplicating notifications.

**Architecture:** Add a backend activity aggregation endpoint that merges recent events from resume, shortlist, outreach, interview, and scoring domains. Then switch the dashboard to consume the structured feed and format items locally.

**Tech Stack:** FastAPI, SQLAlchemy ORM, React Router, TanStack Query, TypeScript

---

### Task 1: Add backend regression coverage for activity aggregation

**Files:**
- Create: `backend/tests/test_activity_endpoints.py`

- [ ] **Step 1: Write failing endpoint tests for merged activity output**

Add tests that seed recent resume, shortlist, outreach, interview, and scoring records for one recruiter and assert:

- items are returned newest first
- other users' data is excluded
- resume upload is hidden when the same resume already has a processed or failed event

- [ ] **Step 2: Run the new backend tests to verify they fail**

Run: `pytest backend/tests/test_activity_endpoints.py -q`

- [ ] **Step 3: Use the failures to drive the backend implementation**

Implement only the minimum backend code needed for the tests to pass before moving to the dashboard.

- [ ] **Step 4: Re-run the backend tests**

Run: `pytest backend/tests/test_activity_endpoints.py -q`

### Task 2: Implement backend activity service and endpoint

**Files:**
- Create: `backend/src/services/activity_service.py`
- Create: `backend/src/api/v1/endpoints/activities.py`
- Modify: `backend/src/api/v1/api.py`

- [ ] **Step 1: Add the activity aggregation service**

Build a service that:

- collects recent events per domain with small bounded queries
- converts them into a common activity item shape
- applies anti-noise rules
- sorts and truncates the merged list

- [ ] **Step 2: Add the FastAPI endpoint and response models**

Expose `GET /activities/` with `limit` and optional `job_id`.

- [ ] **Step 3: Wire the router into the API**

Register the new router in `backend/src/api/v1/api.py`.

- [ ] **Step 4: Re-run the backend tests**

Run: `pytest backend/tests/test_activity_endpoints.py -q`

### Task 3: Switch the dashboard to the new activity source

**Files:**
- Create: `frontend/src/api/endpoints/activities.ts`
- Modify: `frontend/src/api/index.ts`
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/routes/dashboard.tsx`

- [ ] **Step 1: Add frontend API types and endpoint client**

Define the activity response types and expose `api.activities.list(...)`.

- [ ] **Step 2: Replace dashboard activity assembly**

Remove the client-side merge of upload and outreach draft data and replace it with a single `useQuery` against the activity endpoint.

- [ ] **Step 3: Format activity rows locally**

Map activity kinds to dashboard icons, localized labels/subtitles, and optional links without changing notification-center behavior.

- [ ] **Step 4: Verify TypeScript/build health**

Run: `npm run build`

### Task 4: Final verification

**Files:**
- No additional files expected

- [ ] **Step 1: Run focused backend verification**

Run: `pytest backend/tests/test_activity_endpoints.py backend/tests/test_notifications_endpoints.py -q`

- [ ] **Step 2: Run frontend build verification**

Run: `npm run build`

- [ ] **Step 3: Review the diff against the goal**

Confirm:

- dashboard shows meaningful activity beyond uploads and drafts
- notification center remains separate
- feed avoids obvious duplicate/noisy items
