# Shortlist Dispatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build staged batch dispatch from shortlist collections to outreach drafts and interview invitations.

**Architecture:** Add a small dispatch layer to the existing shortlist endpoint module. The frontend shortlist collection route consumes one summary endpoint and two batch command endpoints while preserving existing shortlist item behavior.

**Tech Stack:** FastAPI, SQLAlchemy, Pydantic, pytest, React, TanStack Query, TypeScript, Vite.

---

### Task 1: Backend Dispatch Tests

**Files:**
- Modify: `backend/tests/test_shortlist_endpoints.py`

- [ ] Add tests for dispatch summary, outreach draft batch, and interview invitation batch.
- [ ] Run targeted pytest and verify the new tests fail because endpoints do not exist.

### Task 2: Backend Dispatch Endpoints

**Files:**
- Modify: `backend/src/api/v1/endpoints/shortlist.py`

- [ ] Add Pydantic request/response models for dispatch summary and batch results.
- [ ] Add helpers to load collection candidates with candidate snapshots.
- [ ] Add helpers to find latest outreach and interview invitation per candidate.
- [ ] Implement `GET /shortlist/collections/{collection_id}/dispatch-summary`.
- [ ] Implement `POST /shortlist/collections/{collection_id}/outreach-drafts`.
- [ ] Implement `POST /shortlist/collections/{collection_id}/interview-invitations`.
- [ ] Run targeted pytest and verify green.

### Task 3: Frontend API Surface

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/endpoints/shortlist.ts`

- [ ] Add TypeScript types for dispatch summary, batch requests, and batch results.
- [ ] Add API client methods under `api.shortlist.dispatch`.
- [ ] Run TypeScript build to catch type errors.

### Task 4: Shortlist Collection UI

**Files:**
- Modify: `frontend/src/routes/shortlists/collection.tsx`

- [ ] Replace per-row candidate fetches with dispatch summary data.
- [ ] Add row checkboxes and selected count action bar.
- [ ] Add outreach draft review/confirm modal.
- [ ] Add interview invite review/confirm modal with template selector.
- [ ] Refresh dispatch summary after successful batch action.
- [ ] Run TypeScript build.

### Task 5: Verification and Commit

**Files:**
- All modified files

- [ ] Run targeted backend tests.
- [ ] Run frontend build.
- [ ] Review `git diff`.
- [ ] Commit implementation with a focused message.
