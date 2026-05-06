---
phase: 01-foundation-design-system
plan: "02"
subsystem: frontend
tags: [api-client, axios, tanstack-query, error-handling, typescript, types]
dependency_graph:
  requires:
    - frontend/package.json — axios 1.7, @tanstack/react-query 5.59, sonner 1.7 (from 01-01)
    - frontend/src/vite-env.d.ts — vite/client triple-slash ref (from 01-01)
  provides:
    - frontend/src/api/types.ts — TypeScript interfaces for all backend response shapes
    - frontend/src/api/errors.ts — ApiError class + parseAxiosError normalization
    - frontend/src/api/client.ts — Axios instance with PII-safe interceptor
    - frontend/src/api/queryClient.ts — TanStack QueryClient with global toast on error
    - frontend/src/api/endpoints/upload.ts — Resume upload/management API
    - frontend/src/api/endpoints/jobDescriptions.ts — Job descriptions API
    - frontend/src/api/endpoints/scoring.ts — Candidate scoring API
    - frontend/src/api/endpoints/chat.ts — Recruiter chatbot API
    - frontend/src/api/endpoints/shortlist.ts — Sessions/turns/collections/items API
    - frontend/src/api/endpoints/outreach.ts — Outreach messages API
    - frontend/src/api/endpoints/interviewQuestions.ts — Interview questions API
    - frontend/src/api/index.ts — Barrel export (api namespace + all types)
  affects: []
tech_stack:
  added: []
  patterns:
    - Normalized ApiError class with status/kind/detail/fieldErrors
    - parseAxiosError maps Axios errors to typed ApiError (network/validation/not_found/conflict/server/unknown)
    - TanStack QueryCache + MutationCache onError → toast.error for non-validation errors (FOUND-07)
    - Validation errors (422) skip global toast — forms render fieldErrors inline
    - FastAPI 422 detail array parsed into FieldError[] with loc[].join(".") as field name
    - PII protection — no request/response body logging in interceptor
    - VITE_API_BASE_URL env var controls base URL (never hardcoded)
    - 204 No Content DELETEs return Promise<void>; 200 DELETEs return typed body
    - Extended timeouts for synchronous LLM endpoints (batchParse: 5min, score: 10min)
key_files:
  created:
    - frontend/src/api/types.ts
    - frontend/src/api/errors.ts
    - frontend/src/api/client.ts
    - frontend/src/api/queryClient.ts
    - frontend/src/api/endpoints/upload.ts
    - frontend/src/api/endpoints/jobDescriptions.ts
    - frontend/src/api/endpoints/scoring.ts
    - frontend/src/api/endpoints/chat.ts
    - frontend/src/api/endpoints/shortlist.ts
    - frontend/src/api/endpoints/outreach.ts
    - frontend/src/api/endpoints/interviewQuestions.ts
    - frontend/src/api/index.ts
  modified: []
decisions:
  - "204 No Content DELETE endpoints (shortlist, outreach, interview-questions) return Promise<void>; 200 DELETE endpoints (upload, job-descriptions) return typed body — handles BACKEND.md note 9"
  - "shortlistApi structured as namespaced object (sessions/turns/collections/items) rather than flat functions — maps 1:1 to backend sub-resource structure"
  - "CandidateScore/ComponentScore use camelCase (candidateId, totalScore, componentScores, etc.) per BACKEND.md note 7 — LLM passthrough"
  - "Extended scoring timeout to 10min (vs default 60s) and batchParse to 5min — synchronous LLM operations per BACKEND.md notes 4/5"
metrics:
  duration_minutes: 4
  tasks_completed: 3
  tasks_total: 3
  files_created: 12
  files_modified: 0
  completed_date: "2026-04-28"
---

# Phase 1 Plan 02: API Client + TanStack Query Layer Summary

Typed Axios + TanStack Query API layer: 354-line types file mirroring all BACKEND.md shapes, ApiError normalization with FastAPI 422 field parsing, global non-validation toast via QueryCache/MutationCache, and 7 endpoint modules exposing every backend endpoint through a single `api` namespace.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Define TypeScript interfaces for every backend response shape | f615020 | frontend/src/api/types.ts |
| 2 | Build Axios client + ApiError + QueryClient | 6d2448e | frontend/src/api/errors.ts, client.ts, queryClient.ts |
| 3 | Create endpoint modules (one per resource group) | 54c284a | frontend/src/api/endpoints/*.ts, frontend/src/api/index.ts |

## Verification Results

- `npm run typecheck` (tsc -b) exits 0, no TypeScript errors
- `npm run build` exits 0, produces dist/index.html (143 kB JS / 46 kB gzip)
- `grep -q "candidateId" frontend/src/api/types.ts` passes
- `grep -q "VITE_API_BASE_URL" frontend/src/api/client.ts` passes
- `grep -q "toast.error" frontend/src/api/queryClient.ts` passes
- All 7 endpoint modules confirmed with acceptance criteria grep checks

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking issue] vite-env.d.ts was missing**
- **Found during:** Task 2 verification — `npm run typecheck` failed with `Property 'env' does not exist on type 'ImportMeta'`
- **Issue:** `frontend/src/vite-env.d.ts` was not present in the working tree (the git status showed it as deleted in a prior state), so `import.meta.env` had no type declaration
- **Fix:** Created `frontend/src/vite-env.d.ts` with `/// <reference types="vite/client" />` — the standard Vite triple-slash reference that adds `ImportMeta.env` and `ImportMetaEnv` to the type system
- **Files modified:** frontend/src/vite-env.d.ts (created)
- **Commit:** 6d2448e

## Known Stubs

None — all endpoint functions call the real Axios client. No mock data or placeholder returns. The `api` namespace is fully wired.

## Threat Surface Scan

- `client.ts` interceptor explicitly comments against logging request/response bodies (PII guard for resume PDFs and candidate data)
- `ApiError.detail` is truncated to 500 chars before user display (prevents server message reflection)
- `withCredentials: false` — no cookie-based auth leakage
- `VITE_API_BASE_URL` read from `import.meta.env` — never hardcoded
- 422 `fieldErrors` use `loc[].join(".")` — only sanitized field-path strings reach the UI, not raw server strings
- No new network endpoints, auth paths, or schema changes introduced by this plan

## Self-Check: PASSED

- frontend/src/api/types.ts: FOUND (354 lines, contains ResumeResponse, candidateId, UploadStatus, ScoreResponse)
- frontend/src/api/errors.ts: FOUND (contains class ApiError, fieldErrors, parseAxiosError)
- frontend/src/api/client.ts: FOUND (contains VITE_API_BASE_URL, throw parseAxiosError)
- frontend/src/api/queryClient.ts: FOUND (contains QueryClient, QueryCache, MutationCache, toast.error)
- frontend/src/api/endpoints/upload.ts: FOUND (contains batchParse)
- frontend/src/api/endpoints/scoring.ts: FOUND (contains score)
- frontend/src/api/endpoints/chat.ts: FOUND (contains send, getHistory)
- frontend/src/api/endpoints/shortlist.ts: FOUND (contains sessions, collections, turns, items)
- frontend/src/api/endpoints/outreach.ts: FOUND
- frontend/src/api/endpoints/interviewQuestions.ts: FOUND
- frontend/src/api/index.ts: FOUND (contains export const api)
- Commits f615020, 6d2448e, 54c284a: present in git log
