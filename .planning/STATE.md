# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-27)

**Core value:** Recruiters can run their full hiring loop (upload → score → chat → shortlist → outreach → interview prep) through a polished editorial UI talking to the existing FastAPI backend.
**Current focus:** Phase 1 — Foundation & Design System

## Current Position

Phase: 1 of 12 (Foundation & Design System)
Plan: 3 of 4 (01-01 + 01-02 + 01-03 complete, ready for 01-04)
Status: Executing — Wave 3 complete
Last activity: 2026-04-28 — Executed 01-03 (React Router v7 routing skeleton: 18 routes, lazy chunks, RouterProvider)

Progress: [███░░░░░░░] 6%

## Performance Metrics

**Velocity:**
- Total plans completed: 3
- Average duration: 4-5 minutes
- Total execution time: 0.19 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 (Foundation & Design System) | 3/4 | 13 min | 4.3 min |

**Recent Trend:**
- Last 5 plans: 01-01 (5m), 01-02 (4m), 01-03 (4m)
- Trend: Stable

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table. Recent foundational picks:

- Phase 0 (init): Rebuild frontend from scratch in Vite + React 18 + TS + Tailwind v4 + shadcn/ui + TanStack Query + React Router v7 (do not touch `frontend_backup/`)
- Phase 0 (init): Forest-green `#1F3A2E` accent on off-white `#FAFAF7` confirmed from PNGs; Fraunces + Geist + Geist Mono as the type stack
- Phase 0 (init): Skip real auth enforcement — backend doesn't enforce yet; ship Login UI as visual surface only
- Phase 0 (init): Granularity "fine" — 12 phases, roughly one screen / tight bundle per phase, with parallelization enabled after Phases 1-2
- Phase 1 (01-01): Use `tsc -b` (composite build mode) instead of `tsc --noEmit -b` — invalid flag combo in TS 5.9; `noEmit: true` in tsconfig achieves the same effect
- Phase 1 (01-01): No postcss.config.js — @tailwindcss/vite plugin handles Tailwind v4 processing alone; dual pipeline breaks @theme resolution
- Phase 1 (01-02): 204 No Content DELETE endpoints (shortlist, outreach, interview-questions) return Promise<void>; 200 DELETE endpoints (upload, job-descriptions) return typed body
- Phase 1 (01-02): shortlistApi structured as sessions/turns/collections/items namespaces — maps 1:1 to backend sub-resource structure
- Phase 1 (01-02): Scoring timeout extended to 10min, batchParse to 5min — synchronous LLM operations per BACKEND.md
- Phase 1 (01-03): Router lazy() helper wraps React Router's built-in lazy() to unify { Component } shape — no external library needed
- Phase 1 (01-03): Authenticated shell (Outlet wrapper) stubbed in router.tsx — plan 01-04 replaces it with real AppShell
- Phase 1 (01-03): chat.tsx shared by /chat and /chat/:sessionId — single file handles both empty and loaded session states

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 7 (Candidate Detail) header CTAs link into Phases 5, 9, 10, 11 — if those phases haven't shipped when Phase 7 lands, build stubbed CTAs and revisit the links during Phase 12 polish
- Backend chat sessions are in-memory and disappear on restart — Phase 6 must handle 404 on `session_id` by silently starting a new session and toasting the user (already encoded as CHAT-10)
- Scoring response uses camelCase `componentScores` while the rest of the API is snake_case — Phase 5 client types must accommodate the wart

### Phase 1 Plan Revisions (2026-04-28)

Plan-checker ran post-hoc (originally skipped due to subagent rate-limit). Revisions applied to the 4 plans before execution:

- **01-01**: dropped `postcss.config.js` + `@tailwindcss/postcss` + `autoprefixer` (Tailwind v4 Vite plugin handles processing alone — B-5); added `tsconfig.app.json` + `favicon.svg` to manifest (B-6, W-9); fixed `grain.svg` path to `public/` (W-1); added pre-hydration FOUC-prevention script to `index.html` keyed to `recruitai.theme` storage key (W-3).
- **01-02**: wired `QueryCache` + `MutationCache` `onError` → `toast.error` for non-validation `ApiError`s in `queryClient.ts` (B-4 — fulfills FOUND-07 / Phase 1 SC#4); added CORS allowlist verification step + toast smoke test to verification block (W-8).
- **01-03**: added `01-02` to `depends_on`, bumped to **wave 3** (B-2 — `main.tsx` imports `queryClient` from `@/api`, can't run parallel to 01-02).
- **01-04**: bumped to **wave 4** (B-2 cascade); moved Upload CTA to the BOTTOM of Sidebar per FOUND-10 / SC#2 (B-1 — planner's "PNG wins" override removed); added wordmark + route-derived breadcrumb to TopBar to satisfy SC#2 (B-3); replaced hardcoded `rgba(0,0,0,0.04)` hover/active backgrounds with `var(--hairline)` token in Sidebar + UserMenu + TopBar (W-4 — dark-mode parity); wrapped AppShell content in `mx-auto` so wide displays center (W-5).

Resulting wave map:
- Wave 1: 01-01
- Wave 2: 01-02
- Wave 3: 01-03
- Wave 4: 01-04

Deferred (not blockers): W-2 Geist family-name verification (smoke-test during execution), W-6 alpha-modifier theming pattern, W-7 utility class layer cleanup, W-10 ESLint flat-config react-refresh check, W-11 noUnusedParameters discipline for downstream phases, I-1..I-4 polish notes.

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-04-28
Stopped at: Phase 1, Plan 01-03 complete (React Router v7 routing skeleton — 18 routes, lazy chunks, RouterProvider). Next: execute 01-04 (AppShell layout: Sidebar + TopBar)
Resume file: None
