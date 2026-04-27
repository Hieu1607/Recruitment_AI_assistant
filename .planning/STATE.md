# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-27)

**Core value:** Recruiters can run their full hiring loop (upload → score → chat → shortlist → outreach → interview prep) through a polished editorial UI talking to the existing FastAPI backend.
**Current focus:** Phase 1 — Foundation & Design System

## Current Position

Phase: 1 of 12 (Foundation & Design System)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-04-27 — ROADMAP.md and STATE.md initialized

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: —
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| — | — | — | — |

**Recent Trend:**
- Last 5 plans: —
- Trend: — (no data yet)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table. Recent foundational picks:

- Phase 0 (init): Rebuild frontend from scratch in Vite + React 18 + TS + Tailwind v4 + shadcn/ui + TanStack Query + React Router v7 (do not touch `frontend_backup/`)
- Phase 0 (init): Forest-green `#1F3A2E` accent on off-white `#FAFAF7` confirmed from PNGs; Fraunces + Geist + Geist Mono as the type stack
- Phase 0 (init): Skip real auth enforcement — backend doesn't enforce yet; ship Login UI as visual surface only
- Phase 0 (init): Granularity "fine" — 12 phases, roughly one screen / tight bundle per phase, with parallelization enabled after Phases 1-2

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 7 (Candidate Detail) header CTAs link into Phases 5, 9, 10, 11 — if those phases haven't shipped when Phase 7 lands, build stubbed CTAs and revisit the links during Phase 12 polish
- Backend chat sessions are in-memory and disappear on restart — Phase 6 must handle 404 on `session_id` by silently starting a new session and toasting the user (already encoded as CHAT-10)
- Scoring response uses camelCase `componentScores` while the rest of the API is snake_case — Phase 5 client types must accommodate the wart

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-04-27
Stopped at: ROADMAP.md and STATE.md created; REQUIREMENTS.md traceability updated; ready to plan Phase 1
Resume file: None
