# Session Kickoff — Recruitment AI Assistant Frontend

**Paste the block below at the start of any new Claude Code session in this project.** It gives Claude the full context it needs to resume work without re-asking what we're building.

---

## Kickoff prompt (copy-paste this)

```
You're working on the Recruitment AI Assistant frontend milestone.

CONTEXT (read in order, do not skip):
1. Read CLAUDE.md — project instructions, backend gotchas, GSD workflow rules.
2. Read .planning/STATE.md — current phase + blockers + recent work.
3. Read .planning/PROJECT.md — vision, locked decisions, validated capabilities (backend), active scope (frontend).
4. Read .planning/ROADMAP.md — 12-phase breakdown; check which phase is "in progress".
5. Read docs/BACKEND.md — every endpoint we consume, types, gotchas (camelCase wart in `scores[]`, sync 30s+ jobs, in-memory chat sessions).
6. Read docs/FRONTEND_SCREENS.md — 15-screen design spec, editorial aesthetic brief, shared components.
7. Glance at img/*.png — 5 reference frames (Landing, Dashboard, Candidates, Scoring Results, AI Chat).

WHAT WE'RE BUILDING:
A 15-screen editorial-grade recruiter web app that talks to the existing FastAPI backend at http://localhost:8000/api/v1. The backend is done and validated. The frontend is greenfield — frontend/ is empty. frontend_backup/ is reference-only; do not touch it.

STACK (locked in PROJECT.md):
- Vite + React 18 + TypeScript (strict)
- Tailwind v4 (CSS-first tokens)
- shadcn/ui primitives (Radix, install on demand)
- TanStack Query + Axios for API
- React Router v7 with route-level lazy loading
- Zustand for client state (theme, ⌘K palette)
- Recharts for donut/radar/bars in Scoring
- react-pdf for embedded PDF in Candidate Detail
- @dnd-kit for question reordering
- Sonner for toasts
- Fonts: Fraunces (display), Geist (sans), Geist Mono — via @fontsource

DESIGN TOKENS (locked):
- Accent: forest green #1F3A2E (confirmed from img/*.png)
- Light bg: off-white #FAFAF7 with subtle grain at 2-3% opacity
- Dark bg: #0F1012
- Hairline borders: rgba(0,0,0,0.06) light / rgba(255,255,255,0.08) dark
- 8px grid, sidebar 240px, content max 1440px, top bar 60px

GSD WORKFLOW:
This project uses GSD for structured agentic development. Planning artifacts live in .planning/. Phase plans live in .planning/phases/NN-slug/NN-MM-PLAN.md. Workflow is YOLO + fine granularity + all quality agents on. To advance:
- /gsd-progress — see where we are
- /gsd-execute-phase N — run all plans for a phase in waves
- /gsd-plan-phase N — plan a phase before executing
- /gsd-resume-work — pick up after /clear

CURRENT POSITION (as of last session):
Phase 1 (Foundation & Design System) is PLANNED — 4 PLAN.md files exist in
.planning/phases/01-foundation-design-system/:
- 01-01: Vite scaffold + Tailwind tokens + fonts (Wave 1)
- 01-02: Typed API client + ApiError + endpoint modules (Wave 2)
- 01-03: Routing skeleton with 18 routes + RoutePlaceholder (Wave 2)
- 01-04: AppShell + TopBar + Sidebar + theme toggle (Wave 3)

Plan-checker subagent was not run (rate limit at end of last session) — eyeball plans or run /gsd-review --phase 1 before /gsd-execute-phase 1.

BACKEND GOTCHAS (must respect in every plan):
- POST /upload/batch-parse and POST /score/ are SYNCHRONOUS — block 30s+ for LLM. UI must show editorial sync-job feedback (rotating messages, ETA, no spinners), disable close.
- scores[] response uses camelCase (candidateId, totalScore, componentScores, criterionKey, weightedScore, evidenceSummary). Rest of API is snake_case.
- Chat sessions are in-memory; vanish on backend restart. On 404 from chat endpoints, silently start a new session and toast "Session expired".
- Auth is NOT enforced by backend yet. Build login UI but no real flow.
- DELETE inconsistency: older endpoints (upload, job-descriptions) return 200 with body; newer ones (shortlist, outreach, interview-questions) return 204 empty.
- CORS allowlist must include http://localhost:5173 — backend env var BACKEND_CORS_ORIGINS.

CONSTRAINTS:
- No new backend endpoints. Frontend works with what BACKEND.md documents.
- Don't touch frontend_backup/.
- Don't add features beyond what the active phase requires.
- Commit per phase milestone, not per file (gsd-tools commit handles atomicity).

NEXT STEP:
Run /gsd-execute-phase 1 to build the foundation, OR ask me what I want first if anything has changed since the last session.
```

---

## Why this exists

GSD's `/gsd-resume-work` and `/gsd-progress` already do most of the lifting, but they don't surface the **stack decisions, design tokens, and backend gotchas** that are critical to making the right choices when planning new phases. Pasting this block gets new Claude sessions to a working baseline in ~15 seconds of context-loading instead of 5+ minutes of file-walking.

## When to update this

- After Phase 1 ships → update "CURRENT POSITION" to reflect what's been built
- When PROJECT.md decisions change → mirror them in the STACK / DESIGN TOKENS sections
- When backend changes (new endpoints, shape changes) → mirror in BACKEND GOTCHAS

## Quick alternatives

- **Just resume mid-task:** type `/gsd-resume-work` after pasting the block above — GSD reads STATE.md and tells you the next concrete action.
- **Quick status check:** type `/gsd-progress` for the milestone bar and current focus.
- **Lost?** Type `/gsd-help` for the GSD command reference.
