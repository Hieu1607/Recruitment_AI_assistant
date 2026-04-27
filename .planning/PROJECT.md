# Recruitment AI Assistant — Frontend

## What This Is

An AI-powered recruitment platform for mid-to-senior recruiters. Backend (FastAPI + LangGraph + Postgres) is shipped — it ingests PDF resumes, parses them via LLM, scores candidates against job descriptions, and exposes a recruiter chat agent. **This milestone builds the production web frontend** that recruiters actually use: 15 screens consuming the documented `/api/v1/*` endpoints, with an editorial-meets-enterprise aesthetic (Linear/Notion/Arc/Monocle), not a generic SaaS template.

## Core Value

**Recruiters can run their full hiring loop — upload resumes → score against a JD → chat over the candidate pool → shortlist → outreach → interview prep — through a polished UI that feels trustworthy and premium, talking to the existing backend.**

If everything else slips, the **Scoring Results** screen (Screen 09c) and the **Candidates list + Upload** flow must work and look flagship-quality. They are the wow-factor + most-used surfaces.

## Requirements

### Validated

<!-- Backend capabilities already shipped and reachable from the frontend. -->

- ✓ Resume PDF batch upload + LLM parsing — `POST /api/v1/upload/batch-parse` (existing)
- ✓ Resume document CRUD with status tracking — `/api/v1/upload/*` (existing)
- ✓ Job Description CRUD — `/api/v1/job-descriptions/*` (existing)
- ✓ Candidate scoring against JD with section weights + threshold — `POST /api/v1/score/` (existing)
- ✓ Recruiter chatbot with session memory (LangGraph DSL+LLM router) — `/api/v1/chat/*` (existing)
- ✓ Query sessions + turns persistence — `/api/v1/shortlist/sessions/*` (existing)
- ✓ Shortlist collections + items — `/api/v1/shortlist/collections/*` (existing)
- ✓ Outreach message CRUD — `/api/v1/outreach/*` (existing)
- ✓ Interview question sets CRUD — `/api/v1/interview-questions/*` (existing)
- ✓ RBAC roles (admin/recruiter/viewer) — backend models exist (no frontend enforcement yet)

### Active

<!-- Frontend deliverables for this milestone. Hypotheses until shipped. -->

**Foundation & Design System**
- [ ] Vite + React 18 + TypeScript scaffold with strict mode
- [ ] Tailwind CSS v4 configured with design tokens from FRONTEND_SCREENS.md (forest green accent `#1F3A2E`, off-white bg `#FAFAF7`, Fraunces serif + Geist sans + Geist Mono, hairline borders, 8px grid, subtle grain)
- [ ] Light + dark mode with single source of truth tokens
- [ ] API client (Axios + TanStack Query) wired to `http://localhost:8000/api/v1` with TypeScript types matching BACKEND.md
- [ ] React Router v7 with route shell (TopBar + Sidebar layout from §1)
- [ ] Shared component library (Button, Badge, DataTable, Modal, Toast, Avatar, Tooltip, EmptyState, Skeleton — §3)

**Workflow Screens**
- [ ] Screen 03 — Dashboard with metric cards, activity feed, quick actions, editorial insight card
- [ ] Screen 04 — Candidates list (table + grid views, filters, pagination)
- [ ] Screen 05 — Upload Resumes modal with sync-job UX (rotating messages, progress, "don't close" warning)
- [ ] Screen 06 — Candidate Detail (overview/PDF/scoring/outreach/interview tabs)
- [ ] Screen 07 — Job Descriptions list (grid cards)
- [ ] Screen 08 — Create / Edit Job Description (Notion-style editor)
- [ ] Screen 09 — Scoring 3-step flow (Setup with weights donut → Processing → Results table with expand)
- [ ] Screen 10 — AI Chat (sessions sidebar + chat panel + inline candidate cards)
- [ ] Screen 11 — Shortlists (Collections + Query History tabs)
- [ ] Screen 12 — Collection Detail
- [ ] Screen 13 — Outreach Messages (3-column email-client layout)
- [ ] Screen 14 — Interview Questions (list + detail with grouped questions, drag reorder)

**Marketing & Auth**
- [ ] Screen 01 — Landing Page (hero, value strip, showcase, deep-dives, social proof, CTA)
- [ ] Screen 02 — Login / Sign Up (split-screen, UI-only since backend doesn't enforce auth yet)

**Platform Polish**
- [ ] Screen 15 — Settings (profile, workspace, API keys, notifications, danger zone)
- [ ] ⌘K Command Palette (global search across candidates / JDs / collections / actions)
- [ ] Dark/light mode toggle in user menu
- [ ] Responsive ≥1280px primary, graceful degrade to 1024px and tablet

### Out of Scope

- **Mobile-first or native mobile apps** — recruiters work on desktop; mobile is graceful degrade only
- **Real auth enforcement / token refresh / SSO wiring** — backend doesn't enforce auth yet; UI prepares the surface but no real flow
- **Real email sending from Outreach** — backend marks `sent_status` only; no SMTP integration this milestone
- **Embedded PDF editing or annotation** — view-only PDF in candidate detail
- **Multi-tenant workspace switching logic** — sidebar stub only, single tenant in v1
- **Internationalization (i18n)** — English-only UI strings for v1 (spec is bilingual but copy is English)
- **Real-time websocket updates** — all data flows are request/response; polling for processing status if needed
- **Offline mode / service worker caching** — online-only
- **Accessibility beyond WCAG 2.1 AA** — full AAA audit deferred
- **Analytics / telemetry / Sentry** — deferred to a later milestone

## Context

**Backend reality (from `docs/BACKEND.md`):**
- Base URL `http://localhost:8000`, all routes under `/api/v1`. Swagger at `/docs`.
- CORS allowlist includes `http://localhost:5173` (Vite dev server).
- **Auth:** JWT infrastructure exists but no endpoint enforces `Authorization: Bearer …` yet — frontend should prepare for it but not require it.
- **Sync gotchas:** `POST /upload/batch-parse` and `POST /score/` block until the LLM finishes (30s+ for big batches). UI must show editorial-style sync-job feedback, not generic spinners.
- **Camelcase wart:** scoring response `scores[]` uses camelCase keys (`candidateId`, `totalScore`, `componentScores`, `criterionKey`, `weightedScore`, `evidenceSummary`); the rest of the API is snake_case.
- **Chat sessions are in-memory** — they vanish on backend restart. Frontend must handle "session not found" by silently starting a new session and toasting the user.
- **List endpoints** all return `{ total, items[] }` and accept `limit` (1–200) + `offset`. Outreach `total` is real, others are page-size.
- **DELETE inconsistency:** older endpoints (upload, job-descriptions) return 200 with body; newer ones (shortlist, outreach, interview-questions) return 204 empty. Frontend handles both.

**Design reality (from `docs/FRONTEND_SCREENS.md` + `img/*.png`):**
- Editorial-meets-enterprise aesthetic: Fraunces or PP Editorial New for display, Geist or Söhne for body, JetBrains/Geist Mono for technical data.
- Single accent: **deep forest green ~`#1F3A2E`** (confirmed from PNGs). No purple gradients, no neon.
- Background: off-white `#FAFAF7`. Hairline borders `1px solid rgba(0,0,0,0.06)`. Subtle grain texture (2-3% opacity). 8px grid.
- Sidebar fixed 240–260px, content max-width 1280–1440px, top bar 56–64px.
- Motion: 240ms ease-out page transitions, staggered list reveals (50–80ms), skeleton shimmer not spinners.
- Status badges everywhere — must be consistent for `UploadStatus`, `ProfileStatus`, `MatchRunStatus`, `SentStatus`.

**Reference screens read from `img/`:** `Landing Page.png`, `Dashboard.png`, `Candidates Management.png`, `Scoring Results.png`, `AI Recruiter Chat.png` (5 frames).

**Prior frontend attempt** lives in `frontend_backup/` (do not touch — per CLAUDE.md). Rebuilding from scratch.

## Constraints

- **Tech stack**: Vite + React 18 + TypeScript + Tailwind v4. Picked for HMR speed, alignment with prior backup, and CSS-first design tokens that suit the editorial palette.
- **Component primitives**: shadcn/ui (Radix under the hood) — accessible, restyle-friendly. Avoid Material/Chakra (wrong aesthetic).
- **Server state**: TanStack Query — handles synchronous LLM jobs cleanly with `mutation.isPending` + retry semantics.
- **Routing**: React Router v7 (familiar, no need for file-based).
- **Charts**: Recharts (donut for weights, radar for component scores, mini bars for inline breakdown).
- **PDF viewer**: react-pdf for candidate detail.
- **DnD**: @dnd-kit for question reordering.
- **Backend port**: Frontend assumes `http://localhost:8000` for the backend. Vite dev server runs on `:5173` (already in CORS allowlist).
- **No new backend endpoints**: frontend works with what BACKEND.md documents. If a screen needs data the backend doesn't expose, scope it down or ship UI-only.
- **Performance**: Initial bundle target < 250KB gzipped (without PDF viewer chunk). Charts and PDF lazy-loaded.
- **Browser support**: Latest Chrome / Edge / Firefox / Safari (last 2 versions). No IE / legacy.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Vite + React 18 + TS | Matches frontend_backup stack, fast HMR, strong TS support, proven for editorial UIs | — Pending |
| Tailwind v4 over CSS Modules / styled-components | CSS-first tokens align with editorial design system; faster iteration; smaller bundle | — Pending |
| shadcn/ui as primitive base | Restyleable to editorial aesthetic, accessibility built-in via Radix, no runtime CSS-in-JS overhead | — Pending |
| TanStack Query over Redux/SWR | Handles 30s+ sync LLM jobs, cache invalidation, retry, optimistic updates out of the box | — Pending |
| Skip auth enforcement on frontend | Backend doesn't enforce it; building login UI but no real flow until backend protects routes | — Pending |
| Forest green `#1F3A2E` accent | Confirmed from PNGs; aligns with "trustworthy, professional, premium" brief | ✓ Good |
| Fraunces (serif) + Geist (sans) + Geist Mono | Editorial character, readable, all on Google Fonts | — Pending |
| Light mode primary, dark mode at parity | PNGs show light; spec asks for both with toggle in user menu | — Pending |
| Build screens in §5 priority order | User most-impacted surfaces first (Scoring → Candidates → Chat) | — Pending |
| Single Vite frontend (not Next.js) | No SSR needs, no SEO requirements for the auth'd app, landing can be static-prerendered later | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-27 after initialization*
