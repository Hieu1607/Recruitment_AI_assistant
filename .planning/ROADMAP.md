# Roadmap: Recruitment AI Assistant — Frontend

## Overview

Build the production frontend for the Recruitment AI Assistant in 12 phases, sequenced by impact and dependency. Phases 1 and 2 establish the foundation (scaffold, design tokens, layout shell, API client) and the shared component library. Phases 3-11 deliver feature surfaces in priority order — Candidates + Upload, Job Descriptions, the Scoring flagship, AI Chat, Candidate Detail, Dashboard, Shortlists, Outreach, and Interview Questions. Phase 12 closes with public marketing, auth UI, and platform polish (settings, command palette, dark-mode QA). After Phases 1-2 land, most feature phases (3-11) can run in parallel since they share primitives but not data.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation & Design System** - Vite scaffold, design tokens, theme, layout shell, API client, routing (complete 2026-04-28)
- [x] **Phase 2: Primitives Library** - Shared components (Button, Badge, DataTable, Modal, Toast, Avatar, Tooltip, EmptyState, Skeleton, Pagination, FilterChip, ScoreVisualization) (complete 2026-04-28)
- [x] **Phase 3: Candidates List & Upload Flow** - Screens 04 + 05; the most-used surface and PDF batch upload sync-job UX (complete 2026-04-29)
- [x] **Phase 4: Job Descriptions** - Screens 07 + 08; JD grid plus Notion-style editor (complete 2026-04-28)
- [x] **Phase 5: Scoring Flagship (3-step flow)** - Screen 09; Setup with weights donut, Processing animation, Results with expand (complete 2026-04-28)
- [x] **Phase 6: AI Recruiter Chat** - Screen 10; sessions sidebar, prose chat panel, inline candidate cards (complete 2026-04-28)
- [x] **Phase 7: Candidate Detail Hub** - Screen 06; tabbed profile linking into scoring, outreach, interview (complete 2026-04-28)
- [x] **Phase 8: Dashboard** - Screen 03; greeting, metric cards, activity feed, quick actions, editorial insight (complete 2026-04-28)
- [x] **Phase 9: Shortlists & Collection Detail** - Screens 11 + 12; collections, query history, items management (complete 2026-04-28)
- [ ] **Phase 10: Outreach Messages** - Screen 13; 3-column email-client layout, compose modal
- [ ] **Phase 11: Interview Questions** - Screen 14; generate, grouped detail with drag-reorder, export
- [ ] **Phase 12: Marketing, Auth & Platform Polish** - Screens 01 + 02 + 15, command palette, dark-mode QA, error mapping

## Phase Details

### Phase 1: Foundation & Design System
**Goal**: Recruiter-facing app shell renders with editorial design tokens, working API client, lazy-loaded routes for all 15 screens, and a togglable light/dark theme.
**Depends on**: Nothing (first phase)
**Requirements**: FOUND-01, FOUND-02, FOUND-03, FOUND-04, FOUND-05, FOUND-06, FOUND-07, FOUND-08, FOUND-09, FOUND-10
**Success Criteria** (what must be TRUE):
  1. User runs `npm run dev` and the Vite server boots on port 5173 with TypeScript strict mode and no errors
  2. User navigates to any of the 15 placeholder routes (e.g. `/candidates`, `/scoring`, `/chat`) and the authenticated shell renders with TopBar (logo, breadcrumb, search trigger, bell, avatar) and 240px Sidebar showing 8 nav items in workflow order plus the pinned "Upload resume" CTA
  3. User toggles dark mode from the user menu, every shell pixel switches palette, and the choice survives a page refresh
  4. User triggers a backend call (e.g. ping `GET /api/v1/upload/?limit=1`) through the typed Axios + TanStack Query client and receives data; a forced 500 surfaces a toast with the normalized error message
  5. Inspecting the rendered page confirms Fraunces is loaded as `font-display`, Geist as `font-sans`, Geist Mono as `font-mono`, the off-white `#FAFAF7` background with subtle grain is applied, and the active sidebar item shows the forest-green `#1F3A2E` left bar
**Plans**:
- [x] 01-01: Vite + React + TS scaffold, design tokens, Tailwind v4 theme (complete 2026-04-28, commits 1d353ba/0513acd/5b3068c)
- [x] 01-02: API client (Axios + TanStack Query) (complete 2026-04-28, commits f615020/6d2448e/54c284a)
- [x] 01-03: React Router v7 routing skeleton (complete 2026-04-28, commits ecdb3b3/c173af9/4659245)
- [x] 01-04: App shell (TopBar + Sidebar layout + theme store) (complete 2026-04-28, commits 3cfa0cc/6b6deb2/4fd7536/a35d834)

### Phase 2: Primitives Library
**Goal**: A reusable, restyleable component library (built on shadcn/ui + Radix) covering every primitive used across the 15 screens, documented in a Storybook-style demo route.
**Depends on**: Phase 1
**Requirements**: PRIM-01, PRIM-02, PRIM-03, PRIM-04, PRIM-05, PRIM-06, PRIM-07, PRIM-08, PRIM-09, PRIM-10, PRIM-11, PRIM-12
**Success Criteria** (what must be TRUE):
  1. User opens an internal `/dev/primitives` showcase route and sees every variant of Button (primary/secondary/ghost/danger/icon × sm/md/lg with focus rings and loading), Status Badge (one for each `UploadStatus`, `ProfileStatus`, `MatchRunStatus`, `SentStatus` value), Avatar (initial fallback + photo), and FilterChip (single + multi-select active states)
  2. User mounts a DataTable demo with 50 fake rows and can sort by clicking column headers, bulk-select via checkboxes, sees the sticky header on scroll, and triggers an empty state by clearing the dataset
  3. User opens a Modal that traps focus, closes on `Esc` and backdrop click, and dispatches a Toast that appears top-right and auto-dismisses after 4 seconds
  4. User loads a list page in degraded network and sees Skeleton shimmer rows (no spinners) plus a Pagination footer reading "Showing 1-50 of 234" with Prev/Next and a 50/100/200 page-size selector
  5. User views a ScoreVisualization in three sizes (mini bar inline, donut at 200px, radar at 400px) and a Tooltip renders on hover with hairline-bordered editorial styling
**Plans**:
- [x] 02-01: Button, Badge, Avatar, FilterChip (complete 2026-04-28)
- [x] 02-02: DataTable + Pagination (complete 2026-04-28)
- [x] 02-03: Modal/Dialog + Tooltip (complete 2026-04-28)
- [x] 02-04: EmptyState, Skeleton, ScoreVisualization (complete 2026-04-28)
- [x] 02-05: /dev/primitives showcase route (complete 2026-04-28)
**UI hint**: yes

### Phase 3: Candidates List & Upload Flow
**Goal**: Recruiter can browse, filter, search, sort, and bulk-manage every uploaded resume, and run the synchronous PDF batch-upload flow with editorial sync-job UX.
**Depends on**: Phase 1, Phase 2
**Requirements**: CAND-01, CAND-02, CAND-03, CAND-04, CAND-05, CAND-06, CAND-07, CAND-08, CAND-09, CAND-10, CAND-15, CAND-16
**Success Criteria** (what must be TRUE):
  1. User opens `/candidates`, sees the paginated table populated from `GET /upload/`, applies the `processing` status filter, and the URL updates so the filter survives a refresh
  2. User toggles between table and card-grid views, the URL persists the choice, and sorting by upload date or status reorders rows with tabular-nums score columns staying aligned
  3. User bulk-selects three rows, the sticky action bar appears with a count, and clicking Delete fires `DELETE /upload/{id}` per row with a confirmation modal
  4. User opens the Upload modal, drag-drops two PDFs plus one `.docx` (which is rejected inline with a danger-colored error), submits the remaining PDFs, and watches an indeterminate progress bar with rotating editorial messages ("Reading resumes...", "Extracting skills...", "Building profiles...") while the close button is disabled and a "don't close" warning is visible
  5. After `POST /upload/batch-parse` completes, the modal shows a per-file success/failure summary with green checks and red error reasons, and the "View candidates" CTA returns the recruiter to the table with the new rows visible
**Plans**:
- [x] 03-01: Candidates List page (Screen 04) — table/grid views, URL-persisted filters/sort/search/pagination, bulk-delete, row edit/delete (complete 2026-04-29)
- [x] 03-02: Upload Resumes modal (Screen 05) — dropzone, PDF-only validation, indeterminate progress + rotating messages, per-file result summary (complete 2026-04-29)
**UI hint**: yes

### Phase 4: Job Descriptions
**Goal**: Recruiter can browse, create, edit, activate/deactivate, and delete job descriptions through a Notion-style editor and a hover-revealing card grid.
**Depends on**: Phase 1, Phase 2
**Requirements**: JD-01, JD-02, JD-03, JD-04, JD-05, JD-06
**Success Criteria** (what must be TRUE):
  1. User opens `/job-descriptions`, sees the 2-3 column grid populated from `GET /job-descriptions/`, toggles the `is_active` filter, and the active-only set re-renders without a full reload
  2. User hovers a JD card and the "View" and "Score candidates" CTAs reveal with a soft lift shadow; clicking "Score candidates" routes to the Scoring Setup with this JD pre-selected
  3. User clicks "Create JD", lands on the full-page editor with a large serif title input ("Untitled position" placeholder) and a Notion-style body editor supporting bold/italic/bullet/H2 toolbar
  4. User types a body, blurs the editor, and sees an autosave indicator ("Saved 2s ago"); clicking explicit Save fires `POST /job-descriptions/` and routes back to the grid with the new JD visible
  5. User edits an existing JD, toggles `is_active` off in the right-side settings panel (which fires `PATCH`), and deletes another JD via a confirmation modal that calls `DELETE /job-descriptions/{id}`
**Plans**:
- [x] 04-01: Job Descriptions list (Screen 07) — grid cards, is_active filter chips, hover-reveal View/Score CTAs, delete modal (complete 2026-04-28)
- [x] 04-02: JD editor (Screen 08) — serif title input, contentEditable body with Bold/Italic/H2/Bullet toolbar, autosave on blur, is_active toggle in right-side settings panel, explicit Save, delete modal (complete 2026-04-28)
**UI hint**: yes

### Phase 5: Scoring Flagship (3-step flow)
**Goal**: Recruiter can configure a scoring run with editable section weights, watch an editorial processing animation during the synchronous LLM job, and explore the results table with expandable rationales and component score visualizations. This is the wow-factor surface and gets extra design budget.
**Depends on**: Phase 1, Phase 2, Phase 4
**Requirements**: SCORE-01, SCORE-02, SCORE-03, SCORE-04, SCORE-05, SCORE-06, SCORE-07, SCORE-08, SCORE-09, SCORE-10, SCORE-11, SCORE-12
**Success Criteria** (what must be TRUE):
  1. User launches Scoring from the sidebar nav OR from a JD's "Score candidates" hover CTA; Step 1 loads with a stepper showing 1/3, JD selector with body preview, and a candidate selector toggling "All candidates" vs a multi-select with search
  2. User drags the Skills slider from 30 to 50, types Experience to 25, adds a custom "Languages" section via "+ Add section", and the live donut chart re-balances all sections to a normalized 100% total
  3. User adjusts the threshold slider to 60 and batch size to 20, sees the estimated time update ("~2 min for 50 candidates"), clicks "Start scoring" and Step 2 takes over with an editorial SVG animation, rotating status messages, ETA, progress bar, and a disabled close button
  4. After `POST /score/` completes, Step 3 renders the summary strip (Total/Passed/Average/Highest in serif numerals), the truncated `match_run_id` `#A7F2…3B91` with a copy button, and a sortable table with rank, candidate avatar+name, gradient-colored total score (tabular-nums), passed badge, mini bar chart of `componentScores`, and a 1-line rationale preview
  5. User expands a row to see full rationale prose, the component score table (criterion / weight / score / weighted score / italic evidence quote), and a radar chart; bulk-selects 5 rows and the sticky bar offers "Add 5 to shortlist", "Export", "Draft outreach for 5"
**Plans**:
- [x] 05-01: Step 1 Setup — JD dropdown with body preview, All/Specific candidate selector with search, weight sliders + live donut chart, threshold + batch size sliders, estimated time, Start button (complete 2026-04-28)
- [x] 05-02: Step 2 Processing — editorial pulsing animation, rotating status messages, indeterminate progress bar, elapsed timer, disabled Cancel (complete 2026-04-28)
- [x] 05-03: Step 3 Results — 4-card summary strip, match_run_id copy, sortable table with expand rows (full rationale + component score table + radar chart), bulk action sticky bar (complete 2026-04-28)
**UI hint**: yes

### Phase 6: AI Recruiter Chat
**Goal**: Recruiter can have multi-turn conversations with the candidate-pool chatbot, see inline matched-candidate cards rendered from chat responses, manage past sessions, and recover gracefully when in-memory sessions expire.
**Depends on**: Phase 1, Phase 2
**Requirements**: CHAT-01, CHAT-02, CHAT-03, CHAT-04, CHAT-05, CHAT-06, CHAT-07, CHAT-08, CHAT-09, CHAT-10, CHAT-11, CHAT-12
**Success Criteria** (what must be TRUE):
  1. User opens `/chat`, the empty state shows the editorial hero "Ask anything about your candidates" plus 3-4 prompt suggestion chips, and clicking "+ New chat" populates the input ready to send
  2. User types "Who has 5+ years of Python?" and presses Enter; the textarea resets, the user message appears right-aligned (max 70%), and the AI response renders left-aligned as full-width prose with no bubble
  3. When the response includes matched candidates, inline candidate cards render below the message in a horizontal scroll/grid (avatar + name + top 3 skills + "View" link) with a "Found N candidates" caption
  4. User reopens a past session from the sidebar, message history loads via `GET /chat/{session_id}`, and renaming the session inline persists; deleting the session removes it from the sidebar
  5. After a backend restart triggers a 404 on `session_id`, the frontend silently starts a fresh session, shows a toast "Session expired, started new conversation", and the user's next message succeeds
**Plans**:
- [x] 06-01: Chat route — sessions sidebar (localStorage persistence), prose chat panel, inline candidate cards, session expiry recovery (complete 2026-04-28)
**UI hint**: yes

### Phase 7: Candidate Detail Hub
**Goal**: Recruiter can drill into any candidate to see parsed profile, source PDF, scoring history, outreach history, and interview question sets, and trigger cross-feature actions (score, generate questions, draft outreach, add to shortlist) from the header.
**Depends on**: Phase 1, Phase 2, Phase 3 (links from Candidates list); soft links to Phase 5, Phase 9, Phase 10, Phase 11 (header CTAs may stub if those phases ship later)
**Requirements**: CAND-11, CAND-12, CAND-13, CAND-14
**Success Criteria** (what must be TRUE):
  1. User clicks a candidate name from `/candidates` and the detail page header renders the full name in 36-48px serif, current role + years subtitle, and the four header action buttons (Score against JD, Generate interview questions, Draft outreach, Add to shortlist)
  2. On the Overview tab the user reads the summary, key skills as hairline-bordered chips, a vertical experience timeline with dots, and the education list
  3. User clicks the Resume PDF tab and the embedded react-pdf viewer renders the source PDF on the left while parsed data renders on the right for verification
  4. User opens the Scoring history, Outreach history, and Interview questions tabs and sees populated lists tied to this `candidate_profile_id` (each list links into its respective feature surface)
  5. User clicks "Add to shortlist" in the header, picks an existing collection or creates a new one inline, and a success toast confirms the candidate is added
**Plans**: TBD
**UI hint**: yes

### Phase 8: Dashboard
**Goal**: First-impression-after-login screen presents pool health at a glance — greeting, four metric cards with sparklines, recent activity feed, quick actions, top collections, and an editorial AI insight card.
**Depends on**: Phase 1, Phase 2
**Requirements**: DASH-01, DASH-02, DASH-03, DASH-04, DASH-05, DASH-06, DASH-07
**Success Criteria** (what must be TRUE):
  1. User lands on `/` after login and sees a serif greeting that adapts to time-of-day ("Good morning, Hieu"), today's date below it, and four metric cards (Total candidates, Resumes processed today, Active JDs, Pending outreach) each with serif tabular-nums numerals and a sparkline plus % change vs prior period
  2. User scrolls and the 2/3-width Recent Activity feed lists the most recent resume uploads, score runs, chat sessions, and outreach sends — newest first with relative timestamps and an icon per activity type
  3. User clicks any of the four Quick Action buttons in the 1/3 right card (Upload resumes / Create JD / Start scoring / Open chat) and the appropriate modal opens or route loads
  4. The Top shortlist collections row shows 3-4 most recent collections as cards with item counts, and the Editorial Insight side card surfaces an LLM-style headline observation about the pool
  5. With no data present (empty backend), the Dashboard switches to an onboarding checklist of 4 steps with progress checkmarks instead of empty metric cards
**Plans**:
- [x] 08-01: Dashboard route — greeting, 4 metric cards with sparklines + % change, activity feed, quick actions, editorial insight, onboarding checklist, top collections (complete 2026-04-28)
**UI hint**: yes

### Phase 9: Shortlists & Collection Detail
**Goal**: Recruiter can manage saved candidate collections plus the persisted query-session history, and drill into any collection to add/remove members or take collection-wide actions.
**Depends on**: Phase 1, Phase 2; soft link to Phase 6 (Query History tab references chat sessions)
**Requirements**: SHORT-01, SHORT-02, SHORT-03, SHORT-04, SHORT-05, SHORT-06, SHORT-07, SHORT-08, SHORT-09, SHORT-10
**Success Criteria** (what must be TRUE):
  1. User opens `/shortlists`, the Collections tab shows a 3-column grid populated from `GET /shortlist/collections/`, each card displays name (serif), item-count badge, relative created-at, and a "from query" indicator with tooltip when `source_query_turn_id` is set
  2. User hovers a collection card and "View / Rename / Delete" actions reveal; renaming via `PATCH` to a duplicate name surfaces a 409 conflict inline (not a toast) with a helpful message
  3. User clicks a collection and the detail page lists members with avatar + name + top skills + latest match score + added-at, with pagination, and a "Remove from collection" row action calls `DELETE /shortlist/collections/{id}/items/{candidate_id}` (handling the 204 empty body)
  4. User switches to the Query History tab, picks a session from the left list, and the right panel renders a vertical timeline of turns — each turn showing the user question (serif italic), AI answer (editorial prose), matched_count badge, and a "Show matched candidates" toggle
  5. User clicks "Create collection from this turn" on a turn, names the collection, and `POST /shortlist/collections/` is called with `source_query_turn_id` set; the new collection appears in the Collections tab with the source-query indicator
**Plans**: TBD
**UI hint**: yes

### Phase 10: Outreach Messages
**Goal**: Recruiter can manage outreach drafts and sent messages in an email-client style 3-column layout, and compose new messages (AI draft or template) tied to specific candidates.
**Depends on**: Phase 1, Phase 2; soft link to Phase 7 (compose can launch from Candidate Detail header)
**Requirements**: OUT-01, OUT-02, OUT-03, OUT-04, OUT-05, OUT-06
**Success Criteria** (what must be TRUE):
  1. User opens `/outreach` and sees the 3-column layout: left folders (All / Not sent / Sent / Failed each with counts), middle message list (candidate name bold + subject + body preview + status badge + relative timestamp), and right detail panel
  2. User filters the list by `sent_status=sent` and by a specific `candidate_profile_id`; the URL persists both filters and the list updates without a full reload
  3. User clicks "+ New message", picks a candidate, toggles content source AI-draft vs Template, types subject + body, clicks Save and `POST /outreach/` succeeds with the new message appearing in the Not sent folder
  4. User opens an existing message in detail, edits the body, clicks "Mark as sent" and `PATCH /outreach/{id}` sets `sent_status=sent` with the server filling `sent_at`; the badge flips to green and the message moves to the Sent folder
  5. User deletes a message via the detail panel action; `DELETE /outreach/{id}` returns 204 (handled), the message disappears from the list, and a toast confirms the action
**Plans**: 2 plans
- [ ] 10-01-PLAN.md — 3-column shell, folder sidebar, message list, URL persistence (OUT-01, OUT-02)
- [ ] 10-02-PLAN.md — detail panel, compose modal, edit/mark-sent/delete mutations (OUT-03, OUT-04, OUT-05, OUT-06)
**UI hint**: yes

### Phase 11: Interview Questions
**Goal**: Recruiter can generate, view, edit, drag-reorder, and export interview question sets grouped by category, ready to print and bring to interviews.
**Depends on**: Phase 1, Phase 2; soft link to Phase 7 (Generate modal can launch from Candidate Detail)
**Requirements**: INTV-01, INTV-02, INTV-03, INTV-04, INTV-05, INTV-06, INTV-07
**Success Criteria** (what must be TRUE):
  1. User opens `/interview-questions`, sees the list populated from `GET /interview-questions/`, filters by candidate, JD, and creator, and each item shows candidate + JD title + created-at + derived question count
  2. User clicks "Generate new set", picks a candidate and JD in the modal, clicks Generate, and `POST /interview-questions/` creates a set; the new set appears in the list and routing into it loads the detail page
  3. On the detail page the user sees the serif header "Interview for [Candidate] — [JD Title]", questions grouped by category (Technical / Behavioral / Culture-fit), each card with serif quote-style question text, category + difficulty badges, and an editable notes textarea
  4. User drags a question from Technical into Behavioral via @dnd-kit, edits its text inline, adds a new question via "+ Add question", and clicking Save persists the updated `question_payload` via `PATCH /interview-questions/{id}`
  5. User clicks "Export as PDF" (or Print) and a print-optimized layout renders all questions cleanly without sidebar/topbar chrome; "Delete set" calls `DELETE /interview-questions/{id}` with confirmation
**Plans**: TBD
**UI hint**: yes

### Phase 12: Marketing, Auth & Platform Polish
**Goal**: Public marketing surface, Auth UI (UI-only since backend doesn't enforce), Settings page, ⌘K command palette, comprehensive HTTP error handling, and a final dark-mode QA sweep — closing v1.
**Depends on**: Phase 1, Phase 2; cross-cuts every feature phase (3-11) for the polish/error sweep
**Requirements**: MKTG-01, MKTG-02, MKTG-03, MKTG-04, MKTG-05, MKTG-06, AUTH-01, AUTH-02, AUTH-03, AUTH-04, AUTH-05, PLAT-01, PLAT-02, PLAT-03, PLAT-04, PLAT-05, PLAT-06, PLAT-07, PLAT-08
**Success Criteria** (what must be TRUE):
  1. User visits `/` while logged out and sees the landing page: 72-96px serif hero ("Hire like it's 2030."), 4-column value strip with line icons, browser-frame product showcase, 3-4 alternating left/right feature deep-dives with screenshots, social-proof logo bar with italicized testimonial, big-CTA closer, and editorial footer
  2. User clicks "Get started", lands on the split-screen Auth page (60% editorial accent panel + 40% form), toggles between Sign in / Sign up, types an invalid email and the field shake-animates with inline validation, and clicking "Sign in" shows a clear "auth not yet enforced" UI hint
  3. User presses ⌘K from any authenticated screen, the command palette opens with fuzzy search across candidates, JDs, collections, and core actions plus a recent-items list; selecting an item navigates or fires the action
  4. User opens Settings via the avatar dropdown, navigates the left tabs (Profile, Workspace, API keys, Notifications, Danger zone), and the avatar dropdown also exposes Profile / Settings / theme toggle / Sign out
  5. User triggers each HTTP error class — 400 (form validation inline), 404 (empty state), 409 (inline conflict text), 422 (field-level errors), 500 (toast with retry) — and confirms the UI maps correctly; user verifies all 15 screens look correct in both light and dark themes with tabular-nums for numerics, truncated UUIDs with copy, and local-timezone relative timestamps with UTC tooltips
**Plans**: TBD
**UI hint**: yes

## Progress

**Execution Order:**
Phases execute in numeric order. Phases 1-2 are strictly sequential (foundation must land before primitives, primitives before features). Phases 3-11 may be parallelized after Phase 2 ships (config has `parallelization: true`); Phase 7 (Candidate Detail) ideally lands after Phases 3, 5, 9, 10, 11 to wire its header CTAs, but a stubbed version can ship earlier and links revisited. Phase 12 closes last because the polish sweep cross-cuts every feature.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation & Design System | 4/4 | Complete | 2026-04-28 |
| 2. Primitives Library | 5/5 | Complete | 2026-04-28 |
| 3. Candidates List & Upload Flow | 0/TBD | Not started | - |
| 4. Job Descriptions | 2/2 | Complete | 2026-04-28 |
| 5. Scoring Flagship (3-step flow) | 3/3 | Complete | 2026-04-28 |
| 6. AI Recruiter Chat | 1/1 | Complete | 2026-04-28 |
| 7. Candidate Detail Hub | 1/1 | Complete | 2026-04-28 |
| 8. Dashboard | 1/1 | Complete | 2026-04-28 |
| 9. Shortlists & Collection Detail | 2/2 | Complete | 2026-04-28 |
| 10. Outreach Messages | 0/2 | Planned | - |
| 11. Interview Questions | 0/TBD | Not started | - |
| 12. Marketing, Auth & Platform Polish | 0/TBD | Not started | - |
