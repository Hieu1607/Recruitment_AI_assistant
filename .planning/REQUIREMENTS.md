# Requirements: Recruitment AI Assistant — Frontend

**Defined:** 2026-04-27
**Core Value:** Recruiters can run their full hiring loop (upload → score → chat → shortlist → outreach → interview prep) through a polished editorial-grade UI that talks to the existing FastAPI backend.

## v1 Requirements

### Foundation

- [ ] **FOUND-01**: Project scaffolds with Vite + React 18 + TypeScript (strict) and runs `npm run dev` on port 5173
- [ ] **FOUND-02**: Tailwind CSS v4 is configured with design tokens (colors, fonts, spacing, motion, shadows, radii) sourced from FRONTEND_SCREENS.md
- [ ] **FOUND-03**: Forest green accent `#1F3A2E`, off-white bg `#FAFAF7`, dark theme palette, hairline border token, and subtle grain background are applied as theme variables
- [ ] **FOUND-04**: Fraunces (display serif), Geist (sans), and Geist Mono are loaded and exposed as `font-display`, `font-sans`, `font-mono` utilities
- [ ] **FOUND-05**: Light mode and dark mode both render correctly across all screens, toggleable from the user menu, with persisted preference
- [ ] **FOUND-06**: API client (Axios + TanStack Query) is wired to `http://localhost:8000/api/v1` with TypeScript interfaces matching every response shape in BACKEND.md
- [ ] **FOUND-07**: API errors are normalized into a single shape and surfaced via toast for non-form errors and inline for form errors
- [ ] **FOUND-08**: React Router v7 routes are registered for all 15 screens with lazy loading
- [ ] **FOUND-09**: Authenticated layout shell renders TopBar (logo + breadcrumb + search + ⌘K + bell + avatar) + Sidebar (240px nav) + Content area on every authenticated route
- [ ] **FOUND-10**: Sidebar shows the 8 primary nav items in workflow order with Lucide icons, accent left bar on active item, and an "Upload resume" primary CTA pinned at the bottom

### Primitives (Shared Components)

- [ ] **PRIM-01**: Button component supports `primary | secondary | ghost | danger | icon` variants and `sm | md | lg` sizes with focus rings and loading state
- [ ] **PRIM-02**: Status badge component supports a dot indicator + label and renders correctly for `UploadStatus`, `ProfileStatus`, `MatchRunStatus`, `SentStatus` enum values
- [ ] **PRIM-03**: Data table component supports sortable columns, row hover, sticky header, bulk select, empty state, and skeleton loading
- [ ] **PRIM-04**: Modal/Dialog component centers a card with backdrop blur, traps focus, and closes on `Esc` and backdrop click
- [ ] **PRIM-05**: Toast notification component displays top-right, auto-dismisses after 4s, supports `success | error | info` variants
- [ ] **PRIM-06**: Empty state component renders editorial heading + minimal SVG illustration + CTA
- [ ] **PRIM-07**: Skeleton loader component animates with subtle shimmer (no spinners) for table rows, cards, and text
- [ ] **PRIM-08**: Avatar component renders deterministic initial-based fallback or photo, with consistent background palette per name
- [ ] **PRIM-09**: Tooltip component renders subtle hairline-bordered popover on hover/focus
- [ ] **PRIM-10**: Reusable score visualization renders as bar, donut, or radar at multiple sizes
- [ ] **PRIM-11**: Pagination component shows "Showing X-Y of Z entries" + Prev/Next + page-size selector (50/100/200)
- [ ] **PRIM-12**: Filter chip component supports single + multi-select with active-state styling

### Candidates (Resume Management)

- [ ] **CAND-01**: User can view a paginated list of all uploaded resumes via `GET /upload/` with status / uploader / date filters
- [ ] **CAND-02**: User can switch between table and card-grid views in the Candidates list, with state persisted in URL
- [ ] **CAND-03**: User can search the Candidates list by filename or candidate name (client-side filter on the loaded page)
- [ ] **CAND-04**: User can sort the Candidates list by upload date, status, or name
- [ ] **CAND-05**: User can bulk-select multiple resumes and trigger a bulk action (delete) from a sticky action bar
- [ ] **CAND-06**: User can open the Upload modal from the sidebar CTA or the Candidates page header
- [ ] **CAND-07**: User can drop or browse to select multiple PDFs in the Upload modal, with non-PDF files rejected client-side with inline error
- [ ] **CAND-08**: User can submit the Upload modal which calls `POST /upload/batch-parse` and shows an indeterminate progress bar with rotating editorial messages while processing
- [ ] **CAND-09**: User cannot close the Upload modal during synchronous processing; a warning explains why
- [ ] **CAND-10**: User sees a per-file success/failure summary after upload completes, with primary CTAs "View candidates" and "Upload more"
- [ ] **CAND-11**: User can open a Candidate Detail page that shows full name, current role, summary, key skills, experience timeline, and education
- [ ] **CAND-12**: User can view the embedded source PDF side-by-side with parsed data on the Candidate Detail page (PDF tab)
- [ ] **CAND-13**: User can see Scoring history, Outreach history, and Interview question sets for a candidate in dedicated tabs
- [ ] **CAND-14**: User can trigger "Score against JD", "Generate interview questions", "Draft outreach", and "Add to shortlist" from the Candidate Detail header
- [ ] **CAND-15**: User can edit a candidate's `original_file_name` or `upload_status` via `PATCH /upload/{resume_id}` from a row action
- [ ] **CAND-16**: User can delete a resume (with optional file deletion) via `DELETE /upload/{resume_id}` with confirmation

### Job Descriptions

- [ ] **JD-01**: User can view a paginated grid of job descriptions via `GET /job-descriptions/` with `is_active` filter
- [ ] **JD-02**: User can open the JD creation page and enter a title (large serif input, optional) and JD body (Notion-style editor with bold/italic/bullet/H2)
- [ ] **JD-03**: User can save a new JD via `POST /job-descriptions/`, with autosave-on-blur and an explicit Save button
- [ ] **JD-04**: User can edit an existing JD via `PATCH /job-descriptions/{id}` and toggle `is_active`
- [ ] **JD-05**: User can delete a JD via `DELETE /job-descriptions/{id}` with confirmation
- [ ] **JD-06**: User can hover a JD card to reveal "View" and "Score candidates" CTAs

### Scoring

- [ ] **SCORE-01**: User can launch the Scoring 3-step flow from a JD or from the global "Scoring" nav
- [ ] **SCORE-02**: In Step 1, user can pick a JD (with preview), choose "All candidates" or specific candidate set, and adjust section weights
- [ ] **SCORE-03**: User can edit per-section weights (skills/experience/projects/education/summary) with sliders + numeric input, and add additional sections (languages/achievements/etc.) via "+ Add section"
- [ ] **SCORE-04**: A live donut chart in Step 1 reflects the normalized weight distribution as the user edits
- [ ] **SCORE-05**: User can adjust the threshold slider (0-100, default 50) and batch size (1-50, default 10) in Step 1
- [ ] **SCORE-06**: User sees an estimated processing time before clicking "Start scoring"
- [ ] **SCORE-07**: Step 2 (Processing) shows an editorial-style animation, rotating status messages, and ETA — close button disabled
- [ ] **SCORE-08**: Step 3 (Results) shows summary stats: total candidates, passed threshold, average score, highest score (serif numerals)
- [ ] **SCORE-09**: User sees a sortable results table with rank, candidate name+avatar, total score (gradient color, tabular nums), passed badge, mini bar chart of component scores, rationale preview, and row actions
- [ ] **SCORE-10**: User can expand a result row to see full rationale, component score table, and a radar chart of component scores
- [ ] **SCORE-11**: User can bulk-select results and "Add N to shortlist", "Export", or "Draft outreach for N" from a sticky bar
- [ ] **SCORE-12**: User can see the `match_run_id` in the Results header (truncated UUID + copy button)

### AI Chat (Recruiter Chatbot)

- [ ] **CHAT-01**: User can view the list of past chat sessions in a left sidebar, ordered by most recently updated
- [ ] **CHAT-02**: User can click "+ New chat" to start a fresh session (sends first message without `session_id`)
- [ ] **CHAT-03**: User can search past sessions by title
- [ ] **CHAT-04**: User can rename a session inline and delete it from the chat header
- [ ] **CHAT-05**: User can send a message via Enter key or Send button; textarea auto-grows
- [ ] **CHAT-06**: AI messages render as full-width editorial prose; user messages render as right-aligned bubbles (max 70%)
- [ ] **CHAT-07**: When a chat response includes matching candidates, inline candidate cards (avatar, name, top 3 skills, "View" link) render below the message with a "Found N candidates" caption
- [ ] **CHAT-08**: User can copy any message to clipboard via a hover button
- [ ] **CHAT-09**: User can open settings to adjust `candidate_limit` (1-2000, default 500)
- [ ] **CHAT-10**: When a session_id returns 404 (backend restart), frontend silently starts a new session and shows a "Session expired, started new conversation" toast
- [ ] **CHAT-11**: Empty state offers 3-4 prompt suggestions as chips
- [ ] **CHAT-12**: User can view a session's message history via `GET /chat/{session_id}` when reopening it

### Shortlists (Sessions, Turns, Collections, Items)

- [ ] **SHORT-01**: User can view shortlist collections in a 3-column grid via `GET /shortlist/collections/?user_id=...`
- [ ] **SHORT-02**: Each collection card shows name, item count, created-at relative time, and source query indicator (if `source_query_turn_id` is set)
- [ ] **SHORT-03**: User can hover a collection card to reveal View / Rename / Delete actions
- [ ] **SHORT-04**: User can create a new collection via `POST /shortlist/collections/` from a header button or from a query turn
- [ ] **SHORT-05**: User can rename a collection via `PATCH /shortlist/collections/{id}` and sees a 409 conflict surfaced inline if the name duplicates
- [ ] **SHORT-06**: User can delete a collection via `DELETE /shortlist/collections/{id}` with confirmation
- [ ] **SHORT-07**: User can view a collection detail page that lists items via `GET /shortlist/collections/{id}/items` with pagination
- [ ] **SHORT-08**: User can remove a candidate from a collection via `DELETE /shortlist/collections/{id}/items/{candidate_id}` (handles 204 empty body)
- [ ] **SHORT-09**: User can switch to the "Query History" tab and see persisted `QuerySession`s with their turns timeline
- [ ] **SHORT-10**: User can create a collection directly from a query turn via `POST /shortlist/collections/` with `source_query_turn_id` set

### Outreach

- [ ] **OUT-01**: User can view outreach messages in a 3-column email-client layout: folders (All/Not sent/Sent/Failed) + message list + detail panel
- [ ] **OUT-02**: User can filter outreach messages by `created_by_user_id`, `candidate_profile_id`, or `sent_status`
- [ ] **OUT-03**: User can compose a new outreach message from the Candidate Detail page or a "+ New message" button, choosing AI-draft or template content source
- [ ] **OUT-04**: User can save a new outreach message via `POST /outreach/`
- [ ] **OUT-05**: User can edit subject/body or change `sent_status` via `PATCH /outreach/{id}`; setting status to `sent` auto-fills `sent_at` server-side
- [ ] **OUT-06**: User can delete an outreach message via `DELETE /outreach/{id}` (handles 204 empty body)

### Interview Questions

- [ ] **INTV-01**: User can view a list of interview question sets via `GET /interview-questions/` with filter by candidate / JD / creator
- [ ] **INTV-02**: User can open the Generate modal, pick a candidate and a JD, and create a new question set via `POST /interview-questions/`
- [ ] **INTV-03**: User can view a Question Set Detail page with questions grouped by category (technical/behavioral/culture-fit), each card showing question text, category and difficulty badges, and an editable notes textarea
- [ ] **INTV-04**: User can drag-reorder questions within or across categories
- [ ] **INTV-05**: User can edit, add, or delete individual questions, and persist the updated `question_payload` via `PATCH /interview-questions/{id}`
- [ ] **INTV-06**: User can export a question set as PDF or print it directly
- [ ] **INTV-07**: User can delete a question set via `DELETE /interview-questions/{id}`

### Dashboard

- [ ] **DASH-01**: User sees a serif greeting "Good morning/afternoon/evening, [Name]" plus today's date on the Dashboard
- [ ] **DASH-02**: Four metric cards display Total candidates, Resumes processed today, Active JDs, and Pending outreach with serif numerals and sparklines
- [ ] **DASH-03**: A 2/3-width Recent Activity feed lists resume uploads, score runs, chat sessions, outreach sent — newest first with relative timestamps
- [ ] **DASH-04**: A 1/3-width Quick Actions card offers Upload resumes / Create JD / Start scoring / Open chat primary buttons
- [ ] **DASH-05**: A "Top shortlist collections" row shows 3-4 most recent collections as cards with item count
- [ ] **DASH-06**: When data is empty, the Dashboard shows an onboarding checklist with 4 steps and progress checkmarks
- [ ] **DASH-07**: An Editorial Insight side card surfaces an LLM-generated headline observation about the candidate pool

### Marketing

- [ ] **MKTG-01**: Landing page renders the editorial hero (serif headline 72-96px, subhead, primary "Get started" + ghost "Watch demo")
- [ ] **MKTG-02**: Landing page shows a 4-column product value strip with line icons + tagline (Parse 500 CVs, AI scoring, Chat with pool, Generate questions)
- [ ] **MKTG-03**: Landing page features a product showcase with a tinted browser frame mockup
- [ ] **MKTG-04**: Landing page presents 3-4 alternating left/right feature deep-dive blocks with screenshots
- [ ] **MKTG-05**: Landing page includes a social proof logo bar with editorial typefaces and an italicized testimonial quote
- [ ] **MKTG-06**: Landing page closes with a big CTA + minimal editorial footer

### Authentication (UI only — backend does not enforce yet)

- [ ] **AUTH-01**: Login/Sign Up page uses a split-screen layout (60% editorial accent panel + 40% form panel)
- [ ] **AUTH-02**: User can enter email + password and click "Sign in" — backend call is stubbed with a clear "auth not yet enforced" UI hint
- [ ] **AUTH-03**: User can toggle between Sign in and Sign up modes from a single page
- [ ] **AUTH-04**: User sees inline validation errors on email format and required fields with subtle shake animation
- [ ] **AUTH-05**: SSO buttons (Google, Microsoft) render as outline-style placeholders with "coming soon" tooltip

### Platform Polish

- [ ] **PLAT-01**: ⌘K command palette opens from any screen, fuzzy-searches across candidates, JDs, collections, and core actions, and shows recent items
- [ ] **PLAT-02**: Settings page renders with left tabs: Profile, Workspace, API keys, Notifications, Danger zone
- [ ] **PLAT-03**: User menu (top-right avatar dropdown) provides Profile, Settings, theme toggle, Sign out
- [ ] **PLAT-04**: All dates render in user's local timezone with relative-time + tooltip for exact UTC value
- [ ] **PLAT-05**: All numeric values use `tabular-nums` so layouts don't jump on update
- [ ] **PLAT-06**: All UUIDs render truncated `#A7F2…3B91` with a copy-to-clipboard button
- [ ] **PLAT-07**: All sync-job UX (Upload + Scoring) blocks navigation away with confirmation
- [ ] **PLAT-08**: All HTTP errors map to inline UI: 400 = form validation, 404 = empty state, 409 = inline conflict, 422 = field errors, 500 = toast retry

## v2 Requirements

Deferred to a future release. Tracked but not in current roadmap.

### Authentication & Security
- **V2-AUTH-01**: Real JWT login enforced by backend with token refresh and 401 redirect
- **V2-AUTH-02**: SSO via Google + Microsoft fully wired
- **V2-AUTH-03**: Role-based UI gating (admin / recruiter / viewer)

### Realtime & Background Jobs
- **V2-RT-01**: Server-Sent Events or WebSocket stream for upload + scoring progress
- **V2-RT-02**: Background-task queue UI showing Celery workers and job retries
- **V2-RT-03**: Streaming chat responses (token-by-token AI output)

### Analytics & Insights
- **V2-ANL-01**: Analytics dashboard for recruiter funnel metrics
- **V2-ANL-02**: Candidate pool diversity / skills heatmap
- **V2-ANL-03**: Sentry / Datadog integration for production error tracking

### Outreach
- **V2-OUT-01**: Real SMTP / SendGrid integration to actually send emails
- **V2-OUT-02**: Email template library with variables
- **V2-OUT-03**: Reply detection and threading

### Mobile / I18n
- **V2-MOB-01**: Responsive optimization down to 375px viewport
- **V2-I18N-01**: Vietnamese UI (and other locales) for the bilingual recruiter audience

## Out of Scope

| Feature | Reason |
|---------|--------|
| Native mobile apps (iOS/Android) | Recruiters work on desktop; mobile-first not required |
| PDF editing / annotation | View-only is sufficient; complex authoring not in scope |
| Multi-tenant workspace switching logic | Single-tenant in v1; sidebar stub only |
| Real-time websocket updates | Request/response is sufficient for v1; deferred to v2 |
| Offline mode / service worker caching | Online-only product; no field-use case |
| WCAG AAA audit | Targeting AA, full AAA is out of v1 budget |
| Telemetry / Sentry / analytics | Deferred — must define data model first |
| Real email sending | Backend models the artifact; SMTP integration is v2 |
| New backend endpoints | Frontend works with what BACKEND.md documents; no backend changes |
| File types other than PDF | Backend rejects non-PDF; frontend mirrors that constraint |

## Traceability

Populated by `gsd-roadmapper` during ROADMAP.md creation (2026-04-27).

| Requirement | Phase | Status |
|-------------|-------|--------|
| FOUND-01 | Phase 1 | Pending |
| FOUND-02 | Phase 1 | Pending |
| FOUND-03 | Phase 1 | Pending |
| FOUND-04 | Phase 1 | Pending |
| FOUND-05 | Phase 1 | Pending |
| FOUND-06 | Phase 1 | Pending |
| FOUND-07 | Phase 1 | Pending |
| FOUND-08 | Phase 1 | Pending |
| FOUND-09 | Phase 1 | Pending |
| FOUND-10 | Phase 1 | Pending |
| PRIM-01 | Phase 2 | Pending |
| PRIM-02 | Phase 2 | Pending |
| PRIM-03 | Phase 2 | Pending |
| PRIM-04 | Phase 2 | Pending |
| PRIM-05 | Phase 2 | Pending |
| PRIM-06 | Phase 2 | Pending |
| PRIM-07 | Phase 2 | Pending |
| PRIM-08 | Phase 2 | Pending |
| PRIM-09 | Phase 2 | Pending |
| PRIM-10 | Phase 2 | Pending |
| PRIM-11 | Phase 2 | Pending |
| PRIM-12 | Phase 2 | Pending |
| CAND-01 | Phase 3 | Pending |
| CAND-02 | Phase 3 | Pending |
| CAND-03 | Phase 3 | Pending |
| CAND-04 | Phase 3 | Pending |
| CAND-05 | Phase 3 | Pending |
| CAND-06 | Phase 3 | Pending |
| CAND-07 | Phase 3 | Pending |
| CAND-08 | Phase 3 | Pending |
| CAND-09 | Phase 3 | Pending |
| CAND-10 | Phase 3 | Pending |
| CAND-15 | Phase 3 | Pending |
| CAND-16 | Phase 3 | Pending |
| JD-01 | Phase 4 | Pending |
| JD-02 | Phase 4 | Pending |
| JD-03 | Phase 4 | Pending |
| JD-04 | Phase 4 | Pending |
| JD-05 | Phase 4 | Pending |
| JD-06 | Phase 4 | Pending |
| SCORE-01 | Phase 5 | Pending |
| SCORE-02 | Phase 5 | Pending |
| SCORE-03 | Phase 5 | Pending |
| SCORE-04 | Phase 5 | Pending |
| SCORE-05 | Phase 5 | Pending |
| SCORE-06 | Phase 5 | Pending |
| SCORE-07 | Phase 5 | Pending |
| SCORE-08 | Phase 5 | Pending |
| SCORE-09 | Phase 5 | Pending |
| SCORE-10 | Phase 5 | Pending |
| SCORE-11 | Phase 5 | Pending |
| SCORE-12 | Phase 5 | Pending |
| CHAT-01 | Phase 6 | Pending |
| CHAT-02 | Phase 6 | Pending |
| CHAT-03 | Phase 6 | Pending |
| CHAT-04 | Phase 6 | Pending |
| CHAT-05 | Phase 6 | Pending |
| CHAT-06 | Phase 6 | Pending |
| CHAT-07 | Phase 6 | Pending |
| CHAT-08 | Phase 6 | Pending |
| CHAT-09 | Phase 6 | Pending |
| CHAT-10 | Phase 6 | Pending |
| CHAT-11 | Phase 6 | Pending |
| CHAT-12 | Phase 6 | Pending |
| CAND-11 | Phase 7 | Pending |
| CAND-12 | Phase 7 | Pending |
| CAND-13 | Phase 7 | Pending |
| CAND-14 | Phase 7 | Pending |
| DASH-01 | Phase 8 | Pending |
| DASH-02 | Phase 8 | Pending |
| DASH-03 | Phase 8 | Pending |
| DASH-04 | Phase 8 | Pending |
| DASH-05 | Phase 8 | Pending |
| DASH-06 | Phase 8 | Pending |
| DASH-07 | Phase 8 | Pending |
| SHORT-01 | Phase 9 | Pending |
| SHORT-02 | Phase 9 | Pending |
| SHORT-03 | Phase 9 | Pending |
| SHORT-04 | Phase 9 | Pending |
| SHORT-05 | Phase 9 | Pending |
| SHORT-06 | Phase 9 | Pending |
| SHORT-07 | Phase 9 | Pending |
| SHORT-08 | Phase 9 | Pending |
| SHORT-09 | Phase 9 | Pending |
| SHORT-10 | Phase 9 | Pending |
| OUT-01 | Phase 10 | Pending |
| OUT-02 | Phase 10 | Pending |
| OUT-03 | Phase 10 | Pending |
| OUT-04 | Phase 10 | Pending |
| OUT-05 | Phase 10 | Pending |
| OUT-06 | Phase 10 | Pending |
| INTV-01 | Phase 11 | Pending |
| INTV-02 | Phase 11 | Pending |
| INTV-03 | Phase 11 | Pending |
| INTV-04 | Phase 11 | Pending |
| INTV-05 | Phase 11 | Pending |
| INTV-06 | Phase 11 | Pending |
| INTV-07 | Phase 11 | Pending |
| MKTG-01 | Phase 12 | Pending |
| MKTG-02 | Phase 12 | Pending |
| MKTG-03 | Phase 12 | Pending |
| MKTG-04 | Phase 12 | Pending |
| MKTG-05 | Phase 12 | Pending |
| MKTG-06 | Phase 12 | Pending |
| AUTH-01 | Phase 12 | Pending |
| AUTH-02 | Phase 12 | Pending |
| AUTH-03 | Phase 12 | Pending |
| AUTH-04 | Phase 12 | Pending |
| AUTH-05 | Phase 12 | Pending |
| PLAT-01 | Phase 12 | Pending |
| PLAT-02 | Phase 12 | Pending |
| PLAT-03 | Phase 12 | Pending |
| PLAT-04 | Phase 12 | Pending |
| PLAT-05 | Phase 12 | Pending |
| PLAT-06 | Phase 12 | Pending |
| PLAT-07 | Phase 12 | Pending |
| PLAT-08 | Phase 12 | Pending |

**Coverage:**
- v1 requirements: 117 total (re-counted; original PROJECT.md task brief said 100, but FOUND(10) + PRIM(12) + CAND(16) + JD(6) + SCORE(12) + CHAT(12) + SHORT(10) + OUT(6) + INTV(7) + DASH(7) + MKTG(6) + AUTH(5) + PLAT(8) = 117)
- Mapped to phases: 117 ✓
- Unmapped: 0

---
*Requirements defined: 2026-04-27*
*Last updated: 2026-04-27 — traceability populated by `gsd-roadmapper`*
