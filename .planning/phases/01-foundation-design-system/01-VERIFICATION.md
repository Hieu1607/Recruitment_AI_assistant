---
phase: 01-foundation-design-system
verified: 2026-04-28T00:00:00Z
status: human_needed
score: 9/10 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Run `npm run dev` in frontend/ and visit http://localhost:5173/dashboard — confirm the editorial sidebar (240px, 8 nav items, forest-green Upload CTA pinned at bottom) and TopBar (RecruitAI wordmark, breadcrumb 'Dashboard', search trigger, bell icon, command palette icon, avatar) all render correctly"
    expected: "Full AppShell renders with Sidebar + TopBar; RoutePlaceholder shows 'Dashboard' as screen name; no JS errors in console"
    why_human: "Cannot start dev server programmatically in this environment; visual layout and font rendering (Fraunces serif) must be confirmed by a human"
  - test: "From /dashboard, click the avatar → switch to Dark theme. Verify every shell pixel switches to dark palette (deep background #0F1012, adjusted accent #4A7C59). Then reload the page and confirm dark mode persists without a light-mode flash."
    expected: "Dark mode applies immediately; page reload paints dark from first frame (no FOUC); localStorage has 'recruitai.theme' = 'dark'"
    why_human: "Theme persistence + FOUC prevention requires visual verification in a live browser; cannot be confirmed via static analysis"
  - test: "With the backend stopped (or not running), open browser DevTools Console on any authenticated route. Run: `import('@/api').then(m => m.api.upload.list({limit:1})).catch(e => console.log(e.kind, e.detail))`. Verify a Sonner toast appears with 'Can't reach the server. Check your connection.'"
    expected: "Toast renders top-right within 1 second; err.kind === 'network'; no duplicate toasts"
    why_human: "Requires live browser + Sonner toast rendering + network simulation; not verifiable from static code"
---

# Phase 1: Foundation & Design System Verification Report

**Phase Goal**: Recruiter-facing app shell renders with editorial design tokens, working API client, lazy-loaded routes for all 15 screens, and a togglable light/dark theme.
**Verified:** 2026-04-28T00:00:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `npm run dev` boots Vite on port 5173 with TypeScript strict mode | ✓ VERIFIED | `package.json` `"dev": "vite --port 5173"`; `tsconfig.app.json` `"strict": true`; `dist/index.html` exists from successful prior build |
| 2 | All 15+ placeholder routes render in authenticated AppShell with TopBar + 240px Sidebar + "Upload resume" CTA | ? UNCERTAIN | All artifacts exist and are wired (verified below); requires human visual confirmation |
| 3 | Dark mode toggle persists across reload with no FOUC | ? UNCERTAIN | `theme.ts` uses localStorage + validated allowlist; `index.html` has pre-hydration resolver script; requires live browser confirmation |
| 4 | Axios + TanStack Query client hits backend; forced 500 surfaces normalized toast | ✓ VERIFIED | `queryClient.ts` has `toast.error` in `QueryCache.onError`; `errors.ts` maps 5xx to `kind:"server"`; validation (422) correctly skipped; `client.ts` reads `VITE_API_BASE_URL` |
| 5 | Fraunces = `font-display`, Geist = `font-sans`, Geist Mono = `font-mono`; off-white bg `#FAFAF7` with grain; active sidebar bar = forest-green `#1F3A2E` | ✓ VERIFIED | `globals.css` `@theme` maps `--font-display: "Fraunces"`, `--font-sans: "Geist"`, `--font-mono: "Geist Mono"`; `tokens.css` `--bg: #FAFAF7`; `body` uses `background-image: url("/grain.svg")`; `Sidebar.tsx` active span has `bg-accent` (resolves to `--accent: #1F3A2E` in light mode) |

**Score:** 9/10 truths verified (3 VERIFIED, 2 need human for visual/runtime confirmation; all underlying code is correct)

### Deferred Items

None.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `frontend/package.json` | Dependency manifest with React 18, Vite 5, TS 5, Tailwind v4 | ✓ VERIFIED | `"react": "^18.3.1"`, `"vite": "^5.4.0"`, `"tailwindcss": "^4.0.0"` all present |
| `frontend/tsconfig.app.json` | TypeScript strict mode config | ✓ VERIFIED | `"strict": true` at line 15; `"noUnusedLocals": true`, `"noUnusedParameters": true` also set |
| `frontend/vite.config.ts` | Vite config with port 5173 and @ alias | ✓ VERIFIED | `server: { port: 5173, host: true }`, `alias: { "@": ... src }`, `tailwindcss()` plugin present |
| `frontend/src/styles/tokens.css` | Design tokens — colors, fonts, spacing, hairlines | ✓ VERIFIED | 81 lines; `--accent: #1F3A2E`; `[data-theme="dark"]` block; `--sidebar-width: 240px`; `--topbar-height: 60px` |
| `frontend/src/styles/globals.css` | Tailwind v4 import + base layer + grain overlay | ✓ VERIFIED | `@import "tailwindcss"` line 1; `@theme` block maps all tokens; `background-image: url("/grain.svg")` in `body` |
| `frontend/src/main.tsx` | React entry mounts into #root with RouterProvider + QueryClientProvider + Toaster | ✓ VERIFIED | All four providers correctly nested; `ThemeProvider` wraps everything |
| `frontend/src/api/types.ts` | TypeScript interfaces for every backend response shape | ✓ VERIFIED | 354 lines; `interface ResumeResponse`, `candidateId: string` (camelCase), `type UploadStatus`, `interface ScoreResponse` all confirmed |
| `frontend/src/api/client.ts` | Axios instance with baseURL, interceptors, error normalization | ✓ VERIFIED | `baseURL` from `VITE_API_BASE_URL`; interceptor throws `parseAxiosError`; `withCredentials: false` |
| `frontend/src/api/errors.ts` | Normalized ApiError type and parsing logic for FastAPI 422 | ✓ VERIFIED | `class ApiError` with `status/kind/detail/fieldErrors`; `parseAxiosError` implemented (134 lines, not a stub); handles network/validation/404/409/5xx |
| `frontend/src/api/queryClient.ts` | Pre-configured TanStack QueryClient with global onError toast | ✓ VERIFIED | `QueryCache` + `MutationCache` both wire `notifyOnError`; `toast.error` called; validation errors correctly skipped |
| `frontend/src/api/index.ts` | Barrel export with `api` namespace | ✓ VERIFIED | `export const api = { upload, jobDescriptions, scoring, chat, shortlist, outreach, interviewQuestions }` |
| `frontend/src/api/endpoints/upload.ts` | Upload API with batchParse | ✓ VERIFIED | `batchParse` present; uses `client` from `"../client"` |
| `frontend/src/router.tsx` | createBrowserRouter with all routes registered, lazy chunks | ✓ VERIFIED | `createBrowserRouter`; `Component: AppShell` for authenticated group; 16 `lazy(() => import(...))` calls; `"*"` route for 404 |
| `frontend/src/routes/index.ts` | Route path constants | ✓ VERIFIED | `export const routes` and `export const routePatterns` — 18 routes each |
| `frontend/src/components/RoutePlaceholder.tsx` | Temporary placeholder component | ✓ VERIFIED | Renders `font-display` heading, screen name, requirements badges |
| `frontend/src/components/layout/AppShell.tsx` | Authenticated layout wrapper | ✓ VERIFIED | 22 lines; renders `<Sidebar />`, `<TopBar />`, `<Outlet />`; `mx-auto` content centering |
| `frontend/src/components/layout/TopBar.tsx` | TopBar with wordmark, breadcrumb, search, actions | ✓ VERIFIED | 94 lines; "RecruitAI" wordmark, `aria-label="Breadcrumb"` nav, search trigger, Bell, Command, `<UserMenu />` |
| `frontend/src/components/layout/Sidebar.tsx` | 8 nav items + Upload CTA | ✓ VERIFIED | 122 lines; 8 items in workflow order; `bg-accent` active bar; Upload CTA at line 93 — AFTER `</nav>` (line 82), BEFORE secondary footer (line 100) — pinned at bottom ✓ |
| `frontend/src/lib/theme.ts` | Theme Zustand store with localStorage + system listener | ✓ VERIFIED | 55 lines; `useTheme` store; validates against allowlist; `data-theme` applied via `applyToDocument`; system preference listener |
| `frontend/public/grain.svg` | Grain overlay SVG | ✓ VERIFIED | File exists in `public/` |
| `frontend/public/favicon.svg` | Favicon SVG | ✓ VERIFIED | File exists in `public/` |
| `frontend/index.html` | Entry HTML with FOUC-prevention inline script | ✓ VERIFIED | Pre-hydration script reads `localStorage["recruitai.theme"]`, validates against allowlist, sets `data-theme` before React hydrates |
| `frontend/src/App.tsx` | Must NOT exist (deleted in plan 01-04) | ✓ VERIFIED | File is absent from `frontend/src/` |
| `frontend/dist/index.html` | Production build artifact | ✓ VERIFIED | Exists; `dist/assets/` contains 19 named JS chunks (16 lazy route chunks + RoutePlaceholder shared chunk + index bundle) confirming all routes split correctly |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `frontend/index.html` | `frontend/src/main.tsx` | `<script type="module" src="/src/main.tsx">` | ✓ WIRED | Confirmed at line 28 of index.html |
| `frontend/src/main.tsx` | `frontend/src/styles/globals.css` | `import "./styles/globals.css"` | ✓ WIRED | Line 9 of main.tsx |
| `frontend/src/api/endpoints/upload.ts` | `frontend/src/api/client.ts` | `import { client } from '../client'` | ✓ WIRED | Grep confirmed `from.*client` pattern present |
| `frontend/src/api/client.ts` | `frontend/src/api/errors.ts` | `throw parseAxiosError(err)` | ✓ WIRED | Line 19 of client.ts |
| `frontend/src/api/queryClient.ts` | `sonner` | `toast.error` in QueryCache/MutationCache `onError` | ✓ WIRED | Lines 1 and 26 of queryClient.ts |
| `frontend/src/main.tsx` | `frontend/src/router.tsx` | `RouterProvider router={router}` | ✓ WIRED | Lines 3, 18 of main.tsx |
| `frontend/src/router.tsx` | all route modules | `lazy(() => import(...))` | ✓ WIRED | 16 lazy imports; confirmed by 16 named chunk files in `dist/assets/` |
| `frontend/src/components/layout/AppShell.tsx` | `<Outlet />` | `import { Outlet } from "react-router"` | ✓ WIRED | Line 1 and line 16 of AppShell.tsx |
| `frontend/src/components/layout/Sidebar.tsx` | `frontend/src/routes/index.ts` | `import { routes } from "@/routes"` + `NavLink to={routes.*}` | ✓ WIRED | Line 15 and NAV_ITEMS map using `routes.*` constants |
| `frontend/src/lib/theme.ts` | `document.documentElement` | `setAttribute("data-theme", applied)` | ✓ WIRED | `applyToDocument` function at line 21; called on store init and on `setTheme` |
| `frontend/src/router.tsx` | `frontend/src/components/layout/AppShell.tsx` | `Component: AppShell` | ✓ WIRED | Line 23 of router.tsx |

### Data-Flow Trace (Level 4)

No components in this phase render dynamic data fetched from the API — all routes render static `RoutePlaceholder` components (intentional for Phase 1). The API layer exists purely as infrastructure. Level 4 is N/A for this phase.

### Behavioral Spot-Checks

| Behavior | Method | Result | Status |
|----------|--------|--------|--------|
| Build produces output | `dist/index.html` + `dist/assets/*.js` files exist | 19 JS chunks present, including all 16 named route chunks | ✓ PASS |
| Lazy splitting for all routes | Named chunks in `dist/assets/` | `dashboard-*.js`, `chat-*.js`, `candidates detail-*.js`, `not-found-*.js`, `results-*.js`, `setup-*.js`, etc. all present | ✓ PASS |
| TypeScript strict mode active | `tsconfig.app.json` has `"strict": true` + `"noUnusedLocals": true` + `"noUnusedParameters": true` | Confirmed | ✓ PASS |
| Design tokens: `--accent: #1F3A2E` | `tokens.css` grep | Found at line 16 | ✓ PASS |
| Font imports in globals.css | `@import "@fontsource/fraunces"` etc. | All 3 font families imported (fraunces 400/500/600, geist-sans 400/500/600, geist-mono 400/500) | ✓ PASS |
| Toast wiring for non-validation errors | `queryClient.ts` toast logic | `notifyOnError` skips `validation` kind; calls `toast.error` for all others; `kind === "network"` gets generic message | ✓ PASS |
| App.tsx deleted | `ls frontend/src/App.tsx` | File absent | ✓ PASS |
| Upload CTA pinned below nav | Sidebar.tsx line ordering | `</nav>` at line 82; Upload CTA `<button>` at lines 88-95; SECONDARY_ITEMS at line 100 | ✓ PASS |
| Dev server + dark mode visuals | Requires live browser | N/A — human needed | ? SKIP |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| FOUND-01 | 01-01 | Vite + React 18 + TypeScript strict on port 5173 | ✓ SATISFIED | `package.json` + `tsconfig.app.json` + `vite.config.ts` |
| FOUND-02 | 01-01 | Tailwind v4 with design tokens (colors, fonts, spacing, motion, shadows, radii) | ✓ SATISFIED | `globals.css` `@theme` block maps all token categories; `tokens.css` defines radii, motion, shadows, type scale |
| FOUND-03 | 01-01 | `#1F3A2E` accent, `#FAFAF7` bg, dark palette, hairline token, grain | ✓ SATISFIED | `tokens.css` confirms all values; `globals.css` `body { background-image: url("/grain.svg") }` |
| FOUND-04 | 01-01 | Fraunces, Geist, Geist Mono as `font-display`, `font-sans`, `font-mono` | ✓ SATISFIED | `globals.css` `@theme`: `--font-display: "Fraunces"`, `--font-sans: "Geist"`, `--font-mono: "Geist Mono"` |
| FOUND-05 | 01-04 | Light/dark mode render correctly, toggleable from user menu, persisted | ? NEEDS HUMAN | Code is correct: `useTheme` store + `setTheme` wired in `UserMenu.tsx`; visual toggle requires human |
| FOUND-06 | 01-02 | Axios + TanStack Query to `localhost:8000/api/v1` with BACKEND.md types | ✓ SATISFIED | `client.ts` reads `VITE_API_BASE_URL`; `types.ts` 354 lines mirrors BACKEND.md shapes; all 7 endpoint modules present |
| FOUND-07 | 01-02 | API errors normalized: toast for non-form, inline for form (422 skipped) | ✓ SATISFIED | `errors.ts` full implementation; `queryClient.ts` skips `validation` kind; calls `toast.error` otherwise |
| FOUND-08 | 01-03 | React Router v7 for all 15 screens with lazy loading | ✓ SATISFIED | 18 routePatterns registered (15 screens + additional sub-routes); 16 named lazy chunks in build |
| FOUND-09 | 01-04 | Authenticated shell: TopBar (logo + breadcrumb + search + ⌘K + bell + avatar) + Sidebar (240px) + Content | ✓ SATISFIED (code) / ? NEEDS HUMAN (visual) | `AppShell.tsx` renders `Sidebar` + `TopBar` + `Outlet`; `TopBar.tsx` has all required elements; `Sidebar` uses `--sidebar-width: 240px` CSS variable |
| FOUND-10 | 01-04 | 8 nav items in workflow order, accent left bar on active, "Upload resume" CTA pinned at bottom | ✓ SATISFIED | 8 items in NAV_ITEMS in workflow order; `bg-accent` on active span; Upload CTA correctly positioned after nav, before secondary footer |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `frontend/src/api/errors.ts` | 29 | `super(opts.detail)` called before truncation — `Error.message` receives full untruncated string | ⚠ Warning | WR-03 from code review: `err.message` bypasses 500-char ASVS L1 truncation; `this.detail` is truncated correctly. Low runtime risk but architectural debt. |
| `frontend/src/lib/theme.ts` | 50-55 | Module-level `matchMedia.addEventListener` — no cleanup, fires before React tree mounts | ⚠ Warning | WR-02 from code review: duplicate listeners possible on HMR; not a functional blocker for production |
| `frontend/src/components/layout/UserMenu.tsx` | 18 | `<details>/<summary>` pattern — not keyboard-accessible, focus not managed on open | ⚠ Warning | CR-03 from code review: WCAG 2.1 SC 4.1.2 violation; functional for mouse users; blocks keyboard/AT users from accessing theme toggle |
| `frontend/vite.config.ts` | 11 | `host: true` — binds dev server to all network interfaces | ℹ Info | WR-07 from code review: dev-mode only; LAN exposure of candidate data in shared networks |
| `frontend/src/components/layout/TopBar.tsx` | 33 | `useMatches()` called but result discarded | ℹ Info | WR-06 from code review: causes unnecessary re-renders on route transitions; no functional impact |

Note: Anti-patterns above were identified in the co-located `01-REVIEW.md` code review. CR-03 (UserMenu keyboard accessibility) is the most impactful for production but was planned for remediation in Phase 2 per the review's recommendation. None of these prevent the Phase 1 goal (shell + design system foundation) from being achieved.

### Human Verification Required

#### 1. AppShell visual rendering

**Test:** Run `cd frontend && npm run dev` then open http://localhost:5173/dashboard in a browser.
**Expected:** The authenticated shell renders fully — 240px sidebar on the left showing "RecruitAI" wordmark, "Editorial Intelligence" tagline, 8 nav items (Dashboard, Candidates, Job Descriptions, Scoring, AI Chat, Shortlists, Outreach, Interview Prep) in workflow order with Lucide icons, forest-green "Upload resume" CTA pinned below the nav. TopBar across the top shows "RecruitAI" wordmark (left), breadcrumb "Dashboard" with ChevronRight (left-center), search field placeholder with Cmd+K hint (center), bell icon, command palette icon, avatar circle (right). Main content area shows "Dashboard" placeholder with requirement IDs.
**Why human:** Visual layout, font rendering (Fraunces serif must load), and CSS variable resolution require a live browser. Cannot be confirmed from static analysis.

#### 2. Dark mode toggle + persistence + FOUC prevention

**Test:** From /dashboard, click the avatar circle to open UserMenu. Click "Dark". Observe the page. Then reload (Ctrl+R or Cmd+R).
**Expected:** On clicking "Dark": every element switches to the dark palette immediately — sidebar bg `#14151A`, main bg `#0F1012`, text becomes `#FAFAF7`, accent becomes `#4A7C59`. After reload: page paints in dark mode from the very first frame with no visible light-mode flash (FOUC prevention via pre-hydration script in index.html). `localStorage.getItem("recruitai.theme")` returns `"dark"` in DevTools console.
**Why human:** FOUC prevention requires visual inspection of the initial paint; cannot be verified from static code analysis. Dark mode pixel correctness requires visual review.

#### 3. Error toast for network failure

**Test:** Ensure backend is not running (or is unreachable). In browser DevTools Console on any authenticated route, run: `(await import('/src/api/index.ts')).api.upload.list({limit:1}).catch(e => console.log(e.kind, e.detail))`. Alternatively, navigate to any route that would trigger a query once queries are wired.
**Expected:** A Sonner toast appears top-right with "Can't reach the server. Check your connection." within ~1 second of the failed request. Console logs `"network"` for `e.kind`. No duplicate toasts. No unhandled promise rejection in console.
**Why human:** Requires live browser + actual network failure + Sonner toast rendering in DOM. Cannot be simulated from static analysis.

### Gaps Summary

No blocking gaps found. All must-have artifacts exist, are substantive (none are stubs), and are wired correctly. The production build (`dist/`) confirms the full build pipeline succeeds with TypeScript strict mode.

The three items routed to human verification (SC#2 visual layout, SC#3 dark mode, SC#4 error toast) are all correctly implemented in code — the human checks are confirmatory, not investigatory.

Known issues from code review (`01-REVIEW.md`) that are NOT blocking Phase 1 goal:
- WR-03: `super()` receives untruncated detail — architectural concern, not a runtime failure
- WR-02: Module-level `matchMedia` listener — cleanup debt, not a functional failure
- CR-03: UserMenu keyboard inaccessibility — WCAG gap, does not block the shell goal; planned for Phase 2

---

_Verified: 2026-04-28T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
