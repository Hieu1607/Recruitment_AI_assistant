---
phase: 01-foundation-design-system
plan: "03"
subsystem: frontend
tags: [routing, react-router, lazy-loading, placeholder, typescript]
dependency_graph:
  requires:
    - frontend/src/api/index.ts — queryClient export (from 01-02)
    - frontend/src/styles/globals.css — Tailwind v4 design tokens (from 01-01)
    - frontend/package.json — react-router 7.0, sonner 1.7, @tanstack/react-query 5.59 (from 01-01)
  provides:
    - frontend/src/routes/index.ts — Route path constants (routes + routePatterns) for all 18 screens
    - frontend/src/components/RoutePlaceholder.tsx — Placeholder component used by every route until its phase ships
    - frontend/src/router.tsx — createBrowserRouter with all 18 routes lazily loaded
    - frontend/src/main.tsx — RouterProvider + QueryClientProvider + Toaster entry point
    - frontend/src/routes/*.tsx — 17 route modules (16 placeholders + 1 editorial 404 page)
  affects:
    - frontend/src/main.tsx — replaced placeholder App with RouterProvider
key_files:
  created:
    - frontend/src/routes/index.ts
    - frontend/src/components/RoutePlaceholder.tsx
    - frontend/src/router.tsx
    - frontend/src/routes/landing.tsx
    - frontend/src/routes/login.tsx
    - frontend/src/routes/dashboard.tsx
    - frontend/src/routes/candidates/list.tsx
    - frontend/src/routes/candidates/detail.tsx
    - frontend/src/routes/job-descriptions/list.tsx
    - frontend/src/routes/job-descriptions/edit.tsx
    - frontend/src/routes/scoring/setup.tsx
    - frontend/src/routes/scoring/results.tsx
    - frontend/src/routes/chat.tsx
    - frontend/src/routes/shortlists/list.tsx
    - frontend/src/routes/shortlists/collection.tsx
    - frontend/src/routes/outreach.tsx
    - frontend/src/routes/interview-questions/list.tsx
    - frontend/src/routes/interview-questions/detail.tsx
    - frontend/src/routes/settings.tsx
    - frontend/src/routes/not-found.tsx
  modified:
    - frontend/src/main.tsx — wired RouterProvider + QueryClientProvider + Toaster
decisions:
  - "Router lazy() helper wraps React Router's built-in lazy() API to unify the { Component } shape — no external lazy-loading library needed"
  - "Authenticated shell (Outlet wrapper) stubbed inline in router.tsx — plan 01-04 will replace it with the real AppShell layout"
  - "chat.tsx and /chat/:sessionId both share the same route module — single file handles both empty session and loaded session states"
  - "App.tsx left in place (not deleted) — plan 01-04 will clean it up after the layout shell is wired in"
  - "not-found.tsx renders an editorial 404 (not a RoutePlaceholder) — it is a real permanent screen, not a future-phase stand-in"
metrics:
  duration_minutes: 4
  tasks_completed: 3
  tasks_total: 3
  files_created: 20
  files_modified: 1
  completed_date: "2026-04-28"
---

# Phase 1 Plan 03: React Router v7 Routing Skeleton Summary

React Router v7 createBrowserRouter with 18 lazily-chunked routes, type-safe path constants, editorial RoutePlaceholder component, RouterProvider + QueryClientProvider + Toaster wired in main.tsx — all 18 URL patterns locked, 16 route-level lazy chunks confirmed in build output.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Define route path constants and RoutePlaceholder component | ecdb3b3 | frontend/src/routes/index.ts, frontend/src/components/RoutePlaceholder.tsx |
| 2 | Create 17 lazy-loaded route modules (one per screen) | c173af9 | frontend/src/routes/*.tsx (all 17 route files) |
| 3 | Build router + RouterProvider in main.tsx | 4659245 | frontend/src/router.tsx, frontend/src/main.tsx |

## Verification Results

- `npm run typecheck` (tsc -b) exits 0 after each task — no TypeScript errors
- `npm run build` exits 0, produces dist/index.html (339 kB JS / 110 kB gzip main bundle)
- dist/assets contains 16 named route chunks: dashboard, landing, login, chat, outreach, settings, results, setup, not-found, detail (x2), list (x3), collection, edit — lazy splitting confirmed
- RoutePlaceholder shared chunk emitted separately (0.77 kB) — shared across all placeholder routes
- All 18 routePatterns referenced in router.tsx verified by build (no unused imports)

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

- All 16 placeholder routes (all except not-found.tsx) render `RoutePlaceholder` — this is intentional and expected per the plan. Each stub names its owning phase and requirement IDs, so future agents know exactly what to replace.
- `frontend/src/App.tsx` remains from plan 01-01 — not imported anywhere now (main.tsx uses RouterProvider). Plan 01-04 will delete it when the layout shell ships.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced. Route modules are static React components with no API calls. RouterProvider handles client-side navigation only — no server rendering or data fetching added at this layer.

## Self-Check: PASSED

- frontend/src/routes/index.ts: FOUND (contains "export const routes" and "export const routePatterns")
- frontend/src/components/RoutePlaceholder.tsx: FOUND (contains "RoutePlaceholder" and "font-display")
- frontend/src/router.tsx: FOUND (contains "createBrowserRouter", all 18 routePatterns, "lazy(() => import")
- frontend/src/main.tsx: FOUND (contains "RouterProvider", "QueryClientProvider", "<Toaster")
- frontend/src/routes/dashboard.tsx: FOUND (contains "RoutePlaceholder", "DASH-01")
- frontend/src/routes/candidates/list.tsx: FOUND (contains "CAND-01")
- frontend/src/routes/scoring/results.tsx: FOUND (contains "Match Results")
- frontend/src/routes/not-found.tsx: FOUND (contains "404")
- Commits ecdb3b3, c173af9, 4659245: present in git log
- dist/assets/dashboard-*.js: FOUND (lazy chunk)
- dist/assets/chat-*.js: FOUND (lazy chunk)
- dist/assets/not-found-*.js: FOUND (lazy chunk)
