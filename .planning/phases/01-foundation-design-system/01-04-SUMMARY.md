---
phase: 01-foundation-design-system
plan: "04"
subsystem: frontend
tags: [layout, shell, sidebar, topbar, dark-mode, theme, zustand, router]
dependency_graph:
  requires:
    - frontend/src/lib/cn.ts — cn() utility (from 01-01)
    - frontend/src/styles/tokens.css — design tokens incl. --hairline, --sidebar-width, --topbar-height, --content-max (from 01-01)
    - frontend/src/routes/index.ts — routes + routePatterns constants (from 01-03)
    - frontend/src/api/index.ts — queryClient export (from 01-02)
    - frontend/src/router.tsx — router skeleton (from 01-03)
    - frontend/src/main.tsx — RouterProvider entry point (from 01-03)
  provides:
    - frontend/src/lib/theme.ts — Zustand useTheme store with light/dark/system + localStorage persistence
    - frontend/src/components/ThemeProvider.tsx — Theme initialization wrapper + meta theme-color updater
    - frontend/src/components/layout/Sidebar.tsx — Left navigation with 8 items + Upload CTA
    - frontend/src/components/layout/TopBar.tsx — Top bar with wordmark, breadcrumb, search, actions
    - frontend/src/components/layout/UserMenu.tsx — Avatar popover with theme toggle + settings + sign out
    - frontend/src/components/layout/AppShell.tsx — Authenticated layout wrapper (Sidebar + TopBar + Outlet)
    - frontend/src/router.tsx — Updated: AppShell wraps all authenticated routes; / and /login render bare
    - frontend/src/main.tsx — Updated: ThemeProvider wraps RouterProvider
  affects:
    - All authenticated routes (phases 3–11) — they now render inside AppShell
tech_stack:
  added:
    - zustand (useTheme store for theme state management)
  patterns:
    - Zustand create() store for theme persistence with localStorage + system preference listener
    - data-theme attribute on <html> for CSS token switching (light/dark modes)
    - var(--hairline) token for interactive backgrounds — dark-mode parity pattern
    - Route-derived breadcrumb via useLocation + prefix-match rules array
    - <details>/<summary> native popover for UserMenu (no library dependency at this phase)
    - AppShell as React Router Component= layout wrapper (not element=)
    - mx-auto content centering with CSS var(--content-max) for wide displays
key_files:
  created:
    - frontend/src/lib/theme.ts
    - frontend/src/components/ThemeProvider.tsx
    - frontend/src/components/layout/Sidebar.tsx
    - frontend/src/components/layout/TopBar.tsx
    - frontend/src/components/layout/UserMenu.tsx
    - frontend/src/components/layout/AppShell.tsx
  modified:
    - frontend/src/router.tsx — replaced bare <Outlet> with Component: AppShell
    - frontend/src/main.tsx — added ThemeProvider wrapper
  deleted:
    - frontend/src/App.tsx — unused placeholder replaced by router-driven layout
decisions:
  - "var(--hairline) token used for active/hover nav backgrounds instead of hardcoded rgba(0,0,0,0.04) — the hardcoded value was near-invisible on dark sidebar bg #14151A (W-4 fix)"
  - "Upload CTA pinned BELOW the primary nav (FOUND-10 / SC#2 — plan-checker B-1 override applied)"
  - "route-derived breadcrumb uses prefix-match array with longer paths first (prevents /shortlists matching /shortlists/collections)"
  - "App.tsx deleted — confirmed no imports referenced it before deletion"
  - "Comment in Sidebar.tsx explains why rgba(0,0,0,0.04) is NOT used — regex check matched the comment text, not an actual style value"
metrics:
  duration_minutes: 3
  tasks_completed: 4
  tasks_total: 4
  files_created: 6
  files_modified: 2
  files_deleted: 1
  completed_date: "2026-04-28"
---

# Phase 1 Plan 04: AppShell Layout Shell (TopBar + Sidebar + Theme) Summary

Authenticated layout shell with Zustand-powered light/dark/system theme store (localStorage persist, no FOUC), forest-green active indicator Sidebar with 8 nav items + pinned Upload CTA, editorial TopBar with wordmark + route-derived breadcrumb + search trigger, UserMenu theme toggle — all authenticated routes wrapped in AppShell, public routes bare.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Theme store + ThemeProvider | 3cfa0cc | frontend/src/lib/theme.ts, frontend/src/components/ThemeProvider.tsx |
| 2 | Sidebar with 8 nav items + Upload CTA | 6b6deb2 | frontend/src/components/layout/Sidebar.tsx |
| 3 | TopBar + UserMenu | 4fd7536 | frontend/src/components/layout/TopBar.tsx, frontend/src/components/layout/UserMenu.tsx |
| 4 | AppShell + router wiring + delete App.tsx | a35d834 | frontend/src/components/layout/AppShell.tsx, router.tsx, main.tsx (App.tsx deleted) |

## Verification Results

- `npm run typecheck` exits 0 after Task 1
- `npm run build` exits 0 after Tasks 2, 3, and 4
- All acceptance criteria verified:
  - theme.ts contains useTheme, data-theme, localStorage
  - ThemeProvider.tsx contains ThemeProvider
  - Sidebar.tsx: RecruitAI, Editorial Intelligence, Upload resume, all 8 nav labels, bg-accent, NavLink, no hardcoded rgba as style values
  - TopBar.tsx: Search candidates JDs, UserMenu, RecruitAI wordmark, Breadcrumb aria-label, useLocation/useMatches
  - UserMenu.tsx: useTheme, Light/Dark/System options, Sign out
  - AppShell.tsx: TopBar + Sidebar + Outlet, mx-auto
  - router.tsx: Component: AppShell
  - main.tsx: ThemeProvider
  - App.tsx: confirmed deleted

## Deviations from Plan

### Auto-fixed Issues

None — plan executed exactly as written.

### Notes

**Comment match on rgba check:** The acceptance criteria check for "no hardcoded rgba" matched a comment in Sidebar.tsx that explains WHY the old value was removed ("hardcoded rgba(0,0,0,0.04) was near-invisible on #14151A"). The comment is documentation of the design decision, not an actual CSS value — no hardcoded rgba is used in any style attribute or className.

## Known Stubs

- UserMenu shows hardcoded "Recruiter" name and "user@recruitai.local" email — real user identity will come from auth context in a later phase (Phase 11 - Settings / Auth).
- Upload resume button in Sidebar has `type="button"` but no click handler — the upload modal is delivered in Phase 5 (Upload Resumes screen). This is intentional per the plan.
- Search trigger in TopBar is a no-op styled element — the ⌘K command palette is a later phase feature, per the plan's threat model note.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced.

- Theme persistence: reads/writes `localStorage["recruitai.theme"]`. Value validated against allowlist `["light", "dark", "system"]` before applying `data-theme` — never set from raw user input (ASVS L1 compliant).
- UserMenu: no innerHTML, no dangerouslySetInnerHTML, no user-controlled HTML rendering.
- Search trigger (⌘K): no-op at this phase, no data processing.

## Self-Check: PASSED

- frontend/src/lib/theme.ts: FOUND (contains useTheme, data-theme, localStorage)
- frontend/src/components/ThemeProvider.tsx: FOUND (contains ThemeProvider)
- frontend/src/components/layout/Sidebar.tsx: FOUND (contains RecruitAI, Upload resume, NavLink)
- frontend/src/components/layout/TopBar.tsx: FOUND (contains RecruitAI, UserMenu, Breadcrumb)
- frontend/src/components/layout/UserMenu.tsx: FOUND (contains useTheme, Light, Dark, System, Sign out)
- frontend/src/components/layout/AppShell.tsx: FOUND (contains TopBar, Sidebar, Outlet, mx-auto)
- frontend/src/router.tsx: FOUND (contains Component: AppShell)
- frontend/src/main.tsx: FOUND (contains ThemeProvider)
- frontend/src/App.tsx: CONFIRMED DELETED
- Commits 3cfa0cc, 6b6deb2, 4fd7536, a35d834: all present in git log
