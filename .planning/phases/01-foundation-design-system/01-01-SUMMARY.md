---
phase: 01-foundation-design-system
plan: "01"
subsystem: frontend
tags: [scaffold, vite, react, typescript, tailwind, design-tokens, fonts]
dependency_graph:
  requires: []
  provides:
    - frontend/package.json — Vite 5 + React 18 + TS 5 + Tailwind v4 dependency manifest
    - frontend/src/styles/tokens.css — design token definitions (light + dark)
    - frontend/src/styles/globals.css — Tailwind v4 entry, font imports, base layer
    - frontend/src/lib/cn.ts — cn() class-name utility (clsx + tailwind-merge)
    - frontend/public/grain.svg — subtle grain overlay
    - frontend/public/favicon.svg — forest-green editorial favicon
  affects: []
tech_stack:
  added:
    - Vite 5.4
    - React 18.3
    - TypeScript 5.9
    - Tailwind CSS v4 (via @tailwindcss/vite plugin)
    - @fontsource/fraunces 5.1 + @fontsource/geist-sans 5.1 + @fontsource/geist-mono 5.1
    - clsx 2.1 + tailwind-merge 2.5
    - react-router 7.0
    - @tanstack/react-query 5.59
    - zustand 5.0
    - lucide-react 0.460
    - sonner 1.7
    - axios 1.7
  patterns:
    - CSS-first design tokens via @theme in Tailwind v4
    - Pre-hydration FOUC-prevention theme resolver in index.html inline script
    - data-theme attribute on <html> for dark mode switching
    - cn() utility (clsx + twMerge) for conditional Tailwind class merging
key_files:
  created:
    - frontend/package.json
    - frontend/tsconfig.json
    - frontend/tsconfig.app.json
    - frontend/tsconfig.node.json
    - frontend/vite.config.ts
    - frontend/index.html
    - frontend/.gitignore
    - frontend/.env.example
    - frontend/eslint.config.js
    - frontend/src/main.tsx
    - frontend/src/App.tsx
    - frontend/src/styles/tokens.css
    - frontend/src/styles/globals.css
    - frontend/public/grain.svg
    - frontend/public/favicon.svg
    - frontend/src/lib/cn.ts
  modified:
    - .gitignore (added !frontend/src/lib/ negation to allow lib/ under frontend/src)
decisions:
  - "Use tsc -b (composite build mode) instead of tsc --noEmit -b (invalid flag combo in TS 5.9)"
  - "No postcss.config.js — Tailwind v4 Vite plugin handles processing alone; dual pipeline would break @theme resolution"
  - "Added *.tsbuildinfo to frontend/.gitignore to exclude TypeScript composite build artifacts"
  - "Root .gitignore lib/ rule required !frontend/src/lib/ negation to allow cn.ts to be tracked"
metrics:
  duration_minutes: 5
  tasks_completed: 3
  tasks_total: 3
  files_created: 16
  files_modified: 1
  completed_date: "2026-04-28"
---

# Phase 1 Plan 01: Vite + React + TS Scaffold with Design Tokens Summary

Vite 5 + React 18 + TypeScript strict scaffold with Tailwind v4 CSS-first design tokens (forest green `#1F3A2E` accent, off-white `#FAFAF7` bg, Fraunces/Geist/Geist Mono fonts, grain overlay, hairline borders).

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Initialize Vite + React + TS package and tooling configs | 1d353ba | package.json, tsconfig.app.json, vite.config.ts, index.html, eslint.config.js |
| 2 | Write design tokens (CSS variables) and Tailwind v4 theme | 0513acd | tokens.css, globals.css, grain.svg, favicon.svg, cn.ts |
| 3 | Create entry point and placeholder App | 5b3068c | .gitignore (*.tsbuildinfo added) |

## Verification Results

- `npm install` exits 0 (224 packages, 2 moderate audit warnings — not security-blocking for frontend-only bundle)
- `npm run typecheck` (tsc -b) exits 0, no TypeScript errors
- `npm run build` exits 0, produces dist/index.html (143 kB JS / 46 kB gzip, 16 kB CSS / 4 kB gzip)
- `grep -q "#1F3A2E" frontend/src/styles/tokens.css` succeeds
- `frontend/public/grain.svg` and `frontend/dist/index.html` exist

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed invalid typecheck script flag**
- **Found during:** Task 1 verification
- **Issue:** `tsc --noEmit -b` is invalid in TypeScript 5.9 — `-b` (build mode) and `--noEmit` cannot be combined
- **Fix:** Changed `typecheck` script to `tsc -b` (composite build mode, which also does type checking without emitting when `noEmit: true` is in tsconfig)
- **Files modified:** frontend/package.json
- **Commit:** 1d353ba

**2. [Rule 2 - Missing critical functionality] Added *.tsbuildinfo to .gitignore**
- **Found during:** Task 3
- **Issue:** TypeScript composite build generates `tsconfig.app.tsbuildinfo` and `tsconfig.node.tsbuildinfo` which should not be committed
- **Fix:** Added `*.tsbuildinfo` pattern to `frontend/.gitignore`
- **Files modified:** frontend/.gitignore
- **Commit:** 5b3068c

**3. [Rule 3 - Blocking issue] Root .gitignore lib/ rule blocked frontend/src/lib/**
- **Found during:** Task 2 staging
- **Issue:** Root `.gitignore` has a `lib/` pattern (Python convention) which prevented staging `frontend/src/lib/cn.ts`
- **Fix:** Added `!frontend/src/lib/` and `!frontend/src/lib/**` negation rules to root `.gitignore`
- **Files modified:** .gitignore
- **Commit:** 0513acd

## Known Stubs

- `frontend/src/App.tsx` — placeholder smoke test component; will be replaced by routing skeleton in plan 01-03. Does not block the plan's goal (foundation verification).

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced. Bundle integrity only (ASVS L1 scope). Pre-hydration theme resolver script reads/writes localStorage only — no user-controlled patterns, no eval, no dynamic script injection. `.env.example` contains placeholder values only; no real secrets committed.

## Self-Check: PASSED

- frontend/package.json: FOUND
- frontend/tsconfig.app.json: FOUND
- frontend/vite.config.ts: FOUND
- frontend/src/styles/tokens.css: FOUND (contains --accent: #1F3A2E)
- frontend/src/styles/globals.css: FOUND (contains @import "tailwindcss" and --font-display: "Fraunces")
- frontend/public/grain.svg: FOUND
- frontend/public/favicon.svg: FOUND
- frontend/src/lib/cn.ts: FOUND
- frontend/dist/index.html: FOUND (build artifact, not committed)
- Commits 1d353ba, 0513acd, 5b3068c: present in git log
