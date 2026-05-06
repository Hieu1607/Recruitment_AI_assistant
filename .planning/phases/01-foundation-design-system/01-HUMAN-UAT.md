---
status: partial
phase: 01-foundation-design-system
source: [01-VERIFICATION.md]
started: 2026-04-28T00:00:00Z
updated: 2026-04-28T00:00:00Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. AppShell visual rendering
expected: Visiting /dashboard shows editorial shell — 240px Sidebar with 8 nav items (Dashboard, Candidates, Job Descriptions, Scoring, AI Chat, Shortlists, Outreach, Interview Prep) + "Upload resume" CTA pinned at bottom; TopBar with "RecruitAI" wordmark, route breadcrumb ("Dashboard"), search field with ⌘K hint, bell + command-palette icons, avatar; Fraunces serif loaded for headings.
result: [pending]

### 2. Dark mode toggle + FOUC prevention
expected: Click avatar → select "Dark" → every shell pixel switches to dark palette. Reload the page — dark mode paints immediately on first paint with no light-mode flash (FOUC-prevention script in index.html fires before React hydrates).
result: [pending]

### 3. Error toast for network failure
expected: With backend unreachable (or after `docker compose stop backend`), navigate to any authenticated route that triggers a query. A Sonner toast appears top-right: "Can't reach the server. Check your connection." — confirming FOUND-07 / Phase 1 SC#4 is wired end-to-end.
result: [pending]

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
