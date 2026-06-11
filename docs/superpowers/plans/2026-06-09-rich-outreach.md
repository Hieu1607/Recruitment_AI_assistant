# Rich Outreach Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build shared rich HTML outreach editing, reusable templates, MinIO-backed images, and Gmail delivery that preserves email formatting across individual and shortlist bulk outreach flows.

**Architecture:** Add backend persistence for HTML/text/template snapshots, centralize rendering and sanitization in backend services, and use one shared frontend rich editor across outreach compose, edit, and shortlist bulk draft creation. Rich outreach assets use the existing MinIO object storage service, while Gmail delivery sends multipart plain-text and HTML content.

**Tech Stack:** FastAPI, SQLAlchemy, Alembic, Celery, Gmail API integration, MinIO, React, TypeScript, Tiptap, React Query

---

### Task 1: Backend Outreach Persistence

**Files:**
- Create: `backend/migrations/versions/20260609_0012_rich_outreach_templates.py`
- Create: `backend/src/models/outreach_template.py`
- Modify: `backend/src/models/outreach.py`
- Modify: `backend/src/models/__init__.py`
- Modify: `backend/src/api/v1/endpoints/outreach.py`
- Test: `backend/tests/test_outreach_endpoints.py`

- [ ] Add failing backend tests for rich outreach fields and template CRUD expectations
- [ ] Run the targeted tests and verify they fail for missing fields/endpoints
- [ ] Implement the model and migration changes for `body_text`, `body_html`, `template_id`, `render_variables`, and `outreach_templates`
- [ ] Update outreach endpoint schemas/serialization to use the rich payload shape
- [ ] Re-run the targeted backend tests until green

### Task 2: Backend Rendering, Sanitization, and Gmail HTML Delivery

**Files:**
- Create: `backend/src/services/outreach_service.py`
- Modify: `backend/src/services/email_templates.py`
- Modify: `backend/src/services/gmail_service.py`
- Modify: `backend/worker/tasks.py`
- Modify: `backend/src/api/v1/endpoints/shortlist.py`
- Test: `backend/tests/test_email_templates.py`
- Test: `backend/tests/test_outreach_send_endpoint.py`
- Test: `backend/tests/test_shortlist_endpoints.py`

- [ ] Add failing tests for placeholder rendering, sanitized HTML, and multipart send behavior
- [ ] Run the targeted tests and verify they fail for the expected missing behavior
- [ ] Implement centralized render/sanitize helpers and bulk draft rendering from templates
- [ ] Update Gmail send flow and worker task flow to send `text/plain` plus `text/html`
- [ ] Re-run the targeted backend tests until green

### Task 3: Outreach Asset Upload API

**Files:**
- Create: `backend/src/api/v1/endpoints/outreach_assets.py`
- Modify: `backend/src/api/v1/api.py`
- Modify: `backend/src/core/config.py`
- Modify: `backend/src/services/object_storage.py`
- Test: `backend/tests/test_outreach_assets.py`

- [ ] Add failing tests for authenticated outreach asset upload and returned URL shape
- [ ] Run the targeted tests and verify they fail for missing endpoint behavior
- [ ] Implement upload handling using the existing MinIO service and a durable bucket/config path for outreach assets
- [ ] Re-run the targeted backend tests until green

### Task 4: Frontend Shared Rich Editor and API Types

**Files:**
- Create: `frontend/src/components/outreach/OutreachRichEditor.tsx`
- Create: `frontend/src/components/outreach/VariableTokenMenu.tsx`
- Modify: `frontend/package.json`
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/endpoints/outreach.ts`
- Modify: `frontend/src/api/endpoints/shortlist.ts`
- Test: `frontend/src/components/outreach/` via typecheck/build

- [ ] Add the frontend dependencies required for Tiptap and email-safe editing
- [ ] Introduce API type changes for rich outreach messages, templates, and upload responses
- [ ] Build the shared editor component with toolbar actions, image URL insertion, upload, and variable insertion
- [ ] Run frontend typecheck/build to catch typing and integration issues

### Task 5: Outreach Route Integration

**Files:**
- Modify: `frontend/src/routes/outreach.tsx`

- [ ] Replace compose/edit plain text controls with the shared rich editor
- [ ] Add template selection/creation affordances where appropriate without breaking existing draft management flow
- [ ] Keep list/detail interactions and send/save behavior consistent with current UX
- [ ] Run frontend typecheck/build to verify route integration

### Task 6: Shortlist Bulk Outreach Integration

**Files:**
- Create: `frontend/src/components/outreach/OutreachTemplatePicker.tsx`
- Modify: `frontend/src/routes/shortlists/collection.tsx`
- Modify: `frontend/src/routes/scoring/setup.tsx` if the dormant bulk action is wired in this pass

- [ ] Replace the shortlist bulk plain text modal with the shared template-capable rich editor flow
- [ ] Allow choosing a stored template or editing a blank draft before rendering per-candidate outreach drafts
- [ ] Ensure every outreach entry point that currently drafts content uses the same rich editor path
- [ ] Run frontend typecheck/build to verify integration

### Task 7: Verification

**Files:**
- Verify modified backend/frontend files above

- [ ] Run targeted backend tests for outreach, shortlist, email template, and send flows
- [ ] Run `npm run typecheck` and `npm run build` in `frontend`
- [ ] Run any focused browser or e2e smoke if needed for outreach compose/save flows
- [ ] Review `git diff` for accidental unrelated changes before final summary
