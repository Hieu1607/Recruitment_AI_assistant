# Rich Outreach Design

## Goal

Upgrade outreach from plain text drafts to reusable rich HTML email composition with shared editing across all outreach entry points, reusable templates with variables, image support via MinIO or direct URL, and Gmail delivery that preserves formatting.

## Current State

- `frontend/src/routes/outreach.tsx` uses plain text `input` and `textarea` for compose and edit.
- `frontend/src/routes/shortlists/collection.tsx` creates bulk outreach drafts from plain text `subject_template` and `body_template`.
- `backend/src/models/outreach.py` stores only `subject` and `body`.
- `backend/src/api/v1/endpoints/outreach.py` and `backend/src/api/v1/endpoints/shortlist.py` accept and return plain text outreach payloads.
- `backend/src/services/email_templates.py` only trims text and does not build multipart HTML email payloads.
- MinIO object storage infrastructure already exists in `backend/src/services/object_storage.py`.

## Requirements

1. Recruiters can compose professional outreach with formatting such as bold, italic, highlight/color, links, and images.
2. The same editor model is reused anywhere outreach is created or edited, including shortlist bulk outreach flows.
3. Templates are reusable and support variables such as candidate name, company name, and job title.
4. AI-generated templates can populate the same template model later without requiring a second storage format.
5. Gmail delivery preserves formatting by sending HTML and text alternatives.
6. Images can be inserted by upload to MinIO or by supplying an external image URL.

## Proposed Architecture

### Data Model

Extend `OutreachMessage` with:

- `body_text`: plain-text fallback for email clients and indexing
- `body_html`: sanitized HTML used for delivery and preview
- `template_id`: optional foreign key to `OutreachTemplate`
- `render_variables`: JSON snapshot of variables used to render the message

Add `OutreachTemplate` with:

- `id`
- `name`
- `created_by_user_id`
- `job_id` nullable for future scoping
- `content_source`
- `subject_template`
- `body_text_template`
- `body_html_template`
- `editor_json`
- `variables_used`
- timestamps

The frontend editor works primarily with `editor_json`, while the backend stores the rendered `body_html` and `body_text` for stable delivery.

### Shared Editor

Create a shared frontend component, `OutreachRichEditor`, using Tiptap. It should support:

- bold, italic, underline
- text color and highlight
- bullet and ordered lists
- link insertion and editing
- image insertion from URL
- image upload to backend, returning a durable MinIO-backed public asset URL
- placeholder chips or token insertion for template variables

The same component is used by:

- outreach compose modal
- outreach detail edit panel
- shortlist bulk outreach modal
- template create/edit modal

### Template System

Templates are stored separately from sent drafts. A template contains placeholders such as:

- `{{candidate_name}}`
- `{{candidate_email}}`
- `{{company_name}}`
- `{{job_title}}`

Bulk draft creation selects a template or starts from a blank rich editor, then renders one draft per selected candidate. Rendering occurs on the backend for consistency and to avoid divergent frontend behavior.

### Image Support

Two insertion modes:

- Upload image file to a dedicated MinIO bucket and insert returned URL into the HTML
- Paste a direct image URL into the editor

The backend must expose:

- an authenticated upload endpoint for outreach assets
- a stable asset URL strategy suitable for email delivery

Short-lived presigned URLs are not suitable for outbound email because older emails would lose access to their images.

### HTML Email Sending

Gmail delivery should send a multipart message with:

- `text/plain` using `body_text`
- `text/html` using `body_html`

Before storing or sending, `body_html` should be sanitized to an allowlist of email-safe tags and attributes.

## Implementation Notes

- Reuse existing object storage patterns instead of introducing a second storage stack.
- Keep placeholder rendering centralized in backend services, not in multiple frontend call sites.
- Do not allow arbitrary raw HTML injection from the client without sanitization.
- Rich editor support should be introduced only for outreach surfaces, not globally across unrelated forms.

## Risks

- Email client compatibility varies, so editor output must stay conservative.
- Stored HTML requires sanitization for UI safety and outbound mail quality.
- Existing outreach records only have plain text, so migration must backfill `body_text` and generate simple HTML wrappers.
- Adding a rich editor package increases frontend bundle size and requires careful styling to match the existing design system.

## Phasing

### Phase 1

- Rich outreach messages
- Shared editor
- Reusable templates
- Shortlist bulk draft integration
- MinIO image upload
- Gmail HTML delivery

### Phase 2

- AI template generation UI
- job-scoped template libraries
- richer template preview and A/B variants
