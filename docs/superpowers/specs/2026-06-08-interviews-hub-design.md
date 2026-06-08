# Interviews Hub Design

Date: 2026-06-08
Status: Proposed
Scope: Add a recruiter-facing `/interviews` hub for invitation link management and make interview templates a child area with full CRUD

## Goal

Turn interviews into a first-class recruiter workspace area instead of a scattered set of template and report screens.

The new `/interviews` route becomes the parent management surface for:

- interview invitation links
- invitation status and report access
- revocation of active links
- navigation into template management

`/interviews/templates` remains a dedicated child route and supports create, read, update, and delete for job-scoped interview templates.

## User Decisions Captured

- `/interviews` is a parent hub, not just a template list.
- Interview links need `create`, `read`, and `revoke`.
- Interview templates must sit under `/interviews` and support full CRUD.
- Public candidate interview URLs stay unchanged.

## Existing Context

The frontend already contains:

- route-level screens for interview templates and interview reports
- a recruiter dialog that sends interview invitations
- API clients for listing and creating invitations
- API clients for listing, creating, getting, and updating templates
- a public interview route at `/interviews/:token`

Current gaps:

- no authenticated recruiter route at `/interviews`
- no obvious recruiter navigation entry for the overall interviews module
- no template delete client
- no invitation revoke client
- route naming in `routes/index.ts` is inconsistent around interviews vs interview questions

## UX Design

### `/interviews` hub

The hub is the default recruiter entry point for interview operations.

Primary content:

- page title and short module description
- selected-job guardrail matching existing job-scoped flows
- primary CTA to create or send a new interview invitation
- data table of invitations for the selected job

Invitation table columns:

- candidate
- template
- status
- created or sent timestamp
- expiry timestamp
- completion timestamp when available
- actions

Row actions:

- copy public link
- open report when the invitation is completed and a report exists
- revoke invitation when it is still active

Top-level secondary navigation inside the page:

- `Interview links`
- `Templates`

This keeps `/interviews` as the functional landing page while preserving a direct path to `/interviews/templates`.

### `/interviews/templates`

This route remains a full-page template management screen, but now explicitly sits under the interviews module.

Capabilities:

- list templates for the selected job
- create template
- open template detail
- edit template
- delete template

Delete behavior:

- deletion is explicit and confirmed in the UI
- successful deletion returns the user to the template list
- the UI surfaces backend errors such as template-in-use constraints

## Navigation

Sidebar changes:

- replace the current `Interview Prep` primary nav item with `Interviews`
- point it to `/interviews`

Route relationships:

- `/interviews` -> recruiter hub for invitations
- `/interviews/templates` -> template list
- `/interviews/templates/:id` -> template detail/editor
- `/interviews/reports/:interviewSessionId` -> report detail
- `/interviews/:token` -> public candidate route, unchanged

Because `/interviews/:token` already exists, the authenticated hub must be registered so static recruiter paths like `/interviews`, `/interviews/templates`, and `/interviews/reports/:id` remain matched correctly before the public token route catches other dynamic paths.

## Data And API Requirements

Frontend client additions required:

- invitation revoke endpoint
- template delete endpoint

Expected API shapes:

- `POST` or `PATCH /interview-invitations/{invitation_id}/revoke`
- `DELETE /interview-templates/{template_id}`

If the backend exposes slightly different paths, the frontend should adapt to the real API rather than inventing a new contract. The recruiter UX and route structure stay the same.

Frontend query behavior:

- `/interviews` uses `["interview-invitations", selectedJobId]`
- `/interviews/templates` uses `["interview-templates", selectedJobId]`
- revoke and delete mutations invalidate their respective lists

## Component Boundaries

Keep the module split into small focused parts:

- route component for `/interviews`
- presentational table/actions for invitations if extraction improves clarity
- reuse `InvitationSendDialog` where possible instead of duplicating send flow
- keep template editor flow in existing `TemplateEditor`
- keep report rendering in existing `ReportView`

This work should avoid a monolithic interviews page that mixes template editing, invitation sending, and report rendering in one file.

## Error Handling

The recruiter hub should surface predictable failure states:

- no selected job
- no templates available when attempting to create a link
- invitation revoke failure
- template delete failure
- list fetch failure

Desired UI behavior:

- empty states when there are no invitations or no templates
- toast feedback for create, revoke, update, and delete actions
- destructive actions require explicit confirmation

## Testing Strategy

Frontend behavior to cover:

- recruiter can open `/interviews`
- recruiter sees invitation rows for the selected job
- recruiter can navigate from `/interviews` to `/interviews/templates`
- recruiter can create a template
- recruiter can update a template
- recruiter can delete a template
- recruiter can revoke an invitation

Verification approach:

- add targeted frontend tests where the current suite has coverage patterns for route rendering and interaction
- run TypeScript build and relevant tests after implementation

## Non-Goals

- changing the public candidate interview URL structure
- redesigning the report page
- merging template editing directly into the `/interviews` hub
- introducing a new interview-question feature area as part of this change

## Recommendation

Implement `interviews` as a normalized recruiter module with a real landing page at `/interviews`, child template management at `/interviews/templates`, and minimal but complete invitation-link operations: create, read, copy, report access, and revoke.
