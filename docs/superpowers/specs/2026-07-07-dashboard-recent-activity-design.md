# Dashboard Recent Activity Design

## Goal

Make the dashboard's `Recent Activity` panel reflect meaningful recruiter-facing work that actually happened, without turning it into a duplicate of the notification center.

## Current State

- `Recent Activity` is assembled only in [frontend/src/routes/dashboard.tsx](C:/Users/Admin/Desktop/Recruitment_AI_assistant/frontend/src/routes/dashboard.tsx) from recent resume uploads and outreach drafts.
- `Notifications` are backed by `user_notifications` and currently represent alert-style events such as candidate applications, scoring completion, and interview completion.
- The two concepts overlap slightly today, but they are not modeled as the same thing.

## Decision

Introduce a dedicated activity feed for the dashboard via a new backend endpoint and service.

- `Recent Activity` becomes an operational timeline.
- `Notifications` remain an alert feed.
- The dashboard consumes the new activity endpoint instead of assembling mixed activity client-side.

## Activity Rules

Include only events with clear operational value and trustworthy timestamps:

- resume uploaded
- resume processed
- resume processing failed
- candidate added to shortlist
- outreach email sent
- outreach send failed
- interview link created
- interview invitation sent
- interview completed
- interview cancelled
- scoring completed

Exclude low-signal noise:

- outreach drafts that were only created but not sent
- generic UI interactions
- duplicate alert-only entries already better represented by richer domain events

## Anti-Noise Rules

- If a resume already has a terminal event (`processed` or `failed`), suppress the raw upload event for that same resume.
- If an interview invitation has `sent_at`, prefer the sent event over the plain created event for that invitation.
- Merge events from multiple domains, sort by timestamp descending, then truncate to the requested limit.

## Backend Shape

Add a new authenticated endpoint, likely `/api/v1/activities/`, with optional `job_id` and `limit`.

Each item should return structured data rather than preformatted UI strings:

- `id`
- `kind`
- `timestamp`
- `subject_name`
- `context_name`
- `status`
- `target_url`
- `metadata`

This keeps localization and visual treatment in the frontend.

## Frontend Shape

Update the dashboard route to:

- fetch the activity feed from the backend
- map `kind` to icon, accent color, title, and subtitle
- render clickable items when `target_url` is present
- keep the current empty state and loading skeleton behavior

## Testing

- backend endpoint test for merged ordering and filtering
- backend test for de-noising rules around resume and interview events
- frontend typecheck/build verification for the dashboard integration
