# Shortlist Dispatch Design

## Goal

Turn a shortlist collection into a dispatch workspace where recruiters can select candidates, create outreach drafts, and send interview invitations without losing candidate context or duplicating communication.

## Scope

The MVP uses staged batch actions:

- Create outreach drafts for selected shortlist candidates.
- Send interview invitations for selected shortlist candidates.
- Show per-candidate readiness and latest status before action.
- Skip blocked or duplicate candidates by default.

This design does not add a full candidate pipeline stage model. Existing `OutreachMessage` and `InterviewInvitation` records remain the source of truth for communication status.

## Frontend UX

The shortlist collection detail page gains selectable candidate rows and two primary actions:

- `Create outreach drafts`
- `Send interview invites`

Rows display:

- candidate identity and email readiness
- latest outreach status
- latest interview invitation status
- blockers such as missing email, missing Gmail permission, or missing interview template

Batch actions open a review modal before the final command. The modal shows eligible candidates, skipped candidates, and blockers. After confirmation, it shows per-candidate results and refreshes the summary.

## Backend API

Add dispatch endpoints under the existing shortlist route:

- `GET /shortlist/collections/{collection_id}/dispatch-summary`
- `POST /shortlist/collections/{collection_id}/outreach-drafts`
- `POST /shortlist/collections/{collection_id}/interview-invitations`

The summary endpoint returns collection metadata, candidate snapshots, latest outreach status, latest interview status, and capability information.

Batch endpoints accept selected candidate IDs and return per-candidate results. They skip missing email and duplicate communication by default.

## Data Rules

- A candidate must belong to the shortlist collection before a batch command can act on them.
- Outreach draft creation skips candidates that already have a non-failed outreach message unless `force_update` is true.
- Interview invitation creation skips candidates that already have an active or completed invitation for the selected job/template.
- Sending interview invitations requires an active interview template and candidate email.
- Gmail connection blocks sending email, but does not block creating outreach drafts.

## Testing

Backend tests cover:

- dispatch summary includes candidate, latest outreach, latest interview, and blockers
- outreach draft batch creates drafts only for eligible selected candidates
- outreach draft batch skips duplicates and missing email
- interview invitation batch creates invitations only for eligible selected candidates
- interview invitation batch skips duplicates, missing email, and missing template

Frontend verification covers TypeScript build and existing lint/test commands where available.
