# Job Architecture Plan

## Goal
Move the product to a job-first workspace model where every authenticated user works inside owned `Job` records, with one current JD and a job-owned resume set per job.

## Milestones
1. Additive schema rollout
2. Backfill existing JD/resume rows into default jobs
3. Job-scoped backend endpoints and ownership helpers
4. Frontend selected-job state and first-run create-job guard
5. Compatibility pass for legacy endpoints and screens

## Dependencies
- `jobs` table must land before nested API rollout.
- `job_id` backfill must complete before strict owner scoping can depend on jobs.
- Frontend selected-job state depends on `GET /jobs`.

## Acceptance Gates
- Authenticated users can create a job before JD/resume actions.
- JD and resume listings are job-scoped by default.
- Scoring and chat operate only on the selected job’s data.
- Legacy endpoints no longer trust client-supplied user IDs for authorization.
