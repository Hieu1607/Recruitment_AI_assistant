# Job Architecture Test Plan

## Backend
- Run migration against an existing database snapshot.
- Verify each user gets a backfilled default job.
- Verify JD/resume rows receive `job_id`.
- Verify cross-user access to jobs, JDs, resumes, scoring, and chat returns `404`.

## Frontend
- New authenticated user with zero jobs sees create-job flow first.
- After job creation, the first job becomes selected automatically.
- Uploading resumes stores them under the selected job.
- Editing the JD updates the selected job’s single current JD.
- Scoring and chat only use the selected job’s candidates.

## Smoke Checks
- Dashboard loads with selected-job metrics.
- Candidate list reflects job switching.
- Legacy screens still function through selected-job-backed compatibility paths.
