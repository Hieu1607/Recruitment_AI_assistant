# Job Architecture Frontend

## Selected Job Model
- Persist a single `selectedJobId` in app state plus local storage.
- Load jobs after auth in the shell.
- If the user has zero jobs, block normal navigation with a create-job modal.

## Route Strategy
- Existing routes remain temporarily stable.
- Their data sources resolve through the selected job and the new nested APIs.
- Top bar exposes a job switcher as the single source of truth for active context.

## Screen Mapping
- Dashboard: selected-job metrics only.
- JD editor/list: selected job’s current JD.
- Candidates/upload: selected job’s resumes only.
- Scoring: selected job JD vs selected job candidates only.
- Chat: selected job candidate pool only.
