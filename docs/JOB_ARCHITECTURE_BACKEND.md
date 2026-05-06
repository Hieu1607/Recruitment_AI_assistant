# Job Architecture Backend

## Schema
- New `jobs` table: `id`, `owner_user_id`, `title`, `status`, `created_at`, `updated_at`, `archived_at`.
- Add `job_id` to `job_descriptions`.
- Add `job_id` to `resume_documents`.
- Legacy ownership columns remain for compatibility and backfill traceability.

## Ownership Rules
- JWT user is the only source of acting identity.
- A user may access only jobs where `jobs.owner_user_id == current_user.id`.
- Candidate ownership remains indirect through `resume_documents.job_id`.

## Preferred Endpoints
- `POST /api/v1/jobs`
- `GET /api/v1/jobs`
- `GET|PATCH|DELETE /api/v1/jobs/{job_id}`
- `GET|POST|PATCH /api/v1/jobs/{job_id}/job-description`
- `GET|POST /api/v1/jobs/{job_id}/resumes`
- `GET|PATCH|DELETE /api/v1/jobs/{job_id}/resumes/{resume_id}`
- `GET /api/v1/jobs/{job_id}/candidates`
- `POST /api/v1/jobs/{job_id}/score`
- `POST /api/v1/jobs/{job_id}/chat`

## Migration Notes
- Create a default job per existing user.
- Backfill JDs by `created_by_user_id`.
- Backfill resumes by `uploaded_by_user_id`.
- Make `job_id` non-null after backfill.
