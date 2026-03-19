# Quickstart: Recruitment AI Assistant Website

## 1. Prerequisites
- Docker Desktop 4.0+ (or Docker Engine 24+)
- Docker Compose v2+
- One LLM provider option available:
	- Groq API account (default path), or
	- Local Ollama runtime for self-hosted models

## 2. Configure environment
Create environment files:
- backend/.env
- frontend/.env

Minimum backend variables:
- DATABASE_URL
- LLM_PROVIDER=groq
- GROQ_API_KEY
- GROQ_MODEL
- OLLAMA_BASE_URL
- OLLAMA_MODEL
- MINIO_ENDPOINT
- MINIO_ACCESS_KEY
- MINIO_SECRET_KEY
- MINIO_BUCKET
- MINIO_USE_SSL
- SMTP_HOST
- SMTP_PORT
- SMTP_USERNAME
- SMTP_PASSWORD
- APP_BASE_URL
- BACKEND_PORT
- FRONTEND_PORT

Minimum frontend variables:
- VITE_API_BASE_URL

## 3. Start full stack (default: Docker)
```bash
docker compose up -d --build
```

## 4. Verify running services
```bash
docker compose ps
```

Expected services:
- api
- worker
- frontend
- postgres
- minio

## 5. Apply database migrations in API container
```bash
docker compose exec api python -m alembic upgrade head
```

## 6. Execute core recruiter flow
1. Sign in with a Recruiter role account.
2. Upload one or more resume PDFs.
3. Review extracted fields and approve candidate profiles.
4. Add a job description and run matching with weighted component criteria and threshold.
5. Ask natural-language filter questions and inspect returned candidate details.
6. Save filtered set as a shortlist collection.
7. Draft outreach email and approve before send.
8. Generate interview questions for a selected candidate + JD pair.

## 7. Automated quickstart validation
Run the validation script after the stack is healthy:

```powershell
./scripts/maintenance/validate_quickstart.ps1
```

What the script validates:
- Required Docker Compose services are present and running (`api`, `worker`, `frontend`, `postgres`, `minio`).
- Required backend/frontend environment keys are present in `backend/.env` and `frontend/.env`.
- API health endpoint returns `status=ok` at `GET /health`.
- API metrics endpoint returns a valid payload at `GET /metrics`.

## 8. Validation checklist (manual spot checks)
- Uploading 20 resumes should complete extraction pipeline within target time.
- Matching output must return component score list plus total 0-100 score and threshold pass/fail for each candidate.
- Query answers must provide matched count and candidate set when applicable.
- Viewer role must be blocked from edit/send actions.
- Outbound email must fail if approval status is not approved.
- Logs must not contain full PII or raw CV dumps.
- Every API response should include `X-Response-Time-Ms`.
- `GET /metrics` should expose `requestCount`, `p95LatencyMs`, and per-route aggregates.

## 9. Cleanup and retention test
- Verify retention scheduler marks/anonymizes records older than 12 months.
- Verify linked shortlist and query history remain auditable without exposing removed PII.

## 10. Optional local (non-Docker) fallback
If Docker is unavailable, run services manually with Python 3.11+, Node.js 20+, and PostgreSQL 15+:
- Backend API:
```bash
cd backend
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
alembic upgrade head
uvicorn src.main:app --reload --port 8000
```
- Worker:
```bash
cd backend
. .venv/Scripts/Activate.ps1
python -m worker.main
```
- Frontend:
```bash
cd frontend
npm install
npm run dev
```
