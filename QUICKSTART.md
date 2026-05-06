# QUICKSTART

## 1. Prerequisites

- Docker Desktop (Windows) with Docker Compose enabled
- At least 4GB RAM available for containers
- A Groq API key if you want CV parsing via LLM

## 2. Start The Stack

From project root:

```powershell
docker compose up --build -d
```

Services started by [docker-compose.yml](docker-compose.yml):

- `db` (PostgreSQL) on `localhost:5432`
- `redis` on `localhost:6379`
- `backend` (FastAPI) on `localhost:8000`
- `worker` (Celery)
- `frontend` (Vite) on `localhost:5173`

## 3. Configure LLM For CV Parsing

Endpoint parse CV uses `LLMProvider` in backend. By default provider is `groq`, so `GROQ_API_KEY` is required.

Update [docker-compose.yml](docker-compose.yml) and add env vars under `backend` and `worker` services:

```yaml
environment:
	- DATABASE_URL=postgresql://${POSTGRES_USER:-postgres}:${POSTGRES_PASSWORD:-postgres}@db:5432/${POSTGRES_DB:-recruitment_db}
	- REDIS_URL=redis://redis:6379/0
	- LLM_PROVIDER=groq
	- GROQ_API_KEY=your_groq_api_key_here
	- GROQ_MODEL_NAME=openai/gpt-oss-20b
```

After editing, restart:

```powershell
docker compose up -d --build
```

## 4. Run Database Migration

Tables are not auto-created at startup. Run Alembic migration once:

```powershell
docker compose exec backend alembic -c alembic.ini upgrade head
```

## 5. Access Apps

- Backend API root: http://localhost:8000/
- Swagger UI: http://localhost:8000/docs
- Frontend: http://localhost:5173/

## 6. Seed Initial User And Roles

Run seed script after first successful migration:

```powershell
docker compose exec -T backend python seeds/seed_initial_user_and_roles.py
```

Default seeded user:

- email: `admin@recruitment.local`
- display_name: `Initial Admin`
- roles: `admin,recruiter,viewer`

## 7. Test Batch Resume Parsing

### Option A: Swagger UI

1. Open http://localhost:8000/docs
2. Find `POST /api/v1/upload/batch-parse`
3. Click `Try it out`
4. In `files`, select multiple PDF files (Ctrl/Shift)
5. Optional: fill `uploaded_by_user_id` with UUID
6. Execute

### Option B: cURL

```bash
curl -X POST "http://localhost:8000/api/v1/upload/batch-parse" \
	-F "files=@C:/path/to/cv1.pdf" \
	-F "files=@C:/path/to/cv2.pdf"
```

Sample response:

```json
{
	"total_files": 2,
	"processed_files": 2,
	"failed_files": 0,
	"items": [
		{
			"file_name": "cv1.pdf",
			"resume_document_id": "...",
			"candidate_profile_id": "...",
			"status": "processed"
		}
	]
}
```

## 8. Useful Commands

Show logs:

```powershell
docker compose logs -f backend
docker compose logs -f worker
```

Stop all services:

```powershell
docker compose down
```

Stop and remove volumes (clean DB):

```powershell
docker compose down -v
```

## 9. Common Issues

- `GROQ_API_KEY is required when LLM_PROVIDER=groq`
	- Add `GROQ_API_KEY` in [docker-compose.yml](docker-compose.yml) for `backend` and restart.

- `relation "resume_documents" does not exist`
	- Run migration command in section 4.

- Upload fails for non-PDF
	- Endpoint only accepts `.pdf` files.
