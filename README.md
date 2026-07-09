# Recruitment AI Assistant

Recruitment AI Assistant is a full-stack hiring workspace for parsing resumes, scoring candidates against job descriptions, searching the candidate pool with AI, managing shortlists, sending outreach, and preparing interviews.

The repository contains:

- `backend/`: FastAPI API, Celery workers, Alembic migrations, and AI services
- `frontend/`: React + Vite recruiter application
- `docs/`: implementation notes, deployment guides, API references, and design docs
- `logs/`: runtime traces useful when debugging scoring, PDF parsing, and chat flows

## Core Capabilities

- Batch upload and parse PDF resumes into structured candidate profiles
- Create and manage job descriptions
- Score candidates against a job description with configurable section weights and pass thresholds
- Query the candidate pool through an AI chat workflow
- Save recruiter query history and shortlist collections
- Draft and send outreach through Gmail OAuth
- Generate interview questions and manage interview-related flows
- Run the stack locally with Docker Compose

## Architecture

### Backend

- FastAPI app served from `backend/src/main.py`
- PostgreSQL for application data
- Redis as Celery broker/backend support
- MinIO for object storage
- Celery workers for resume parsing, candidate evaluation, and scheduled work
- LLM integrations configured through environment variables

### Frontend

- React 18 + TypeScript + Vite
- TanStack Query for server-state management
- React Router for app routing
- Route groups for dashboard, candidates, scoring, chat, shortlists, outreach, interviews, and settings

### Local Services

`docker-compose.yml` starts these services by default:

- `db` on `localhost:5432`
- `redis` on `localhost:6379`
- `minio` on `localhost:9000`
- `minio` console on `localhost:9001`
- `backend` on `localhost:8000`
- `frontend` on `localhost:5173`
- `worker`, `resume-worker`, `evaluation-worker`, and `beat`

## Quick Start

### 1. Prerequisites

- Docker Desktop with Docker Compose
- At least 4 GB RAM available for containers
- An LLM provider key if you want AI-backed parsing, scoring, or chat

### 2. Create `.env`

```powershell
Copy-Item .env.example .env
```

Important variables to review:

- `SHOPAIKEY_API_KEY`
- `LLM_PROVIDER`
- `APP_UI_LANGUAGE`
- `VITE_API_BASE_URL`
- `VITE_UI_LANGUAGE`
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_TOKEN_ENCRYPTION_KEY`
- `GMAIL_SEND_ENABLED`

### 3. Start the development stack

```powershell
docker compose up --build -d
```

This runs the backend with auto-reload and the frontend with the Vite dev server.

### 4. Open the app

- Frontend: `http://localhost:5173`
- Backend root: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- MinIO console: `http://localhost:9001`

### 5. Create the first user

After the stack is healthy and migrations have run, create the first account through the app auth flow:

- `POST /api/v1/auth/register`
- or Google OAuth login if configured

This repository no longer includes a checked-in seed script for initial users.

## Production-Like Frontend Mode

To run the frontend as a built bundle served by `nginx`:

```powershell
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

This override is frontend-only. The backend and supporting services still come from [docker-compose.yml](/C:/Users/Admin/Desktop/Recruitment_AI_assistant/docker-compose.yml:1).

## Useful Commands

```powershell
docker compose logs -f backend
docker compose logs -f worker
docker compose logs -f frontend
docker compose exec backend alembic -c alembic.ini upgrade head
docker compose down
docker compose down -v
```

## Environment Notes

- The canonical local environment file is the repo-root `.env`
- `backend` and worker services load `./.env` through Docker Compose
- Frontend localization is driven by `VITE_UI_LANGUAGE`
- Gmail sending requires Google OAuth configuration and token encryption
- Batch resume parsing and some AI flows can be long-running; check `logs/` when debugging

## Project Structure

```text
backend/
  src/
  migrations/
frontend/
  src/
  tests/
docs/
logs/
docker-compose.yml
docker-compose.prod.yml
.env.example
QUICKSTART.md
```

## Documentation

- Full local setup: [QUICKSTART.md](/C:/Users/Admin/Desktop/Recruitment_AI_assistant/QUICKSTART.md:1)
- Backend API reference: [docs/BACKEND.md](/C:/Users/Admin/Desktop/Recruitment_AI_assistant/docs/BACKEND.md:1)
- Deployment guide: [docs/DEPLOY_1VM_SIMPLE_GUIDE.md](/C:/Users/Admin/Desktop/Recruitment_AI_assistant/docs/DEPLOY_1VM_SIMPLE_GUIDE.md:1)
- Google OAuth and Gmail setup: [docs/GOOGLE_OAUTH_GMAIL_API_SETUP.md](/C:/Users/Admin/Desktop/Recruitment_AI_assistant/docs/GOOGLE_OAUTH_GMAIL_API_SETUP.md:1)
- Frontend screen spec: [docs/FRONTEND_SCREENS.md](/C:/Users/Admin/Desktop/Recruitment_AI_assistant/docs/FRONTEND_SCREENS.md:1)
- Job architecture notes: [docs/JOB_ARCHITECTURE_BACKEND.md](/C:/Users/Admin/Desktop/Recruitment_AI_assistant/docs/JOB_ARCHITECTURE_BACKEND.md:1), [docs/JOB_ARCHITECTURE_FRONTEND.md](/C:/Users/Admin/Desktop/Recruitment_AI_assistant/docs/JOB_ARCHITECTURE_FRONTEND.md:1), [docs/JOB_ARCHITECTURE_PLAN.md](/C:/Users/Admin/Desktop/Recruitment_AI_assistant/docs/JOB_ARCHITECTURE_PLAN.md:1)

## Troubleshooting

- Missing AI responses: verify provider credentials in `.env`
- Upload failures for resumes: only PDF files are accepted
- Missing database tables: rerun Alembic migrations
- Gmail send failures: verify Google OAuth consent, scopes, and encryption key setup
- Slow parsing or scoring: inspect container logs and files under `logs/`
