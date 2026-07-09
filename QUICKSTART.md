# QUICKSTART

## 1. Prerequisites

- Docker Desktop with Docker Compose enabled
- At least 4 GB RAM available for containers
- A ShopAIKey API key if you want CV parsing, scoring, or chat features backed by LLM

## 2. Prepare Environment

Create `.env` at repo root from the single canonical `.env.example` and update the values you need:

```powershell
Copy-Item .env.example .env
```

Important variables:

- `SHOPAIKEY_API_KEY`
- `SHOPAIKEY_MODEL_NAME`
- `APP_UI_LANGUAGE`
- `VITE_API_BASE_URL`
- `VITE_UI_LANGUAGE`
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_TOKEN_ENCRYPTION_KEY`
- `GMAIL_SEND_ENABLED`

Notes:

- `docker-compose.yml` already loads `./.env` for `backend` and `worker`.
- Use this same root `.env` as the source of truth for shared, backend, and frontend env values.
- Frontend language is also read from `.env` via `VITE_UI_LANGUAGE`.

## 3. Start In Dev Mode

Use this for local development with hot reload:

```powershell
docker compose up --build -d
```

What this starts:

- `db` on `localhost:5432`
- `redis` on `localhost:6379`
- `minio` on `localhost:9000` and console `localhost:9001`
- `backend` on `localhost:8000` with `uvicorn --reload`
- `worker` as Celery worker
- `frontend` on `localhost:5173` with Vite dev server

Dev mode behavior:

- backend source is bind-mounted and auto-reloads
- frontend source is bind-mounted and runs `npm run dev`
- this is the default mode for coding

## 4. Start In Prod-Like Mode

Use this when you want the frontend to run as a built production bundle served by `nginx`:

```powershell
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

What changes in this mode:

- `frontend` is built from the `production` stage in [frontend/Dockerfile](/C:/Users/Admin/Desktop/Recruitment_AI_assistant/frontend/Dockerfile:1)
- Vite dev server is replaced by `nginx`
- frontend is still available at `http://localhost:5173`

Important limitation:

- In this repository, `docker-compose.prod.yml` only overrides `frontend`.
- `backend`, `worker`, `db`, `redis`, and `minio` still come from [docker-compose.yml](/C:/Users/Admin/Desktop/Recruitment_AI_assistant/docker-compose.yml:1).
- That means this is best described as a `prod-like frontend` setup, not a full production runtime override for the whole stack.

## 5. Run Database Migration

The backend service already runs Alembic on startup, but you can run it manually if needed:

```powershell
docker compose exec backend alembic -c alembic.ini upgrade head
```

If you started with the prod override, use the same override when executing commands:

```powershell
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec backend alembic -c alembic.ini upgrade head
```

Database bootstrap note:

- This repository does not use `docker-entrypoint-initdb.d` or checked-in Postgres init SQL files.
- Schema setup is handled by Alembic migrations.
- User accounts are created through the authentication flows after the stack is up.

## 6. Access The App

Dev mode:

- Backend API root: `http://localhost:8000/`
- Swagger UI: `http://localhost:8000/docs`
- Frontend: `http://localhost:5173/`
- MinIO console: `http://localhost:9001/`

Prod-like mode:

- Backend API root: `http://localhost:8000/`
- Swagger UI: `http://localhost:8000/docs`
- Frontend: `http://localhost:5173/`

## 7. Configure Gmail Sending

To send candidate emails from a real Gmail account, follow [docs/GOOGLE_OAUTH_GMAIL_API_SETUP.md](/C:/Users/Admin/Desktop/Recruitment_AI_assistant/docs/GOOGLE_OAUTH_GMAIL_API_SETUP.md:1).

Minimum related settings:

- `GMAIL_SEND_ENABLED=true`
- `GOOGLE_TOKEN_ENCRYPTION_KEY`
- Google OAuth consent screen with `https://www.googleapis.com/auth/gmail.send`

## 8. Create Your First User

After the first successful migration, create the first account through one of the supported auth flows:

- Register with `POST /api/v1/auth/register`
- Sign in with Google OAuth if it is configured

The backend no longer ships a checked-in seed script for initial users.

## 9. Test Batch Resume Parsing

### Option A: Swagger UI

1. Open `http://localhost:8000/docs`
2. Find `POST /api/v1/upload/batch-parse`
3. Click `Try it out`
4. In `files`, select multiple PDF files
5. Optional: fill `uploaded_by_user_id` with UUID
6. Execute

### Option B: cURL

```bash
curl -X POST "http://localhost:8000/api/v1/upload/batch-parse" \
  -F "files=@C:/path/to/cv1.pdf" \
  -F "files=@C:/path/to/cv2.pdf"
```

## 10. Useful Commands

Show logs in dev mode:

```powershell
docker compose logs -f backend
docker compose logs -f worker
docker compose logs -f frontend
```

Show logs in prod-like mode:

```powershell
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f backend
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f worker
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f frontend
```

Stop services:

```powershell
docker compose down
```

Stop prod-like stack:

```powershell
docker compose -f docker-compose.yml -f docker-compose.prod.yml down
```

Stop and remove volumes:

```powershell
docker compose down -v
```

## 11. Common Issues

- `SHOPAIKEY_API_KEY is required for ShopAIKey fallback`
  Set `SHOPAIKEY_API_KEY` in `.env`, then rebuild or restart the stack.

- `relation "... " does not exist`
  Run the migration command in section 5.

- Upload fails for non-PDF files
  The batch parse endpoint only accepts `.pdf`.

- Frontend in prod-like mode still looks like dev elsewhere in the stack
  That is expected because [docker-compose.prod.yml](/C:/Users/Admin/Desktop/Recruitment_AI_assistant/docker-compose.prod.yml:1) only overrides `frontend`.

- Need full VM deployment guidance
  Use [docs/DEPLOY_1VM_SIMPLE_GUIDE.md](/C:/Users/Admin/Desktop/Recruitment_AI_assistant/docs/DEPLOY_1VM_SIMPLE_GUIDE.md:1).
