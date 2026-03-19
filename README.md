# Recruitment AI Assistant

Recruitment AI Assistant is a full-stack platform for resume ingestion, candidate normalization, job matching, conversational filtering, and recruiter follow-up workflows.

## Architecture

### System Components

- Backend API: FastAPI service in `backend/src/api` for contract endpoints and RBAC-protected actions.
- Worker: background jobs in `backend/worker` for ingestion and retention workflows.
- Frontend: React + Vite application in `frontend/src` for recruiter UI flows.
- Data Stores: PostgreSQL for relational entities and MinIO for raw resume object storage.
- AI Layer: LLM adapter with Groq default and Ollama fallback in `backend/src/services/llm`.

### Backend Module Boundaries

- API routes in `backend/src/api/routes` provide request/response contracts.
- Services in `backend/src/services` handle business logic, observability, and integrations.
- Repositories in `backend/src/repositories` contain data access and query behavior.
- Agents in `backend/src/agents` handle natural-language route planning and tool execution.
- Orchestration in `backend/src/orchestration` defines graph execution patterns.

### Request and Job Flow

1. Recruiter calls API endpoints (upload, match, query, shortlist, outreach, interview).
2. API validates role access, persists or reads relational data, and delegates to services.
3. Worker handles long-running operations where applicable.
4. Observability writes masked audit logs and request latency metrics.

## Operating Runbook

### 1. Prerequisites

- Docker Desktop or Docker Engine + Compose plugin
- Backend environment file at `backend/.env`
- Frontend environment file at `frontend/.env`

### 2. Start/Stop/Inspect

Use the maintenance helper:

```powershell
./scripts/maintenance/dev.ps1 -Action up
./scripts/maintenance/dev.ps1 -Action logs
./scripts/maintenance/dev.ps1 -Action migrate
./scripts/maintenance/dev.ps1 -Action down
```

### 3. Health and Metrics

- Health endpoint: `GET /health`
- Metrics endpoint: `GET /metrics`
- Timing header: all responses include `X-Response-Time-Ms`

### 4. RBAC and Security Checks

- Role header used by guards: `X-Role` with values `admin`, `recruiter`, `viewer`.
- Invalid role values are downgraded to `viewer`.
- In production mode (`APP_ENV=production`), startup validates secret configuration.

### 5. Quickstart Validation

Run quickstart checks after stack startup:

```powershell
./scripts/maintenance/validate_quickstart.ps1
```

The script verifies:

- Docker services are running
- API health and metrics endpoints are reachable
- Required backend and frontend environment variables are present

### 6. Troubleshooting

- If migrations fail, run `./scripts/maintenance/dev.ps1 -Action migrate` after `up`.
- If LLM calls fail, verify provider env vars (`LLM_PROVIDER`, `GROQ_MODEL`, `OLLAMA_MODEL`) and credentials.
- If MinIO upload fails, verify endpoint and access keys in `backend/.env`.
- If role-based actions are blocked, verify `X-Role` and endpoint-level role requirements.

### 7. Production Hardening Checklist

- Set `APP_ENV=production`.
- Configure non-placeholder secrets for API, storage, and SMTP credentials.
- Restrict ingress and terminate TLS at edge or reverse proxy.
- Enable centralized log collection while preserving PII masking policies.