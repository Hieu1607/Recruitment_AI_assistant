# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Stack Overview

Recruitment AI Assistant is a multi-service app that ingests PDF resumes, extracts structured candidate profiles via an LLM, scores candidates against job descriptions, and exposes a recruiter chat interface.

- **Backend**: FastAPI (Python), SQLAlchemy + Alembic, Celery worker, LangGraph-based chat agent
- **Frontend**: React 18 + TypeScript + Vite
- **Infra**: PostgreSQL 15, Redis (broker + Celery backend)
- **LLM**: Pluggable provider via `LLM_PROVIDER` env (`groq` default, `ollama` supported). CV parsing uses `LLMProvider` in [backend/src/services/llm_service.py](backend/src/services/llm_service.py).

Everything is wired through [docker-compose.yml](docker-compose.yml). See [QUICKSTART.md](QUICKSTART.md) for the full bring-up flow — the important caveat is that tables are **not** auto-created; Alembic migrations must be run once after first boot, and a seed script creates the initial admin and roles.

## Common Commands

All commands assume the Docker stack is running.

```bash
# Bring up / rebuild the stack
docker compose up --build -d

# Run DB migrations (required after first boot and after adding a revision)
docker compose exec backend alembic -c alembic.ini upgrade head

# Create a new migration revision (autogenerate from SQLAlchemy models)
docker compose exec backend alembic -c alembic.ini revision --autogenerate -m "message"

# Seed the initial admin user + roles
docker compose exec -T backend python seeds/seed_initial_user_and_roles.py

# Tail logs
docker compose logs -f backend
docker compose logs -f worker

# Reset DB volume (destroys data)
docker compose down -v
```

Backend tests use `pytest` (listed in [backend/requirements.txt](backend/requirements.txt)) — run inside the backend container, e.g. `docker compose exec backend pytest path/to/test_file.py::test_name`.

Frontend scripts (run from [frontend/](frontend/)):

```bash
npm run dev      # Vite dev server on :5173
npm run build    # tsc && vite build
npm run lint     # eslint with --max-warnings 0
```

## Backend Architecture

Entry point is [backend/src/main.py](backend/src/main.py); the FastAPI app mounts a single v1 router from [backend/src/api/v1/api.py](backend/src/api/v1/api.py) under `/api/v1`. Current endpoint groups:

- `upload` → [endpoints/resume.py](backend/src/api/v1/endpoints/resume.py) — PDF batch parse (`POST /api/v1/upload/batch-parse`)
- `job-descriptions` → [endpoints/jobDescription.py](backend/src/api/v1/endpoints/jobDescription.py)
- `score` → [endpoints/score.py](backend/src/api/v1/endpoints/score.py)
- `chat` → [endpoints/chat.py](backend/src/api/v1/endpoints/chat.py) — recruiter query chatbot

Layering:

- **API endpoints** are thin — they delegate to services in [backend/src/services/](backend/src/services/).
- **Services** encapsulate business logic: `resume_service` (PDF extraction via PyMuPDF + LLM parsing), `job_description_service`, `score_candidate`, `query_service`, `mail_service`, and `llm_service` (provider-agnostic LLM wrapper reading `LLM_PROVIDER`).
- **Models** in [backend/src/models/](backend/src/models/) are SQLAlchemy ORM definitions. [models/entities.py](backend/src/models/entities.py) re-exports all models for Alembic autogeneration — **new models must be added to this file or Alembic will miss them**. [models/base.py](backend/src/models/base.py) holds the declarative base; [models/session.py](backend/src/models/session.py) the engine/session; [models/deps.py](backend/src/models/deps.py) the FastAPI `get_db` dependency.
- **Schemas** ([backend/src/schemas/ai_schema.py](backend/src/schemas/ai_schema.py)) are Pydantic models for LLM-structured output and API payloads.
- **Prompts** live in [backend/src/prompts/build_prompts.py](backend/src/prompts/build_prompts.py) — keep prompt strings centralized here, not inline in services.

### Chat agent (LangGraph)

The recruiter chatbot is a compiled LangGraph workflow defined in [backend/src/services/ai_agent/graph.py](backend/src/services/ai_agent/graph.py), with node implementations in [backend/src/services/ai_agent/nodes.py](backend/src/services/ai_agent/nodes.py). Topology:

```
START → trim → router ─┬─► dsl ─┬─► llm ─► answer → END
                       │        └─────────► answer → END
                       ├─────────────────► llm ──── answer → END
                       └─────────────────────────── answer → END
```

- `trim_node` caps history at `MEMORY_WINDOW = 5` messages.
- `router_node` decides whether the turn needs DSL filtering, an LLM lookup, or a direct answer, by populating `router_output.dsl_question_query` and/or `router_output.llm_question_query`.
- `dsl_node` narrows candidates via a structured DSL filter; `llm_node` answers free-form questions; `answer_node` produces final text.
- The compiled `graph` is module-level — import it directly; don't rebuild per request.

State is a `TypedDict` (`GraphState`) — when editing nodes, return **partial** dicts containing only the fields you changed; LangGraph will merge them.

### Async work (Celery)

Worker app is [backend/worker/celery_app.py](backend/worker/celery_app.py); task modules live in [backend/worker/tasks.py](backend/worker/tasks.py). Broker and result backend both point at `REDIS_URL`. All tasks are routed to the `default` queue via `task_routes`.

## Data Model

The authoritative reference is [docs/data-model.md](docs/data-model.md). Key concepts:

- **ResumeDocument** (1-to-1) → **CandidateProfile** — one uploaded PDF yields one normalized profile. `retention_expires_at = uploaded_at + 12 months`.
- **JobDescription** → **MatchRun** → **MatchResult** — one scoring execution (`MatchRun`) evaluates many candidates against a single JD; each `MatchResult` carries `component_scores` (criterion_key/weight/score/weighted_score) and a `total_score` normalized to 0..100.
- **QuerySession** / **QueryTurn** — persist recruiter chat history; `QueryTurn.matched_candidate_ids` captures the narrowed candidate set for a turn.
- **ShortlistCollection** / **ShortlistItem** — recruiter-saved candidate sets, often sourced from a `QueryTurn`.
- **OutreachMessage**, **InterviewQuestionSet** — AI-generated artifacts tied to candidate (+ JD for interviews).
- **UserAccount** / **RoleAssignment** — RBAC with roles `admin`, `recruiter`, `viewer`.

## Conventions & Gotchas

- **PDF-only uploads**: `/upload/batch-parse` rejects non-`.pdf` files — don't add generic file support without updating validation and the retention rules above.
- **LLM provider is env-switched**: code paths go through `llm_service`; never hard-code Groq/Ollama clients in services or endpoints. `GROQ_API_KEY` is required when `LLM_PROVIDER=groq` (backend will error on boot otherwise).
- **Alembic doesn't run at startup** — migrations are an explicit step. When a test or dev action hits `relation "X" does not exist`, run `alembic upgrade head` before debugging further.
- **Backend bind-mounts `./backend` into the container**, so edits hot-reload via uvicorn; the worker does not auto-reload — restart it (`docker compose restart worker`) after changing task code.
- **CORS** origins come from `BACKEND_CORS_ORIGINS` (comma-separated or JSON list) in [core/config.py](backend/src/core/config.py); empty by default, so the frontend at `:5173` must be explicitly added for browser calls.

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
5. Don't touch to frontend_backup

<!-- gsd:start -->
## GSD Workflow

This project uses [GSD](https://github.com/dnathe4th/get-shit-done-cc) for structured agentic development. The active milestone is **frontend build** — backend is shipped, this milestone delivers the 15-screen recruiter UI.

### Planning artifacts (`.planning/`)

| File | Purpose |
|------|---------|
| `PROJECT.md` | Project vision, validated capabilities (backend), active scope (frontend), key decisions, constraints |
| `REQUIREMENTS.md` | 117 v1 requirements grouped by category (FOUND, PRIM, CAND, JD, SCORE, CHAT, SHORT, OUT, INTV, DASH, MKTG, AUTH, PLAT) |
| `ROADMAP.md` | 12-phase breakdown with goals, dependencies, requirement mapping, success criteria |
| `STATE.md` | Project memory — current phase, blockers, recent work |
| `config.json` | Workflow config: mode=yolo, granularity=fine, all quality agents on, balanced model profile |

### Workflow rules

- **Read `.planning/STATE.md` first** when resuming work to know which phase is active and what blockers exist.
- **Phase artifacts** live in `.planning/phases/NN-slug/` and are created on demand by `/gsd-plan-phase N`. Format: `NN-MM-PLAN.md`, `NN-MM-SUMMARY.md`, `CONTEXT.md`, etc.
- **Commit per phase, not per file**: GSD's `gsd-tools commit` writes a single atomic commit per planning step (PROJECT.md, config, REQUIREMENTS.md, ROADMAP.md+STATE.md+CLAUDE.md, then per-phase plans/summaries).
- **No new backend endpoints** — frontend works against the API documented in `docs/BACKEND.md`. If a screen needs data the backend doesn't expose, scope it down or ship UI-only stub.
- **Frontend lives in `frontend/`** — empty as of this milestone start. Old code in `frontend_backup/` is reference-only; do not touch.

### Common GSD commands

```bash
/gsd-progress              # Where am I, what's next?
/gsd-plan-phase 1          # Generate Phase 1 plan
/gsd-execute-phase 1       # Execute Phase 1 (parallel waves)
/gsd-discuss-phase 5       # Clarify a fuzzy phase before planning
/gsd-ui-phase 5            # Generate UI design contract for a frontend phase
/gsd-resume-work           # Pick up after /clear
```
<!-- gsd:end -->

