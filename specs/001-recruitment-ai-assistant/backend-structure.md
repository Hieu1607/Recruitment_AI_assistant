# Backend Structure and File Usage

This document explains the backend architecture and the purpose of each backend file currently in scope.

## Runtime Overview

1. `src/main.py` starts the FastAPI app from `src/api/app.py`.
2. `src/api/routes/*.py` expose HTTP endpoints.
3. Route handlers call repository and service layers.
4. Services use parsing, LLM, storage, and observability modules.
5. SQLAlchemy models in `src/models/*.py` persist to PostgreSQL.
6. Alembic migration files define and evolve schema.
7. Worker loop in `worker/main.py` runs periodic retention tasks.

## Backend Root Files

- `backend/alembic.ini`: Alembic CLI configuration (script location, logging, DB options).
- `backend/pyproject.toml`: Python project metadata and tooling configuration.
- `backend/requirements.txt`: Runtime dependency pins for API and worker containers.

## Migrations

- `backend/migrations/env.py`: Alembic migration environment, engine/session wiring, metadata loading.
- `backend/migrations/versions/0001_initial_schema.py`: Initial schema creation for candidate, matching, query, shortlist, outreach, and audit-related tables.

## Application Entry and API Layer

- `backend/src/__init__.py`: Package marker for `src` module.
- `backend/src/main.py`: Local entrypoint used by `python -m src.main` and development runs.
- `backend/src/api/__init__.py`: API package marker.
- `backend/src/api/app.py`: FastAPI app assembly, CORS config, health and metrics endpoints, middleware, router registration.
- `backend/src/api/errors.py`: Shared app exceptions and global error handlers.

## API Dependencies and Security

- `backend/src/api/dependencies/__init__.py`: Dependency package marker.
- `backend/src/api/dependencies/auth.py`: Role-based request guard (`X-Role`), current-user abstraction.
- `backend/src/api/security/__init__.py`: Security package marker.
- `backend/src/api/security/hardening.py`: Runtime security checks (provider secrets, allowed roles, header normalization).

## API Routes

- `backend/src/api/routes/__init__.py`: Exports route modules for centralized app registration.
- `backend/src/api/routes/resumes.py`: Resume upload endpoint, storage write, extraction/normalization orchestration.
- `backend/src/api/routes/candidates.py`: Candidate listing, profile patch updates, and extraction trace retrieval.
- `backend/src/api/routes/matching.py`: Batch JD-to-CV matching run and score persistence.
- `backend/src/api/routes/query.py`: Conversational query sessions, ask endpoint with SQL/LLM routing and verification.
- `backend/src/api/routes/shortlists.py`: Create and list candidate shortlist collections.
- `backend/src/api/routes/outreach.py`: Draft outreach messages and approve/send flow.
- `backend/src/api/routes/interview_questions.py`: Generate interview question sets per candidate and JD.

## Agents Layer

- `backend/src/agents/__init__.py`: Agents package marker.
- `backend/src/agents/router/__init__.py`: Router subpackage marker.
- `backend/src/agents/router/query_router.py`: Determines query strategy (SQL, LLM semantic, or hybrid).
- `backend/src/agents/tools/__init__.py`: Agent tools package marker.
- `backend/src/agents/tools/sql_search_tool.py`: SQL-backed candidate retrieval for structured query conditions.
- `backend/src/agents/tools/llm_semantic_tool.py`: LLM-assisted semantic retrieval and tracing.
- `backend/src/agents/verifier/__init__.py`: Verifier package marker.
- `backend/src/agents/verifier/query_verifier.py`: Verifies and merges tool outputs into a safe final answer.

## Data Models

- `backend/src/models/__init__.py`: Models package marker.
- `backend/src/models/auth.py`: Role/user auth-related domain types used by route guards.
- `backend/src/models/candidate.py`: Resume document, candidate profile, extraction trace, and parsing status models.
- `backend/src/models/matching.py`: Job description, match run, component score, and run state models.
- `backend/src/models/engagement.py`: Query sessions/turns, shortlists, outreach messages, interview question entities.

## Repository Layer

- `backend/src/repositories/__init__.py`: Repository package marker.
- `backend/src/repositories/db.py`: SQLAlchemy engine/session setup and `get_session` dependency provider.
- `backend/src/repositories/candidate_repository.py`: Candidate/traces query and update data access.
- `backend/src/repositories/match_repository.py`: Match-result retrieval and threshold filtering data access.

## Service Layer

- `backend/src/services/__init__.py`: Services package marker.

### LLM Services

- `backend/src/services/llm/__init__.py`: LLM services package marker.
- `backend/src/services/llm/llm_client.py`: Provider abstraction for Groq/Ollama text generation.

### Parsing Services

- `backend/src/services/parsing/__init__.py`: Parsing package marker.
- `backend/src/services/parsing/resume_extractor.py`: Extracts text blocks from PDFs with OCR fallback support.
- `backend/src/services/parsing/profile_normalizer.py`: Converts raw extraction output into normalized profile fields.
- `backend/src/services/parsing/extraction_trace_service.py`: Persists extraction source-trace mappings.

### Matching Services

- `backend/src/services/matching/__init__.py`: Matching package marker.
- `backend/src/services/matching/batch_prompt_builder.py`: Builds consolidated LLM prompt for batch candidate scoring.
- `backend/src/services/matching/batch_llm_scorer.py`: Executes LLM scoring request and returns raw score payload.
- `backend/src/services/matching/score_list_parser.py`: Validates/parses score payload and persists match results.

### Query, Shortlist, Outreach, Interview Services

- `backend/src/services/query/__init__.py`: Query service package marker.
- `backend/src/services/query/query_history_service.py`: Session and turn persistence for conversational query history.
- `backend/src/services/shortlist/__init__.py`: Shortlist service package marker.
- `backend/src/services/shortlist/shortlist_service.py`: Creates and reads shortlist collections with membership links.
- `backend/src/services/outreach/__init__.py`: Outreach service package marker.
- `backend/src/services/outreach/outreach_service.py`: Draft generation, approval transitions, and send workflow orchestration.
- `backend/src/services/outreach/email_sender.py`: Message delivery adapter used by outreach send operations.
- `backend/src/services/interview/__init__.py`: Interview package marker.
- `backend/src/services/interview/interview_question_service.py`: Generates and stores interview questions from profile and JD context.

### Storage and Observability

- `backend/src/services/storage/__init__.py`: Storage package marker.
- `backend/src/services/storage/minio_client.py`: Object storage upload/download integration for resume files.
- `backend/src/services/observability/__init__.py`: Observability package marker.
- `backend/src/services/observability/audit_logger.py`: Structured audit event logging helper.
- `backend/src/services/observability/metrics.py`: In-memory metrics registry for HTTP and application metrics snapshots.

## Orchestration

- `backend/src/orchestration/__init__.py`: Orchestration package marker.
- `backend/src/orchestration/recruitment_graph.py`: Cross-service workflow orchestration helpers for recruitment pipelines.

## Worker

- `backend/worker/__init__.py`: Worker package marker.
- `backend/worker/main.py`: Long-running retention loop scheduler.
- `backend/worker/jobs/__init__.py`: Worker jobs package marker.
- `backend/worker/jobs/resume_ingestion_job.py`: Resume-to-profile ingestion pipeline used by upload processing.
- `backend/worker/jobs/retention_job.py`: Data retention/anonymization cleanup task execution.

## Notes

- Most `__init__.py` files are package boundary markers and import surfaces.
- API/worker containers install dependencies at startup using `requirements.txt`.
- Authoritative endpoint schema remains `specs/001-recruitment-ai-assistant/contracts/recruitment-api.yaml`.