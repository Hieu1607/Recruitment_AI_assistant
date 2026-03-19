# Implementation Plan: Recruitment AI Assistant Website

**Branch**: `001-recruitment-ai-assistant` | **Date**: 2026-03-18 | **Spec**: `specs/001-recruitment-ai-assistant/spec.md`
**Input**: Feature specification from `/specs/001-recruitment-ai-assistant/spec.md` and technical guidance from `guide2.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Build a web-based recruitment AI assistant that ingests resume PDFs, extracts and normalizes candidate profiles, matches candidates to job descriptions by sending one batch of CVs plus one JD and one shared scoring prompt to the configured LLM, returns compatibility scores as a list where each candidate has weighted component scores plus a total score, answers recruiter questions through a hybrid SQL + LLM agent workflow, supports shortlist persistence, human-approved outreach emails, and interview question generation. The technical approach uses a Python backend with PyMuPDF extraction, LangGraph orchestration, deterministic SQL filters for structured predicates, LLM reasoning for unstructured sections, and an auditable privacy-preserving data model. LLM execution supports two options: Groq-hosted model APIs (default) and self-hosted Ollama models.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.11 (backend), TypeScript 5.x (frontend)  
**Primary Dependencies**: FastAPI, LangGraph, PyMuPDF, Pydantic, SQLAlchemy, PostgreSQL driver, MinIO Python SDK, Groq SDK, Ollama HTTP API integration, React + Vite, component/UI library for professional dashboard styling, Docker, Docker Compose  
**Storage**: PostgreSQL for relational data and audit traces; MinIO object storage for original resumes  
**Testing**: pytest (unit + integration), Playwright (critical end-to-end recruiter flows)  
**Target Platform**: Dockerized Linux containers for API/worker/frontend/postgres/minio with Compose as default deployment; modern desktop browsers for recruiters
**Project Type**: Web application with backend API, async worker, and frontend client  
**Performance Goals**: Resume ingestion and extraction for 20-file batch in <= 5 minutes; recruiter query response p95 <= 3 seconds for SQL routes and <= 8 seconds for hybrid routes; shortlist save and email approval actions <= 2 seconds p95  
**Constraints**: 12-month data retention auto-delete/anonymize; metadata-only masked logs (no full PII in logs); RBAC (Admin/Recruiter/Viewer); explicit human approval before every outbound email; compatibility output must include weighted component score list and total score normalized 0-100 with configurable threshold; maintain explainability and audit traceability; LLM provider selected by env with `groq` as default  
**Scale/Scope**: Initial production scope for small-to-mid recruiting teams (up to ~50 recruiters, ~100k candidate profiles/year, concurrent active sessions < 200)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Clarity over cleverness: design notes MUST explain the simplest viable approach and reject
  unnecessary abstractions.
- Small, focused units: planned modules/components MUST have single responsibilities with clear
  boundaries.
- Behavior-first testing: failing tests or acceptance checks MUST be identified before
  implementation starts.
- Minimal surface area: new dependencies/public APIs MUST include explicit necessity and
  alternatives.
- Refactor before extend: plan MUST list any targeted simplifications required before feature
  expansion.

Pre-Phase-0 Gate Assessment:
- Clarity over cleverness: PASS. Keep the hybrid query flow explicit (classifier -> SQL/LLM tools -> verifier -> response composer) rather than implicit dynamic chains.
- Small, focused units: PASS. Separate ingestion, matching, query orchestration, shortlist, outreach, and interview generation into bounded services/tools.
- Behavior-first testing: PASS. Define failing-first tests for extraction normalization, query correctness counts, role-gated actions, and email approval guardrails.
- Minimal surface area: PASS with guardrail. Chosen dependencies are feature-essential; avoid adding vector stores or workflow engines unless measured need emerges.
- Refactor before extend: PASS. Start with deterministic SQL schema and explicit tool contracts before adding advanced ranking heuristics.

## Project Structure

### Documentation (this feature)

```text
specs/001-recruitment-ai-assistant/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
backend/
├── src/
│   ├── models/
│   ├── agents/
│   ├── orchestration/
│   ├── services/
│   ├── repositories/
│   └── api/
├── worker/
│   └── jobs/
├── migrations/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   ├── modules/
│   ├── services/
│   └── state/
└── tests/

contracts/
└── recruitment-api.yaml

scripts/
└── maintenance/
```

**Structure Decision**: Use a web application split with `backend/` and `frontend/` to keep runtime concerns isolated while preserving clear API contracts. Keep LangGraph orchestration inside backend bounded modules and process heavy ingestion/matching work via worker jobs.

**Deployment Default**: Docker Compose is the default for local and baseline production-like environments; non-Docker manual runtime is supported only as fallback for constrained hosts.

## Post-Design Constitution Check

- Clarity over cleverness: PASS. Chosen design uses explicit tool contracts and deterministic verification for count answers.
- Small, focused units: PASS. Data extraction, scoring, query routing, shortlist persistence, outreach, and interview generation have separate entities/services.
- Behavior-first testing: PASS. Contracts and quickstart define test-first checkpoints for core behaviors and failure paths.
- Minimal surface area: PASS. No non-essential infrastructure added; hybrid SQL+LLM avoids premature complexity.
- Refactor before extend: PASS. Plan keeps batch prompt scoring explicit and simple without introducing unnecessary abstraction layers.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|

No constitution violations requiring exception at planning stage.
