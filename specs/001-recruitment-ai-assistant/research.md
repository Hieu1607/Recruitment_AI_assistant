# Phase 0 Research: Recruitment AI Assistant Website

## Decision 1: Backend and frontend stack
- Decision: Use Python 3.11 backend (FastAPI + LangGraph + worker) and TypeScript frontend (React + Vite).
- Rationale: Python is the natural fit for PyMuPDF, LangGraph, and LLM tool orchestration; a typed frontend supports complex recruiter workflows and maintainable UI state.
- Alternatives considered: Monolithic server-rendered app (rejected due to lower UI interaction flexibility); all-in-one Python templates (rejected for rich dashboard needs).

## Decision 1a: LLM provider strategy (Groq default + Ollama optional)
- Decision: Support two LLM execution paths: Groq-hosted models and self-hosted Ollama models, with Groq as the default provider selected by environment variables.
- Rationale: Groq provides fast hosted inference as the default production path, while Ollama enables private/local model hosting when required by deployment policy.
- Alternatives considered: Groq-only (rejected because it removes self-host flexibility); Ollama-only (rejected because hosted default is preferred for quicker rollout and operational simplicity).

## Decision 2: Resume extraction strategy (PyMuPDF)
- Decision: Use PyMuPDF structured extraction (`rawdict`/block-level metadata) as primary path, with OCR fallback only for scanned/image PDFs.
- Rationale: Structured blocks provide source traceability and better section reconstruction for recruiter review/edit flows; selective OCR limits latency and cost.
- Alternatives considered: Plain text extraction only (rejected due to poor traceability and layout loss); OCR-first pipeline (rejected due to speed/cost and lower quality for text PDFs).

## Decision 3: Candidate profile normalization and validation
- Decision: Convert extraction output into strict schema-validated profile records before persistence, storing null/false defaults per spec rules and preserving source references.
- Rationale: Deterministic schema validation improves consistency for search and matching, and source links support auditing and correction.
- Alternatives considered: Free-form JSON storage (rejected due to weak queryability and unstable downstream scoring).

## Decision 4: Batch CV-JD scoring via shared LLM prompt
- Decision: Implement scoring by sending one batch payload containing one JD, many candidate CV profiles, and one shared scoring prompt template to LLM, then parse a returned list where each candidate includes weighted component scores (for example: skills, education, experience) and a total score normalized to 0-100.
- Rationale: A single batch prompt keeps evaluation context consistent across candidates in the same run, while component-level weighted scores improve transparency and recruiter control without losing a single ranking total.
- Alternatives considered: Per-candidate scoring calls (rejected due to inconsistent context and higher request overhead); total-score-only output (rejected due to low explainability for hiring decisions).

## Decision 5: Hybrid query architecture (SQL + LLM)
- Decision: Route simple deterministic predicates to SQL tools; route semantic/complex predicates to LLM tool over extracted resume sections; support hybrid sequence for combined queries.
- Rationale: SQL provides reliable counts and speed for structured filters; LLM handles nuanced reasoning across unstructured sections; hybrid approach balances precision and expressiveness.
- Alternatives considered: SQL-only (rejected because it cannot answer semantic questions well); LLM-only (rejected due to count reliability and auditability risks).

## Decision 6: LangGraph orchestration design
- Decision: Build explicit LangGraph nodes/tools: router/orchestrator, SQL search tool, LLM semantic search tool, shortlist write tool, email tool, verifier node, and response composer.
- Rationale: Explicit nodes align with constitution clarity and small-unit principles, simplify testability, and make fallback/retry behavior auditable.
- Alternatives considered: Single agent prompt with implicit tool calls (rejected due to lower control and harder debugging).

## Decision 6a: Environment-driven model selection
- Decision: Keep model names and provider choice in env files with keys `LLM_PROVIDER`, `GROQ_MODEL`, and `OLLAMA_MODEL`; default `LLM_PROVIDER=groq`.
- Rationale: Env-driven configuration avoids code changes during model switching and supports deployment-specific model governance.
- Alternatives considered: Hard-coded model names (rejected due to low flexibility); single shared model variable for all providers (rejected due to provider-specific configuration ambiguity).

## Decision 7: Memory and fallback policy
- Decision: Keep bounded conversation memory per recruiter session plus persisted query history metadata; add deterministic fallbacks (SQL-first for count conflicts, confidence-based escalation, and safe failure responses).
- Rationale: Memory improves multi-turn continuity while bounded scope controls drift; deterministic fallback reduces hallucination and protects decision quality.
- Alternatives considered: Unlimited memory accumulation (rejected due to noise/privacy risk); no memory (rejected due to poor follow-up query handling).

## Decision 8: Data storage and privacy controls
- Decision: Use PostgreSQL for core entities, score history, shortlist collections, and audit traces; store original resume files in MinIO object storage; enforce 12-month retention and masked-log policy.
- Rationale: Relational model suits filtering/reporting needs and role-based governance, while MinIO provides S3-compatible object storage suited for large PDF file lifecycle management.
- Alternatives considered: Generic local file storage (rejected for weaker portability and lifecycle controls); document-only database (rejected due to complex relational filtering and auditing requirements).

## Decision 9: API contract style
- Decision: Define RESTful API contracts for upload, extraction review/save, matching, conversational querying, shortlist persistence, outreach approval/send, and interview question generation.
- Rationale: Clear endpoint contracts keep frontend/backend integration predictable and easy to test.
- Alternatives considered: GraphQL-first (rejected to minimize surface area for initial release).

## Decision 10: Testing strategy
- Decision: Use behavior-first automated tests: unit tests for parsing/scoring/routing, integration tests for tool orchestration and DB persistence, and end-to-end tests for recruiter critical flows.
- Rationale: Matches constitution testing principle and protects high-risk workflows (count correctness, role restrictions, email approval gate).
- Alternatives considered: Integration-only testing (rejected because routing and scoring edge logic needs focused unit coverage).

## Decision 11: Docker-first deployment baseline
- Decision: Use Docker Compose as the default deployment and local development runtime, with services for API, worker, frontend, PostgreSQL, and MinIO.
- Rationale: Containerized runtime reduces environment drift, simplifies onboarding, and keeps production-like service wiring reproducible across developer machines.
- Alternatives considered: Manual host-level service startup as primary path (rejected due to setup inconsistency and higher onboarding friction).
