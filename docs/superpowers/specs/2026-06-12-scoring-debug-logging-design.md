# Scoring Debug Logging Design

## Goal

Add detailed, file-based debug logging for the candidate scoring pipeline so engineers can reconstruct a scoring run end-to-end without relying on console logs alone.

## Scope

This design applies only to the scoring flow implemented in `backend/src/services/score_candidate.py`, especially the `score_candidates(...)` entry point and its rubric extraction, semantic scoring, fallback scoring, candidate breakdown, persistence, and failure handling stages.

It does not change global logging configuration for the backend application.

## Requirements

### Output location

- Write scoring debug artifacts under `backend/logs/scoring/`.
- Each `score_candidates(...)` run must create its own file keyed by `match_run_id`.
- The logging output must be created incrementally while the run is in progress so partial traces survive crashes.

### Output shape

- The debug output should be structured JSON, not plain text paragraphs.
- The file should be append-friendly and easy to inspect manually.
- Each event should include a timestamp, event name, and payload.

### Debug coverage

The trace must cover:

- run context: `match_run_id`, `job_description_id`, `initiated_by_user_id`, `score_threshold`, `batch_size`, requested candidate ids, candidate count
- job description preparation: public JD presence, hidden text presence, combined scoring text length
- rubric extraction: prompt preview, response preview, parsed rubric payload, normalized rubric, filtered criteria and reasons
- section weights normalization
- semantic scoring: prompt preview, response preview, retries, provider/model used, parse failures, fallback behavior
- fallback batch scoring when rubric locking is unavailable
- per-candidate score breakdown including criterion weights, evaluation mode, score, weighted score, and evidence
- measurable criterion evaluation details including actual value, expected value, operator, and match result
- persistence milestones when batch results are saved
- terminal status: completed or failed
- error payloads with enough context to locate the failing stage

### Prompt and response logging

- Prompt and response content should be logged in truncated form, not full raw text.
- Each preview must include the original text length and a shortened text preview.
- The truncation must be deterministic so repeated runs are comparable.

### Compatibility

- Existing `logger.warning(...)`, `logger.error(...)`, and `logger.exception(...)` behavior in scoring should remain in place.
- Existing API responses and database writes must remain unchanged.

## Proposed design

### New helper

Create a focused helper module for scoring debug logging, separate from the standard app logger. The helper should:

- create the target directory if missing
- open a per-run file lazily
- append one JSON object per event
- serialize datetimes, UUIDs, Decimals, and exceptions safely
- expose small methods such as `record_event(...)`, `record_llm_attempt(...)`, and `finalize(...)`

### Event model

Each line should contain:

- `timestamp`
- `match_run_id`
- `event`
- `payload`

Recommended event names:

- `run_started`
- `job_description_prepared`
- `rubric_extraction_started`
- `rubric_extraction_completed`
- `rubric_extraction_failed`
- `semantic_scoring_started`
- `semantic_scoring_attempt`
- `semantic_scoring_completed`
- `semantic_scoring_parse_failed`
- `fallback_batch_scoring_started`
- `fallback_batch_scoring_completed`
- `candidate_scored`
- `batch_persist_started`
- `batch_persist_completed`
- `run_completed`
- `run_failed`

### Data minimization

The trace is for debugging logic, not archiving full source content. Therefore:

- log preview text for prompts and responses
- log lengths for full JD, hidden text, and prompts
- avoid dumping the entire resume text unless a value is already surfaced in the scoring payload
- log candidate identifiers and display names because those are necessary for correlating a score

## Testing strategy

Use TDD with focused tests around:

- successful run creates a scoring log file with expected event sequence
- measurable and semantic scoring emit candidate breakdown events
- semantic retry and fallback paths emit retry/failure events
- quota/rate-limit failure emits terminal failure event without removing existing operational logs

## Risks

- Overly large debug files if previews are not truncated aggressively enough
- Tests that couple too tightly to full payload shapes instead of stable event essentials
- Hidden prompt/response content becoming too verbose if truncation is inconsistent

## Decision

Implement a scoring-specific structured debug logger under `backend/logs/scoring/`, keep standard logger calls unchanged, and add event coverage at each major scoring stage with truncated prompt and response previews.
