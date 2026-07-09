# Resume Processing Batch Performance Design

## Goal

Optimize the common demo workload of 5-10 resumes uploaded in one request:

- Parse resumes concurrently without flooding the LLM provider.
- Extract the active JD rubric once per scoring signature.
- Evaluate parsed candidates in token-safe batches.
- Preserve per-resume and per-candidate status, retry, and failure isolation.

The target is minimum end-to-end batch completion time, not merely a fast HTTP
upload response. Upload remains asynchronous and returns `202`.

## Current Problems

The upload endpoint already accepts multiple files, but it creates one independent
`process_resume` task per file. Each successful parse immediately creates one
independent `evaluate_candidate` task.

That causes two structural costs:

1. Parsing concurrency is controlled only by the shared Celery worker. Resume
   parsing competes with evaluation, email, and interview tasks, and provider
   concurrency is not explicitly bounded for the demo workload.
2. `evaluate_candidate_for_current_jd` calls
   `evaluate_candidate_profile_raw` for every candidate. That path extracts the
   same JD rubric again before scoring each candidate.

## Selected Architecture

Use a durable database-backed processing batch and dedicated Celery queues.

```text
POST 5-10 resumes
        |
        v
Create ResumeProcessingBatch and ResumeDocument rows
        |
        v
Queue one parse task per resume on resume_parse
        |
        v
Parse concurrently, default provider concurrency = 5
        |
        v
Each task records its terminal result in the batch
        |
        v
The task that closes the batch queues one evaluate_resume_batch task
        |
        v
Load active JD and scoring signature
        |
        v
Extract rubric once, score semantic criteria in adaptive candidate batches
        |
        v
Persist one CandidateEvaluation per successfully parsed candidate
```

This design does not rely on a Celery chord as the source of truth. Celery
retries and worker restarts must not lose batch completion state.

## Persistence Model

Add `ResumeProcessingBatch` with:

- `id`: UUID primary key.
- `job_id`: owning job.
- `total_count`: number of submitted resumes.
- `terminal_count`: number of resumes that reached `processed` or `failed`.
- `processed_count`: successfully parsed resumes.
- `failed_count`: failed resumes.
- `status`: `parsing`, `evaluation_pending`, `evaluating`, `completed`,
  `completed_with_errors`, or `failed`.
- `evaluation_task_id`: nullable Celery task identifier for diagnostics.
- `evaluation_dispatch_attempted_at`: nullable dispatch timestamp.
- `created_at`, `updated_at`, `completed_at`.

Add nullable `processing_batch_id` to `ResumeDocument`. Existing rows and legacy
single-resume callers remain valid. New single uploads also create a batch of one
so both upload paths use the same lifecycle.

`CandidateEvaluation` remains the per-candidate result store. Its existing unique
constraint on JD, candidate, and scoring signature remains the idempotency
boundary for evaluation snapshots.

## Parse Scheduling

Route `process_resume` to `resume_parse`.

Run a dedicated parse worker with concurrency `5` by default. Make the value
configurable through `RESUME_PARSE_WORKER_CONCURRENCY`; deployments with stricter
provider limits can lower it without code changes.

The API creates and commits the batch plus all `ResumeDocument` rows before
publishing tasks. If publishing one task fails, mark that resume failed and run
the same batch completion transition rather than leaving the batch permanently
open.

`process_resume` keeps its existing retry policy. After success or terminal
failure, it calls a batch transition service. That service:

1. Locks the `ResumeProcessingBatch` row.
2. Recomputes terminal, processed, and failed counts from `ResumeDocument`
   statuses instead of blindly incrementing counters.
3. Changes `parsing` to `evaluation_pending` exactly once when every resume is
   terminal.
4. Commits the transition before publishing `evaluate_resume_batch`.
5. Records the dispatched task identifier in a second short transaction.

Publishing is at-least-once rather than exactly-once because the database and
Redis do not share a transaction. A lightweight recovery task scans batches
left in `evaluation_pending` and republishes them after a short grace period.
Recomputation and evaluation idempotency make duplicate delivery safe.

## Batch Evaluation

Add `evaluate_resume_batch(processing_batch_id)`, routed to
`candidate_evaluation`.

The task loads:

- Successfully parsed candidate profiles in the batch.
- The current active JD.
- The current scoring signature.
- Existing candidate evaluations for that signature.

Completed evaluations are skipped. Missing or failed evaluations are set to
`pending`, then the batch status changes atomically from `evaluation_pending` to
`evaluating`.

Refactor the scoring service to expose a side-effect-free batch engine:

```python
evaluate_candidate_profiles_raw(
    candidates,
    job_description_text,
    section_weights=None,
) -> dict[candidate_id, RawEvaluation]
```

The engine:

1. Creates one scoring LLM provider.
2. Extracts and normalizes the locked rubric once.
3. Scores measurable criteria locally for every candidate.
4. Uses the existing token-budget batch planner for semantic criteria.
5. Returns independent raw results keyed by candidate ID.

Do not call `score_candidates` directly because that function owns legacy
`MatchRun` and `MatchResult` side effects. Extract and reuse its pure rubric and
adaptive semantic-scoring logic instead.

For 10 candidates, the expected request shape becomes approximately:

- 10 resume parse requests, executed in two waves at concurrency 5.
- 1 rubric extraction request.
- 1-3 semantic scoring requests, depending on token budgets.

## Queue Layout

- `resume_parse`: resume extraction and parsing, worker concurrency 5.
- `candidate_evaluation`: batch evaluation, worker concurrency 1 initially.
- `default`: email, outreach, reports, and unrelated tasks.

Evaluation worker concurrency 1 is sufficient for the demo path because each
task already batches 5-10 candidates. It also prevents two uploaded jobs from
unexpectedly multiplying large semantic requests. This value remains
configurable.

Development and production Compose configurations must define the same routes
and worker topology. Celery beat runs the pending-batch recovery task every 15
seconds; it is a failure-recovery path and not part of normal scheduling.

## Error Handling

### Partial parse failure

If 8 of 10 resumes parse successfully, evaluate those 8. Finish the batch as
`completed_with_errors`; retain the two failed resume statuses and messages.

### Rubric extraction or provider failure

Let the Celery evaluation task retry according to provider-aware retry policy.
Keep the batch `evaluating` during retry. On terminal failure, mark unfinished
candidate evaluations `failed` and the batch `failed`.

### Individual candidate failure

Persist successful candidate results independently. Mark only the invalid
candidate evaluation failed. Finish the batch as `completed_with_errors`.

### Duplicate delivery

Parse status recomputation, guarded batch state transitions, and the existing
candidate-evaluation uniqueness constraint make duplicate tasks no-ops.
The evaluation task claims a batch by atomically changing
`evaluation_pending` to `evaluating`; a duplicate task that loses this claim
exits successfully.

### JD changes during processing

Resolve the active JD and scoring signature when evaluation begins. Every
result stores that signature. Existing outdated-scoring behavior remains
responsible for marking results outdated if the JD changes afterward.

## API Compatibility

Keep the existing upload endpoint and response fields:

- `total_files`
- `queued_files`
- `items`

Add `processing_batch_id` as an optional additive field. Existing frontend
behavior remains valid. A later UI enhancement may poll a batch-status endpoint,
but that endpoint is not required for the performance change.

Public single-resume application follows the same batch-of-one path without
changing its public response contract.

## Observability

Add structured log fields and trace events:

- `processing_batch_id`
- `resume_document_id`
- `candidate_profile_id`
- queue wait time
- parse duration
- batch parse completion duration
- rubric extraction duration
- semantic batch count and duration
- total evaluation duration
- total end-to-end batch duration

Never log raw CV text, JD hidden text, or provider credentials.

## Tests

### Unit tests

- Duplicate parse completion creates one evaluation claim even if Redis receives
  more than one task message.
- Recovery republishes an `evaluation_pending` batch whose first dispatch was
  interrupted.
- A batch with partial parse failures evaluates successful profiles.
- Batch evaluation extracts the rubric once for 5-10 candidates.
- Completed candidate evaluations are skipped on retry.
- Semantic batching respects token limits.
- One candidate failure does not discard successful evaluations.

### Integration tests

- Uploading 10 files creates one processing batch and 10 parse tasks.
- All terminal parse states trigger one evaluation task.
- Worker retry and duplicate task delivery do not create duplicate evaluations.
- A JD change after evaluation marks prior snapshots outdated through the
  existing signature mechanism.

### Performance verification

Use fixed local fixtures representing 5 and 10 CVs. Record:

- Time until first parsed candidate.
- Time until all parsing is terminal.
- Time until all available evaluations are terminal.
- Number of LLM calls by operation.

Acceptance criteria for a 10-CV run:

- No more than 5 simultaneous resume parse calls by default.
- Exactly one rubric extraction for one JD/signature.
- Semantic requests follow the adaptive token plan rather than one call per CV.
- No duplicate candidate evaluation rows.
- Partial failures do not block successful candidates.

Absolute latency is reported rather than hard-coded because provider response
time and quota vary by environment.

## Rollout

1. Add the batch model and migration without changing task routing.
2. Add batch transition and batch evaluation services behind
   `BATCH_RESUME_PIPELINE_ENABLED`.
3. Route new uploads through the batch pipeline in development.
4. Verify 5- and 10-CV traces and provider error rates.
5. Enable the same pipeline in production.
6. Remove the legacy per-candidate auto-evaluation branch after one stable
   release.

The feature flag provides rollback to the existing per-resume behavior without
reverting schema changes.

## Non-Goals

- Combining multiple resumes into one resume-parsing prompt.
- Changing the resume extraction schema.
- Replacing Redis or Celery.
- Replacing the existing scoring-signature or outdated-result model.
- Changing user-configurable scoring weights.
- Adding a new frontend progress screen in this phase.
