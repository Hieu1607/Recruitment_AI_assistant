# Scoring Evaluation Redesign Spec

## Goal

Redesign scoring so each candidate is evaluated by the LLM once per current job description version, stores raw per-criterion percentage scores, and lets users adjust job-level weights later without calling the LLM again.

## Current Context

The current scoring flow is centered on `score_candidates` in `backend/src/services/score_candidate.py`. It creates `MatchRun` and `MatchResult` rows, extracts a locked rubric from the JD, asks the LLM to score semantic criteria, applies rule-based scoring for measurable criteria, then persists weighted totals.

The current scoring UI in `frontend/src/routes/scoring/setup.tsx` is a three-step wizard. It lets users configure hidden information, select candidates, adjust weights, start scoring, and view results. This mixes three concerns that should be separated:

- JD content and recruiter-only hidden information.
- LLM candidate evaluation.
- User-specific or job-specific score weighting.

## Product Decisions

1. Hidden information belongs to each JD.
   The JD has two scoring inputs: public `jd_text` and recruiter-only `hidden_text`.

2. Changing `jd_text` or `hidden_text` does not automatically re-score candidates.
   Existing scoring results become `outdated`. The user can trigger re-scoring from the Scoring page or the bottom of the JD page.

3. UI weights are saved per job.
   Weight changes recalculate displayed scores from stored raw component scores and do not call the LLM.

4. Rule-based measurable criteria stay deterministic.
   Criteria such as `experience_years`, `graduation_status`, and `ever_studied_abroad` continue to be scored by backend rules, not by the LLM.

## Core Model

The system should distinguish raw evaluation from weighted display scoring.

Raw evaluation is the persisted answer to:

> Given this JD version and this candidate CV, how well does the candidate satisfy each rubric criterion?

Weighted display scoring is the computed answer to:

> Given stored raw criterion percentages and the current job-level weights, what total score should the UI show?

## Scoring Signature

Backend should compute a scoring signature for the active JD. The signature identifies the scoring input version and should include:

- `job_description_id`
- `jd_text`
- `hidden_text`
- rubric extraction prompt/version marker
- semantic scoring prompt/version marker
- supported measurable criteria version marker

If the current signature differs from the signature attached to a candidate evaluation, that evaluation is outdated.

## Persistence Design

Add a candidate evaluation snapshot concept keyed by JD and candidate.

Recommended table: `candidate_evaluations`

Fields:

- `id`
- `job_id`
- `job_description_id`
- `candidate_profile_id`
- `scoring_signature`
- `rubric_payload`
- `raw_component_scores`
- `rationale_summary`
- `status`: `pending`, `running`, `completed`, `failed`, `outdated`
- `error_message`
- `created_at`
- `updated_at`
- `scored_at`
- optional `source_match_run_id` for audit compatibility with existing `MatchRun`

`raw_component_scores` stores per-criterion results before user weighting:

```json
[
  {
    "criterionKey": "skills.python",
    "criterionType": "must_have",
    "section": "skills",
    "evaluationMode": "semantic",
    "requirementText": "Python experience",
    "scorePercent": 85,
    "evidenceSummary": "Candidate lists Python in multiple projects."
  }
]
```

For measurable criteria, `evaluationMode` is `measurable` and the score comes from backend rules.

## Job Weight Preferences

Add job-level scoring preferences.

Recommended table: `job_scoring_preferences`

Fields:

- `job_id`
- `section_weights` or `criterion_weights`
- `score_threshold`
- `updated_by_user_id`
- `updated_at`

Initial implementation can use section weights because the current UI already works with sections. Criterion-level weights can be added later if the product needs finer control.

Weight normalization:

- Ignore negative weights.
- Require total weight greater than zero.
- Normalize weights to sum to 1.0 for calculation.
- Criteria without an explicit section weight use weight 0 when the user provides explicit weights.
- If no job preference exists, use backend default section weights.

## Scoring Calculation

Add a pure calculation function that does not call the LLM:

```text
calculate_weighted_score(raw_component_scores, job_weights, score_threshold)
```

It returns:

- `componentScores` with `scorePercent`, `effectiveWeight`, and `weightedScore`
- `totalScore`
- `passedThreshold`

`totalScore` is computed from raw percentages and current weights. This means changing weights updates the displayed score without creating a new LLM evaluation.

## LLM Prompt Changes

Update locked rubric semantic scoring prompt so the LLM:

- Scores every listed semantic criterion.
- Returns score as a percentage from 0 to 100.
- Does not calculate `totalScore`.
- Does not use UI or section weights.
- Does not add, remove, or reinterpret criteria.
- Provides evidence for every criterion.

Rubric extraction still uses the full JD text and hidden information, but UI weights must not filter which criteria are extracted or scored.

## Backend Flow

### Candidate Parsed

When a resume is processed and a `CandidateProfile` is created:

1. Load the job's active JD.
2. Compute the current scoring signature.
3. Check whether a completed evaluation exists for candidate plus signature.
4. If missing, enqueue scoring for that candidate.
5. Store evaluation status as `pending` or `running`.
6. On completion, persist raw component scores and rationale.

This applies to both recruiter uploads and public job applications.

### JD Updated

When `jd_text` or `hidden_text` changes:

1. Compute the new scoring signature.
2. Mark existing completed evaluations for that JD/job as outdated if their signature differs.
3. Do not call the LLM automatically.
4. UI shows outdated state and a `Score again` action.

### Score Again

When the user clicks `Score again`:

1. Load current JD and hidden information.
2. Compute current scoring signature.
3. Find candidates for the job.
4. Enqueue evaluation only for candidates missing a completed evaluation for the current signature.
5. Preserve older snapshots for audit and comparison.
6. Refresh Scoring UI as evaluations complete.

## API Changes

Add or extend endpoints under jobs:

### Get Job Evaluations

`GET /jobs/{job_id}/evaluations`

Returns current evaluation rows for candidates in the job, using saved job weights to compute display scores.

Response includes:

- total candidates
- completed count
- pending/running count
- failed count
- outdated count
- average displayed score
- highest displayed score
- candidate evaluation rows

### Get Candidate Evaluation

`GET /jobs/{job_id}/candidates/{candidate_profile_id}/evaluation`

Returns the latest evaluation for a candidate, including raw component scores and current weighted display scores.

### Update Job Scoring Preferences

`PUT /jobs/{job_id}/scoring-preferences`

Stores weights and threshold for the job, then returns recalculated summary data.

### Trigger Re-score

`POST /jobs/{job_id}/evaluations/score-again`

Queues scoring for candidates that are missing current-signature evaluations.

## Scoring Page UI

Replace the three-step scoring wizard with a results-first screen.

Remove:

- Stepper UI.
- Candidate selection for manual scoring.
- Hidden information editor from Scoring page.
- `Start scoring` as the primary workflow.

Keep or redesign:

- Summary stats.
- Candidate results table.
- Component breakdown.
- Rationale and evidence.
- Radar/chart if it remains readable.
- Job-level weight controls.
- `Score again` action when results are outdated.

Candidate rows should show:

- completed score
- pending/running state
- failed state with retry option
- outdated badge when JD signature changed
- detailed criterion evidence

## JD Page UI

JD page should own both public JD text and hidden information.

At the bottom of the JD page, show scoring status:

- `Current` when evaluations match current signature.
- `Outdated` when JD or hidden information changed after scoring.
- `Not scored` when no evaluation exists.
- `Scoring` when a scoring task is running.

Show `Score again` when status is `Outdated` or `Not scored`.

## Candidate Detail UI

Add an evaluation section or tab to `frontend/src/routes/candidates/detail.tsx`.

It should show:

- Overall displayed match score.
- Current scoring status.
- Per-criterion score percentages.
- Evidence and rationale for each criterion.
- Rule-based vs semantic badge.
- Outdated warning if the candidate evaluation is not based on current JD signature.
- Retry action for failed evaluation.

If the candidate has been parsed but evaluation is still queued/running, show a loading state instead of an empty result.

## Compatibility With Existing Match Runs

Existing `MatchRun` and `MatchResult` can remain during transition.

Recommended migration path:

1. Introduce candidate evaluations and job scoring preferences.
2. Keep existing score endpoint working while new UI migrates.
3. Backfill candidate evaluations from latest completed match results where possible.
4. Move Scoring page to evaluation endpoints.
5. Deprecate manual score run workflow once the new flow is stable.

## Error Handling

Provider quota or rate limit:

- Store evaluation status as `failed`.
- Preserve error message for logs/admin debugging.
- Show retry action in UI.

Parse failure from LLM:

- Retry with strict JSON suffix as current scoring flow does.
- If still failing, mark evaluation failed.
- Do not silently store zero scores for semantic criteria.

Partial batch failure:

- Completed candidate evaluations remain completed.
- Failed candidate evaluations can be retried independently.

## Testing Plan

Backend tests:

- Scoring signature changes when `jd_text` changes.
- Scoring signature changes when `hidden_text` changes.
- Existing evaluations become outdated after JD scoring input changes.
- Weight updates recalculate totals without calling LLM.
- Measurable criteria are scored by backend rules.
- Semantic prompt requests every listed criterion and no total score.
- Candidate parse completion enqueues candidate evaluation.
- `Score again` enqueues only candidates missing current-signature evaluations.

Frontend tests:

- Scoring page no longer shows the three-step wizard.
- Weight changes update displayed totals.
- Outdated evaluations show a badge and `Score again`.
- Candidate detail page shows criterion-level evaluation.
- Pending/running/failed evaluation states render correctly.

## Non-Goals

- No automatic full re-score immediately when JD or hidden information changes.
- No user-editable rubric in this phase.
- No LLM call on weight slider changes.
- No removal of rule-based measurable scoring.
- No full replacement of existing match-run audit data in the first migration step.

## Open Implementation Choice

The implementation can initially compute weighted display scores in the backend response so all clients stay consistent. Frontend may still optimistically recalculate while the user drags sliders, but persisted preferences and canonical totals should come from backend calculation.
