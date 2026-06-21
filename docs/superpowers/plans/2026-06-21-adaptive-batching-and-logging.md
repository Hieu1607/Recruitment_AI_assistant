# Adaptive Batching And Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fixed/count-only LLM batching in scoring and chat with token-budget-aware adaptive batching, and add structured logs that make context-window risk, output-token risk, retries, and latency easy to debug.

**Architecture:** Add shared token-budget estimation utilities, then wire them into scoring and chat independently. Scoring keeps the locked-rubric pipeline but batches candidates and semantic criteria by budget. Chat becomes a token-aware map-reduce pipeline where semantic map batches return compact candidate summaries and the final answer switches to id/name-only mode when the result set is too large.

**Tech Stack:** Python, FastAPI, SQLAlchemy, LangGraph, pytest, existing `ScoringDebugLogger`, existing `LangGraphTraceLogger`

---

## File Structure

- Create `backend/src/services/token_budget.py`: shared token estimation, JSON payload sizing, and budget structs.
- Create `backend/src/services/scoring_batching.py`: scoring-specific candidate and criterion batch planner.
- Create `backend/src/services/ai_agent/chat_batching.py`: chat-specific map/reduce planner and compact payload helpers.
- Modify `backend/src/core/config.py`: add conservative token budget settings.
- Modify `backend/src/services/score_candidate.py`: replace fixed candidate loop with adaptive batches, add split retry hooks, add structured batch logs.
- Modify `backend/src/services/ai_agent/nodes.py`: replace single semantic prompt with map-reduce calls, add compact final answer mode, add structured trace events.
- Modify `backend/src/prompts/build_prompts.py`: add compact chat map/reduce prompts and compact final-answer prompt.
- Modify `backend/src/services/scoring_debug.py`: add optional helpers for token budget payloads if needed.
- Modify `backend/src/services/ai_agent/langgraph_trace.py`: keep existing trace format, but ensure new event payloads serialize cleanly.
- Test `backend/tests/test_token_budget.py`.
- Test `backend/tests/test_scoring_batching.py`.
- Test `backend/tests/test_chat_batching.py`.
- Extend `backend/tests/test_score_candidate_service.py`.
- Extend `backend/tests/test_ai_agent_nodes.py`.
- Extend `backend/tests/test_job_chat_trace_logging.py`.

---

## Token Budget Defaults

Add conservative settings:

```python
SCORING_CONTEXT_WINDOW_TOKENS: int = int(os.getenv("SCORING_CONTEXT_WINDOW_TOKENS", "8192"))
SCORING_OUTPUT_TOKEN_BUDGET: int = int(os.getenv("SCORING_OUTPUT_TOKEN_BUDGET", "4096"))
SCORING_CONTEXT_RESERVE_TOKENS: int = int(os.getenv("SCORING_CONTEXT_RESERVE_TOKENS", "768"))
SCORING_MAX_CANDIDATES_PER_BATCH: int = int(os.getenv("SCORING_MAX_CANDIDATES_PER_BATCH", "8"))
SCORING_MAX_SEMANTIC_CRITERIA_PER_CALL: int = int(os.getenv("SCORING_MAX_SEMANTIC_CRITERIA_PER_CALL", "12"))

CHAT_CONTEXT_WINDOW_TOKENS: int = int(os.getenv("CHAT_CONTEXT_WINDOW_TOKENS", "8192"))
CHAT_OUTPUT_TOKEN_BUDGET: int = int(os.getenv("CHAT_OUTPUT_TOKEN_BUDGET", "4096"))
CHAT_CONTEXT_RESERVE_TOKENS: int = int(os.getenv("CHAT_CONTEXT_RESERVE_TOKENS", "768"))
CHAT_MAX_CANDIDATES_PER_MAP_BATCH: int = int(os.getenv("CHAT_MAX_CANDIDATES_PER_MAP_BATCH", "40"))
CHAT_MAX_DETAILED_FINAL_CANDIDATES: int = int(os.getenv("CHAT_MAX_DETAILED_FINAL_CANDIDATES", "10"))
CHAT_MAX_COMPACT_FINAL_CANDIDATES: int = int(os.getenv("CHAT_MAX_COMPACT_FINAL_CANDIDATES", "50"))
```

Use conservative defaults because current default model is `llama-3.1-8b-instant` and the app currently sets high output budgets without explicit context-window awareness.

---

### Task 1: Shared Token Budget Utilities

**Files:**
- Create `backend/src/services/token_budget.py`
- Create `backend/tests/test_token_budget.py`

- [ ] **Step 1: Write failing tests**

Add tests:

```python
from src.services.token_budget import (
    BudgetWindow,
    estimate_tokens,
    estimate_json_tokens,
    fits_budget,
)


def test_estimate_tokens_uses_conservative_char_ratio():
    assert estimate_tokens("abcd") == 2
    assert estimate_tokens("") == 0


def test_estimate_json_tokens_counts_serialized_payload():
    payload = {"candidate": {"id": "1", "skills_text": "Python FastAPI"}}
    assert estimate_json_tokens(payload) >= estimate_tokens('"skills_text"')


def test_budget_window_computes_input_budget_after_output_and_reserve():
    window = BudgetWindow(context_window=8192, output_budget=2048, reserve=512)
    assert window.input_budget == 5632


def test_fits_budget_rejects_payload_over_input_budget():
    window = BudgetWindow(context_window=100, output_budget=30, reserve=10)
    assert fits_budget(estimate_tokens("x" * 500), window) is False
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest backend/tests/test_token_budget.py -v
```

Expected: import failure because `token_budget.py` does not exist.

- [ ] **Step 3: Implement utility module**

Implement:

```python
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any


_CHARS_PER_TOKEN = 3.2


@dataclass(frozen=True)
class BudgetWindow:
    context_window: int
    output_budget: int
    reserve: int

    @property
    def input_budget(self) -> int:
        return max(0, int(self.context_window) - int(self.output_budget) - int(self.reserve))


def estimate_tokens(text: str | None) -> int:
    normalized = str(text or "")
    if not normalized:
        return 0
    return max(1, math.ceil(len(normalized) / _CHARS_PER_TOKEN))


def estimate_json_tokens(payload: Any) -> int:
    return estimate_tokens(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def fits_budget(estimated_input_tokens: int, window: BudgetWindow) -> bool:
    return estimated_input_tokens <= window.input_budget
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest backend/tests/test_token_budget.py -v
```

Expected: all tests pass.

---

### Task 2: Scoring Adaptive Batch Planner

**Files:**
- Create `backend/src/services/scoring_batching.py`
- Create `backend/tests/test_scoring_batching.py`

- [ ] **Step 1: Write failing tests**

Add tests:

```python
from src.services.scoring_batching import (
    ScoringBatchPlan,
    build_scoring_batch_plan,
)
from src.services.token_budget import BudgetWindow


def _candidate(candidate_id: str, skills: str = "Python") -> dict:
    return {"id": candidate_id, "full_name": f"Candidate {candidate_id}", "skills_text": skills}


def test_scoring_plan_keeps_short_candidates_together():
    plan = build_scoring_batch_plan(
        candidates=[_candidate("1"), _candidate("2"), _candidate("3")],
        semantic_criteria=[{"key": "skills.python"}],
        static_prompt_tokens=50,
        window=BudgetWindow(context_window=1000, output_budget=200, reserve=100),
        max_candidates_per_batch=8,
        max_criteria_per_call=12,
    )

    assert [len(batch.candidates) for batch in plan.candidate_batches] == [3]
    assert plan.criterion_batches == [[{"key": "skills.python"}]]


def test_scoring_plan_splits_long_candidates_by_input_budget():
    long_text = "Python " * 500
    plan = build_scoring_batch_plan(
        candidates=[_candidate("1", long_text), _candidate("2", long_text)],
        semantic_criteria=[{"key": "skills.python"}],
        static_prompt_tokens=50,
        window=BudgetWindow(context_window=900, output_budget=200, reserve=100),
        max_candidates_per_batch=8,
        max_criteria_per_call=12,
    )

    assert [len(batch.candidates) for batch in plan.candidate_batches] == [1, 1]


def test_scoring_plan_splits_many_criteria_by_output_risk():
    criteria = [{"key": f"skills.{idx}"} for idx in range(25)]
    plan = build_scoring_batch_plan(
        candidates=[_candidate("1")],
        semantic_criteria=criteria,
        static_prompt_tokens=50,
        window=BudgetWindow(context_window=2000, output_budget=300, reserve=100),
        max_candidates_per_batch=8,
        max_criteria_per_call=10,
    )

    assert [len(batch) for batch in plan.criterion_batches] == [10, 10, 5]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest backend/tests/test_scoring_batching.py -v
```

Expected: import failure because planner does not exist.

- [ ] **Step 3: Implement planner**

Create dataclasses:

```python
@dataclass(frozen=True)
class CandidateBatch:
    candidates: list[dict[str, Any]]
    estimated_input_tokens: int
    estimated_output_tokens: int


@dataclass(frozen=True)
class ScoringBatchPlan:
    candidate_batches: list[CandidateBatch]
    criterion_batches: list[list[dict[str, Any]]]
    total_candidates: int
    total_criteria: int
    planner_settings: dict[str, Any]
```

Implement:

```python
def build_scoring_batch_plan(
    *,
    candidates: list[dict[str, Any]],
    semantic_criteria: list[dict[str, Any]],
    static_prompt_tokens: int,
    window: BudgetWindow,
    max_candidates_per_batch: int,
    max_criteria_per_call: int,
) -> ScoringBatchPlan:
    ...
```

Packing rules:
- Keep `id` and display fields even when candidate payload is long.
- Greedy-pack candidates until `static_prompt_tokens + candidate_tokens` exceeds `window.input_budget`.
- Also cap by `max_candidates_per_batch`.
- Estimate output as `candidate_count * max(1, criterion_count) * 80 + candidate_count * 120`.
- If output estimate exceeds `window.output_budget`, close the current batch earlier.
- Split criteria into chunks of `max_criteria_per_call`.
- Never produce an empty batch.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest backend/tests/test_scoring_batching.py -v
```

Expected: all tests pass.

---

### Task 3: Wire Adaptive Batching Into Scoring

**Files:**
- Modify `backend/src/core/config.py`
- Modify `backend/src/services/score_candidate.py`
- Extend `backend/tests/test_score_candidate_service.py`
- Extend `backend/tests/test_score_candidate_error_handling.py`

- [ ] **Step 1: Add failing tests for scoring batch planning**

Add tests that monkeypatch:
- `build_scoring_batch_plan`
- `_generate_semantic_scores_with_retries`
- `build_prompts.build_locked_rubric_semantic_scoring_prompt`

Required assertions:
- `score_candidates()` uses planner batches instead of raw `batch_size`.
- Semantic criteria are sent in criterion chunks.
- Final scores merge semantic results from multiple criterion chunks.
- Debug log records `adaptive_batch_plan_created`, `semantic_batch_started`, and `semantic_batch_completed`.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest backend/tests/test_score_candidate_service.py backend/tests/test_score_candidate_error_handling.py -v
```

Expected: new tests fail because scoring still loops by fixed `batch_size`.

- [ ] **Step 3: Add config values**

Add settings listed in "Token Budget Defaults" to `backend/src/core/config.py`.

- [ ] **Step 4: Replace fixed loop with planner**

Current code:

```python
for i in range(0, len(candidate_dicts), batch_size):
    batch = candidate_dicts[i : i + batch_size]
```

Replace with:

```python
semantic_criteria = [criterion for criterion in rubric["criteria"] if criterion.get("measurable") is None]
static_prompt_tokens = estimate_tokens(scoring_jd_text) + estimate_json_tokens({"rubric": rubric})
plan = build_scoring_batch_plan(...)

for candidate_batch in plan.candidate_batches:
    batch = candidate_batch.candidates
    ...
    for criterion_batch in plan.criterion_batches:
        semantic_by_candidate = merge_semantic_scores(
            semantic_by_candidate,
            _generate_semantic_scores_with_retries(... criterion_batch ...),
        )
```

- [ ] **Step 5: Add split retry hook**

When semantic scoring parse returns empty after retries or provider returns length finish reason in raw response, split the candidate batch if possible:

```python
if semantic_result_empty_and_batch_has_multiple_candidates:
    retry left half
    retry right half
```

For this iteration, keep provider-limit errors fatal exactly as current behavior does.

- [ ] **Step 6: Run scoring tests**

Run:

```bash
pytest backend/tests/test_score_candidate_service.py backend/tests/test_score_candidate_error_handling.py backend/tests/test_jobs_score_endpoint.py -v
```

Expected: pass.

---

### Task 4: Chat Adaptive Map-Reduce Planner

**Files:**
- Create `backend/src/services/ai_agent/chat_batching.py`
- Create `backend/tests/test_chat_batching.py`

- [ ] **Step 1: Write failing tests**

Add tests:

```python
from src.services.ai_agent.chat_batching import (
    AnswerMode,
    build_chat_map_batches,
    choose_answer_mode,
    compact_candidate_identity,
)
from src.services.token_budget import BudgetWindow


def test_chat_map_batches_group_short_candidates():
    candidates = [{"id": str(i), "full_name": f"C{i}", "skills_text": "Python"} for i in range(5)]
    batches = build_chat_map_batches(
        question="Who knows Python?",
        candidates=candidates,
        job_context={},
        static_prompt_tokens=100,
        window=BudgetWindow(context_window=2000, output_budget=500, reserve=200),
        max_candidates_per_batch=40,
    )

    assert [len(batch.candidates) for batch in batches] == [5]


def test_chat_map_batches_split_long_candidates():
    candidates = [
        {"id": "1", "full_name": "One", "skills_text": "Python " * 500},
        {"id": "2", "full_name": "Two", "skills_text": "Python " * 500},
    ]
    batches = build_chat_map_batches(
        question="Who knows Python?",
        candidates=candidates,
        job_context={},
        static_prompt_tokens=100,
        window=BudgetWindow(context_window=1000, output_budget=300, reserve=100),
        max_candidates_per_batch=40,
    )

    assert [len(batch.candidates) for batch in batches] == [1, 1]


def test_choose_answer_mode_uses_compact_mode_for_large_result_sets():
    candidates = [{"id": str(i), "full_name": f"C{i}", "skills_text": "Python"} for i in range(100)]
    mode = choose_answer_mode(
        candidates=candidates,
        detailed_threshold=10,
        compact_threshold=50,
        estimated_full_tokens=9000,
        final_input_budget=3000,
    )

    assert mode == AnswerMode.COMPACT_ID_NAME


def test_compact_candidate_identity_drops_profile_fields():
    compact = compact_candidate_identity({"id": "1", "full_name": "A", "skills_text": "Python"})

    assert compact == {"id": "1", "full_name": "A"}
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest backend/tests/test_chat_batching.py -v
```

Expected: import failure because module does not exist.

- [ ] **Step 3: Implement chat batching module**

Define:

```python
class AnswerMode(str, Enum):
    DETAILED = "detailed"
    COMPACT_ID_NAME = "compact_id_name"
    NO_MATCH = "no_match"
```

Implement:
- `build_chat_map_batches(...)`
- `compact_candidate_identity(candidate)`
- `compact_map_result(map_result)`
- `choose_answer_mode(...)`
- `limit_compact_candidates(candidates, max_count)`

Batch rules:
- Greedy-pack by estimated JSON token cost.
- Always keep `id` and `full_name`.
- If one candidate is too large, compress candidate to relevant fields plus `summary_text`.
- Cap by `CHAT_MAX_CANDIDATES_PER_MAP_BATCH`.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest backend/tests/test_chat_batching.py -v
```

Expected: pass.

---

### Task 5: Chat Map And Reduce Prompts

**Files:**
- Modify `backend/src/prompts/build_prompts.py`
- Extend `backend/tests/test_build_prompts.py`

- [ ] **Step 1: Write failing prompt tests**

Add tests for:
- `build_chat_semantic_map_prompt`
- `build_chat_reduce_prompt`
- `build_compact_answer_prompt`

Assertions:
- map prompt asks for JSON only.
- map output schema includes `id`, `name`, `score`, `reason`.
- reduce prompt input is map summaries only, not full profile fields.
- compact answer prompt includes only `id` and `full_name`.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest backend/tests/test_build_prompts.py -v
```

Expected: new tests fail because prompt builders do not exist.

- [ ] **Step 3: Add prompt builders**

Add methods:

```python
def build_chat_semantic_map_prompt(self, question, candidates, job_context=None) -> str:
    ...

def build_chat_reduce_prompt(self, question, map_results, job_context=None) -> str:
    ...

def build_compact_answer_prompt(self, question, compact_candidates, total_count, omitted_count, job_context=None) -> str:
    ...
```

Map response schema:

```json
{
  "qualifiedCandidates": [
    {"id": "uuid", "name": "string", "score": 0.0, "reason": "short string"}
  ],
  "batchQualifiedCount": 0
}
```

Reduce response schema:

```json
{
  "totalQualified": 0,
  "rankedCandidates": [
    {"id": "uuid", "name": "string", "score": 0.0, "reason": "short string"}
  ]
}
```

- [ ] **Step 4: Run prompt tests**

Run:

```bash
pytest backend/tests/test_build_prompts.py -v
```

Expected: pass.

---

### Task 6: Wire Chat Map-Reduce Into `llm_node` And `answer_node`

**Files:**
- Modify `backend/src/services/ai_agent/nodes.py`
- Extend `backend/tests/test_ai_agent_nodes.py`
- Extend `backend/tests/test_job_chat_endpoint.py`

- [ ] **Step 1: Write failing tests for `llm_node`**

Add tests:
- `test_llm_node_batches_large_candidate_pool_by_token_budget`
- `test_llm_node_reduces_map_results_without_full_candidate_payload`
- `test_llm_node_splits_failed_map_batch_once`

Use fake LLM responses:

```json
{"qualifiedCandidates":[{"id":"cand-1","name":"A","score":0.9,"reason":"Python"}],"batchQualifiedCount":1}
```

Expected `llm_node` result:

```python
{
    "llm_result": {
        "total_qualified_candidates": 1,
        "qualified_candidates": {"cand-1": "Python"},
        "ranked_candidates": [{"id": "cand-1", "name": "A", "score": 0.9, "reason": "Python"}],
        "answer_mode": "detailed" or "compact_id_name",
    }
}
```

- [ ] **Step 2: Write failing tests for `answer_node` compact mode**

Add tests:
- `test_answer_node_uses_compact_id_name_prompt_when_llm_result_requests_compact_mode`
- `test_answer_node_keeps_matched_candidate_ids_outside_prompt_for_large_result_set`

Assert that `build_compact_answer_prompt` receives only:

```python
[{"id": "cand-1", "full_name": "A"}]
```

and not `skills_text`, `experience_text`, or raw profile fields.

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
pytest backend/tests/test_ai_agent_nodes.py backend/tests/test_job_chat_endpoint.py -v
```

Expected: new tests fail because map-reduce and compact answer mode are not wired.

- [ ] **Step 4: Implement `llm_node` map-reduce**

Flow:

```python
candidates = _resolve_candidates(...)
batches = build_chat_map_batches(...)
map_results = []
for batch in batches:
    prompt = build_prompts.build_chat_semantic_map_prompt(...)
    response = _get_llm().generate(prompt)
    map_results.append(parse_map_response(response.text))

reduce_prompt = build_prompts.build_chat_reduce_prompt(...)
reduce_response = _get_llm().generate(reduce_prompt)
llm_result = normalize_reduce_response(reduce_response.text)
```

If `len(map_results) == 1`, reducer can be skipped if output already fits target schema.

- [ ] **Step 5: Implement split retry**

When a map response is invalid JSON or appears truncated:
- if batch has more than one candidate, split and retry left/right;
- if batch has one candidate, compress candidate and retry once;
- if still failing, log `chat_map_batch_failed` and continue with an empty result for that candidate.

- [ ] **Step 6: Implement compact final answer mode**

In `answer_node`, after final IDs are known:

```python
if llm_result.get("answer_mode") == "compact_id_name":
    candidates = [{"id": c.get("id"), "full_name": c.get("full_name")} for c in candidates]
    prompt = build_prompts.build_compact_answer_prompt(...)
else:
    prompt = build_prompts.build_answer_prompt(...)
```

- [ ] **Step 7: Run chat tests**

Run:

```bash
pytest backend/tests/test_ai_agent_nodes.py backend/tests/test_job_chat_endpoint.py backend/tests/test_job_chat_trace_logging.py -v
```

Expected: pass.

---

### Task 7: Structured Logging For Scoring

**Files:**
- Modify `backend/src/services/score_candidate.py`
- Modify `backend/src/services/scoring_debug.py` only if helper methods are needed
- Extend `backend/tests/test_score_candidate_error_handling.py`

- [ ] **Step 1: Write failing log tests**

Add tests that run a successful scoring flow and inspect JSONL events for:
- `adaptive_batch_plan_created`
- `candidate_batch_started`
- `semantic_criteria_batch_started`
- `semantic_criteria_batch_completed`
- `candidate_batch_persist_started`
- `candidate_batch_persist_completed`
- `scoring_run_completed`

Required payload fields:

```json
{
  "candidateCount": 123,
  "candidateBatchCount": 12,
  "criteriaBatchCount": 3,
  "estimatedInputTokens": 1234,
  "estimatedOutputTokens": 567,
  "inputBudgetTokens": 5632,
  "outputBudgetTokens": 4096,
  "candidateIds": ["..."],
  "criterionKeys": ["..."],
  "durationMs": 123.4,
  "retryCount": 0,
  "splitDepth": 0,
  "provider": "groq",
  "model": "llama-3.1-8b-instant"
}
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest backend/tests/test_score_candidate_error_handling.py -v
```

Expected: fail because new events do not exist.

- [ ] **Step 3: Add timing helpers**

Use `time.perf_counter()` around:
- rubric extraction
- batch planning
- each semantic LLM call
- each fallback batch scoring LLM call
- persistence per batch

- [ ] **Step 4: Log prompt and response previews without leaking full data by default**

Current `preview_text` is fine. For full debug, include:
- full prompt length,
- preview text,
- response length,
- provider/model,
- usage if available.

Do not log raw full CV text unless current debug trace already does so and the trace directory is local-only.

- [ ] **Step 5: Run scoring log tests**

Run:

```bash
pytest backend/tests/test_score_candidate_error_handling.py backend/tests/test_score_candidate_service.py -v
```

Expected: pass.

---

### Task 8: Structured Logging For Chat

**Files:**
- Modify `backend/src/services/ai_agent/nodes.py`
- Modify `backend/src/services/ai_agent/langgraph_trace.py` only if serialization gaps appear
- Extend `backend/tests/test_job_chat_trace_logging.py`

- [ ] **Step 1: Write failing trace tests**

Add tests that inspect the trace file for events:
- `chat_candidate_scope_loaded`
- `chat_map_plan_created`
- `chat_map_batch_started`
- `chat_map_batch_completed`
- `chat_map_batch_split`
- `chat_reduce_started`
- `chat_reduce_completed`
- `chat_answer_mode_selected`
- `chat_answer_prompt_built`

Required payload fields:

```json
{
  "questionLength": 123,
  "candidateScopeCount": 500,
  "dslCandidateCount": 120,
  "llmCandidateCount": 120,
  "mapBatchCount": 8,
  "batchIndex": 0,
  "candidateIds": ["..."],
  "estimatedInputTokens": 3000,
  "estimatedOutputTokens": 600,
  "inputBudgetTokens": 5632,
  "outputBudgetTokens": 4096,
  "durationMs": 123.4,
  "qualifiedCount": 12,
  "answerMode": "compact_id_name",
  "omittedCount": 70,
  "provider": "groq",
  "model": "llama-3.1-8b-instant",
  "usage": {}
}
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest backend/tests/test_job_chat_trace_logging.py -v
```

Expected: fail because new trace events do not exist.

- [ ] **Step 3: Add trace event helper**

In `nodes.py`, add:

```python
def _record_chat_event(state: dict, event_type: str, payload: dict) -> None:
    trace_id = state.get("trace_id")
    if trace_id:
        get_trace_logger().record_event(trace_id=trace_id, event_type=event_type, payload=payload)
```

- [ ] **Step 4: Instrument every stage**

Log before and after:
- candidate resolve
- batch planning
- each map call
- split retry
- reduce
- answer mode selection
- final answer call

- [ ] **Step 5: Run chat trace tests**

Run:

```bash
pytest backend/tests/test_job_chat_trace_logging.py backend/tests/test_ai_agent_nodes.py -v
```

Expected: pass.

---

### Task 9: End-To-End Verification And Rollout

**Files:**
- No required code files unless test failures reveal gaps.

- [ ] **Step 1: Run targeted backend tests**

Run:

```bash
pytest backend/tests/test_token_budget.py backend/tests/test_scoring_batching.py backend/tests/test_chat_batching.py -v
```

Expected: pass.

- [ ] **Step 2: Run scoring tests**

Run:

```bash
pytest backend/tests/test_score_candidate_service.py backend/tests/test_score_candidate_error_handling.py backend/tests/test_score_endpoint.py backend/tests/test_jobs_score_endpoint.py -v
```

Expected: pass.

- [ ] **Step 3: Run chat tests**

Run:

```bash
pytest backend/tests/test_ai_agent_nodes.py backend/tests/test_job_chat_endpoint.py backend/tests/test_job_chat_trace_logging.py -v
```

Expected: pass.

- [ ] **Step 4: Run frontend type/lint checks if API response shape changes**

Run:

```bash
npm run typecheck
npm run lint
```

Working directory: `frontend`

Expected: pass.

- [ ] **Step 5: Manual log review**

Run one scoring request and one job chat request locally. Inspect:

```text
logs/scoring/YYYY-MM-DD/<match_run_id>.jsonl
logs/langgraph/YYYY-MM-DD/<trace_id>.json
```

Confirm:
- batch counts are visible;
- estimated token budgets are visible;
- each LLM call has duration and provider/model;
- compact answer mode is visible when candidate result set is large;
- errors include split depth and retry count.

---

## Rollout Notes

- Keep existing public API shape unchanged.
- Keep frontend batch size hidden.
- Add adaptive batching behind code-level defaults first; feature flags can be added later if needed.
- Do not remove existing `candidate_limit`; treat it as DB load bound, not LLM batch size.
- Store full matched candidate IDs outside final prompts in `QueryTurn.matched_candidate_ids`.
- Use compact final-answer prompts for large result sets to avoid context-window failure.
- Avoid logging complete raw prompts in production unless an explicit debug mode is enabled later.
