# Scoring Debug Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add structured file-based debug logs for candidate scoring runs so failures and score calculations can be reconstructed from `backend/logs/scoring/`.

**Architecture:** Introduce a small scoring-specific debug logger helper and instrument `score_candidates(...)` plus its retry and scoring helpers with structured events. Keep existing standard logger calls intact while adding targeted tests for success, retry, and failure paths.

**Tech Stack:** Python, pytest, SQLAlchemy, JSON file I/O

---

### Task 1: Add failing tests for scoring debug files

**Files:**
- Modify: `backend/tests/test_score_candidate_error_handling.py`
- Test: `backend/tests/test_score_candidate_error_handling.py`

- [ ] **Step 1: Write the failing test**

```python
def test_score_candidates_writes_debug_trace_for_success(...):
    ...
    assert trace_path.exists()
    assert "run_started" in events
    assert "candidate_scored" in events
    assert "run_completed" in events
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_score_candidate_error_handling.py -k debug_trace -v`
Expected: FAIL because scoring debug file output does not exist yet.

- [ ] **Step 3: Extend tests for failure and retry events**

```python
def test_score_candidates_writes_failure_event_for_provider_limit(...):
    ...
    assert "run_failed" in events

def test_score_candidates_logs_semantic_retry_attempts(...):
    ...
    assert "semantic_scoring_attempt" in events
```

- [ ] **Step 4: Run tests to verify they fail for the expected reason**

Run: `pytest backend/tests/test_score_candidate_error_handling.py -k "debug_trace or semantic_retry" -v`
Expected: FAIL with missing file or missing events, not fixture/setup errors.

### Task 2: Add the scoring debug helper

**Files:**
- Create: `backend/src/services/scoring_debug.py`
- Modify: `backend/src/services/score_candidate.py`
- Test: `backend/tests/test_score_candidate_error_handling.py`

- [ ] **Step 1: Write the minimal helper implementation**

```python
class ScoringDebugLogger:
    def __init__(self, match_run_id: uuid.UUID, base_dir: Path | None = None):
        ...

    def record_event(self, event: str, payload: dict[str, Any]) -> None:
        ...

    def finalize(self, status: str, payload: dict[str, Any] | None = None) -> None:
        ...
```

- [ ] **Step 2: Add safe JSON serialization and text preview helpers**

```python
def serialize_for_json(value: Any) -> Any:
    ...

def preview_text(text: str | None, limit: int = 1200) -> dict[str, Any]:
    ...
```

- [ ] **Step 3: Run focused tests**

Run: `pytest backend/tests/test_score_candidate_error_handling.py -k debug_trace -v`
Expected: Still FAIL because instrumentation in the scoring flow is not complete yet.

### Task 3: Instrument the scoring flow

**Files:**
- Modify: `backend/src/services/score_candidate.py`
- Test: `backend/tests/test_score_candidate_error_handling.py`

- [ ] **Step 1: Attach logger lifecycle to `score_candidates(...)`**

```python
debug_log = ScoringDebugLogger(match_run.id)
debug_log.record_event("run_started", {...})
```

- [ ] **Step 2: Emit rubric, semantic, fallback, and persistence events**

```python
debug_log.record_event("rubric_extraction_started", {...})
debug_log.record_event("semantic_scoring_attempt", {...})
debug_log.record_event("candidate_scored", {...})
debug_log.record_event("batch_persist_completed", {...})
```

- [ ] **Step 3: Capture terminal success and failure events**

```python
debug_log.finalize("completed", {...})
debug_log.finalize("failed", {...})
```

- [ ] **Step 4: Run focused tests**

Run: `pytest backend/tests/test_score_candidate_error_handling.py -k "debug_trace or semantic_retry" -v`
Expected: PASS for the new debug trace coverage.

### Task 4: Verify targeted scoring behavior

**Files:**
- Modify: none
- Test: `backend/tests/test_score_candidate_error_handling.py`

- [ ] **Step 1: Run the full related scoring test module**

Run: `pytest backend/tests/test_score_candidate_error_handling.py -v`
Expected: PASS

- [ ] **Step 2: Run endpoint tests that cover scoring integration**

Run: `pytest backend/tests/test_jobs_score_endpoint.py backend/tests/test_score_endpoint.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add backend/src/services/scoring_debug.py backend/src/services/score_candidate.py backend/tests/test_score_candidate_error_handling.py docs/superpowers/specs/2026-06-12-scoring-debug-logging-design.md docs/superpowers/plans/2026-06-12-scoring-debug-logging.md
git commit -m "feat: add scoring debug trace logs"
```
