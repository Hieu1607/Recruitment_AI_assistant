# Scoring Fixed Batch Size Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the user-facing scoring batch size control and lock scoring batches to an internal default of `3` without showing that value on the frontend.

**Architecture:** Keep the change narrow to the current job-scoped scoring flow. The frontend setup screen stops collecting batch size, the scoring store injects a fixed internal constant, and the backend job-scoped request model defaults to `3` so omitted values behave consistently.

**Tech Stack:** React, TypeScript, Zustand, FastAPI, Pydantic, pytest

---

### Task 1: Lock the backend job-scoped scoring default to 3

**Files:**
- Modify: `backend/tests/test_jobs_score_endpoint.py`
- Modify: `backend/src/api/v1/endpoints/jobs.py`

- [ ] **Step 1: Write the failing test**

```python
def test_job_score_endpoint_defaults_batch_size_to_three(monkeypatch, db, owner):
    job_id = uuid.uuid4()
    jd = types.SimpleNamespace(id=uuid.uuid4())
    captured = {}

    monkeypatch.setattr(
        "src.api.v1.endpoints.jobs.require_job_scoped_jd",
        lambda *args, **kwargs: jd,
    )
    monkeypatch.setattr(
        "src.api.v1.endpoints.jobs.score_candidates",
        lambda **kwargs: captured.update(kwargs) or {},
    )

    score_job_candidates(
        job_id=job_id,
        body=ScoreRequest(),
        db=db,
        current_user=owner,
    )

    assert captured["batch_size"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_jobs_score_endpoint.py::test_job_score_endpoint_defaults_batch_size_to_three -v`
Expected: FAIL because the endpoint still forwards `10`

- [ ] **Step 3: Write minimal implementation**

```python
class ScoreRequest(BaseModel):
    score_threshold: float = Field(50.0, ge=0, le=100)
    candidate_profile_ids: Optional[list[uuid.UUID]] = None
    section_weights: Optional[dict[str, float]] = None
    batch_size: int = Field(3, ge=1, le=50)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_jobs_score_endpoint.py::test_job_score_endpoint_defaults_batch_size_to_three -v`
Expected: PASS

### Task 2: Remove the frontend batch size control and keep an internal constant

**Files:**
- Modify: `frontend/src/routes/scoring/setup.tsx`
- Modify: `frontend/src/lib/scoring-store.ts`
- Modify: `frontend/src/api/endpoints/jobs.ts`
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/components/UiLocalizer.tsx`

- [ ] **Step 1: Remove user-controlled batch size state and UI**

```tsx
const INTERNAL_SCORING_BATCH_SIZE = 3;

return startRun(selectedJobId, {
  scoreThreshold: threshold,
  sectionWeights,
  candidateProfileIds,
  hiddenTextSnapshot: hiddenText,
});
```

- [ ] **Step 2: Keep ETA based on the internal constant without rendering the value**

```tsx
const estSeconds = (() => {
  const n = candidateMode === "all" ? (resumeData?.total ?? 0) : selectedCandIds.size;
  return n > 0 ? Math.max(15, Math.ceil(n / INTERNAL_SCORING_BATCH_SIZE) * 15) : null;
})();
```

- [ ] **Step 3: Update store and API shapes**

```ts
const INTERNAL_SCORING_BATCH_SIZE = 3;

export interface StartScoringRunInput {
  scoreThreshold: number;
  sectionWeights: Record<string, number>;
  candidateProfileIds?: string[];
  hiddenTextSnapshot?: string;
}
```

- [ ] **Step 4: Remove stale localization strings and comments**

```ts
// Delete the "Batch size" and "candidates per LLM batch (1–50)" entries.
// Update ScoreRequest comments from default 10 to default 3 where retained.
```

- [ ] **Step 5: Verify frontend compiles**

Run: `npm run typecheck`
Working directory: `frontend`
Expected: PASS
