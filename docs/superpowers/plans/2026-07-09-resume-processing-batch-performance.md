# Resume Processing Batch Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Process demo uploads of 5-10 resumes with five concurrent parse tasks, one rubric extraction per JD signature, and token-safe batch evaluation.

**Architecture:** Persist each upload as a `ResumeProcessingBatch`, route parse and evaluation work to dedicated Celery queues, and trigger one idempotent evaluation task after every resume reaches a terminal parse state. Extract a pure multi-candidate scoring engine from the existing adaptive scoring path so `CandidateEvaluation` snapshots can share one rubric without creating legacy `MatchRun` side effects.

**Tech Stack:** FastAPI, SQLAlchemy 2, Alembic, Celery, Redis, PostgreSQL, pytest, Docker Compose.

---

## File Map

Create:

- `backend/src/models/resume_processing_batch.py`: durable batch state and relationships.
- `backend/src/services/resume_batch_service.py`: batch creation, reconciliation, dispatch, recovery, and state transitions.
- `backend/tests/test_resume_batch_service.py`: lifecycle and duplicate-delivery tests.
- `backend/tests/test_resume_batch_evaluation.py`: multi-candidate persistence and partial-failure tests.
- `backend/tests/test_resume_batch_upload.py`: endpoint contract for one processing batch per request.
- `backend/migrations/versions/20260709_0017_add_resume_processing_batches.py`: schema migration.

Modify:

- `backend/src/models/enums.py`: add `ResumeProcessingBatchStatus`.
- `backend/src/models/resume_document.py`: add nullable batch foreign key.
- `backend/src/models/entities.py`: export the new model through the model aggregation module.
- `backend/src/models/__init__.py`: expose the model and enum.
- `backend/src/services/resume_service.py`: accept `processing_batch_id` when creating a resume row.
- `backend/src/services/score_candidate.py`: add the pure multi-candidate raw evaluation engine.
- `backend/src/services/candidate_evaluation_service.py`: persist one batch of raw evaluations.
- `backend/src/api/v1/endpoints/jobs.py`: create one batch per multi-file upload.
- `backend/src/api/v1/endpoints/public_jobs.py`: use a batch of one for public applications.
- `backend/worker/tasks.py`: finalize parse batches, evaluate batches, and recover pending dispatches.
- `backend/worker/celery_app.py`: route queues and schedule recovery.
- `backend/src/core/config.py`: feature flag and worker-related defaults.
- `.env.example`: document batch pipeline settings and use the current GPT parse model.
- `docker-compose.yml`: add parse, evaluation, default, and beat processes.
- `docker-compose.prod.yml`: mirror the production worker topology.
- `frontend/src/api/types.ts`: accept additive `processing_batch_id`.
- Existing tests that stub `process_resume` or manually create model tables.

The current worktree already contains user changes in `backend/worker/tasks.py`,
`docker-compose.yml`, and `docker-compose.prod.yml`. Read and merge those changes;
do not replace either file wholesale.

### Task 1: Add Durable Batch Persistence

**Files:**

- Create: `backend/src/models/resume_processing_batch.py`
- Create: `backend/migrations/versions/20260709_0017_add_resume_processing_batches.py`
- Modify: `backend/src/models/enums.py`
- Modify: `backend/src/models/resume_document.py`
- Modify: `backend/src/models/entities.py`
- Modify: `backend/src/models/__init__.py`
- Test: `backend/tests/test_resume_batch_service.py`

- [ ] **Step 1: Write the failing model test**

Create a SQLite model test that imports all participating models and verifies the
relationship and default state:

```python
def test_resume_processing_batch_groups_resume_documents(db, owner, job):
    batch = ResumeProcessingBatch(
        job_id=job.id,
        total_count=2,
        status=ResumeProcessingBatchStatus.PARSING,
    )
    db.add(batch)
    db.flush()

    resumes = [
        make_resume(job=job, owner=owner, processing_batch_id=batch.id),
        make_resume(job=job, owner=owner, processing_batch_id=batch.id),
    ]
    db.add_all(resumes)
    db.commit()
    db.refresh(batch)

    assert batch.status == ResumeProcessingBatchStatus.PARSING
    assert {resume.id for resume in batch.resume_documents} == {
        resume.id for resume in resumes
    }
```

The local fixture must create tables in dependency order:
`user_accounts`, `jobs`, `resume_processing_batches`, `resume_documents`.

- [ ] **Step 2: Run the test and confirm the model is absent**

Run:

```powershell
$env:TMP="$PWD\.codex-tmp"
$env:TEMP="$PWD\.codex-tmp"
python -m pytest backend/tests/test_resume_batch_service.py -q -p no:cacheprovider
```

Expected: collection fails because `ResumeProcessingBatch` does not exist.

- [ ] **Step 3: Add the enum and ORM model**

Add this enum:

```python
class ResumeProcessingBatchStatus(str, Enum):
    PARSING = "parsing"
    EVALUATION_PENDING = "evaluation_pending"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"
```

Implement the model with non-negative counters and timestamps:

```python
class ResumeProcessingBatch(Base):
    __tablename__ = "resume_processing_batches"
    __table_args__ = (
        CheckConstraint("total_count >= 1", name="ck_resume_batches_total_positive"),
        CheckConstraint("terminal_count >= 0", name="ck_resume_batches_terminal_nonnegative"),
        CheckConstraint("processed_count >= 0", name="ck_resume_batches_processed_nonnegative"),
        CheckConstraint("failed_count >= 0", name="ck_resume_batches_failed_nonnegative"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    total_count: Mapped[int] = mapped_column(Integer, nullable=False)
    terminal_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    processed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    failed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    status: Mapped[ResumeProcessingBatchStatus] = mapped_column(
        SqlEnum(
            ResumeProcessingBatchStatus,
            name="resume_processing_batch_status_enum",
            values_callable=_ENUM_VALUES,
        ),
        nullable=False,
        default=ResumeProcessingBatchStatus.PARSING,
        server_default=ResumeProcessingBatchStatus.PARSING.value,
    )
    evaluation_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    evaluation_dispatch_attempted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    resume_documents: Mapped[list["ResumeDocument"]] = relationship(
        back_populates="processing_batch"
    )
```

Add nullable `processing_batch_id` to `ResumeDocument` with `ondelete="SET NULL"`
and `index=True`, plus the matching `processing_batch` relationship.

- [ ] **Step 4: Add the Alembic migration**

The migration must:

1. Create PostgreSQL enum `resume_processing_batch_status_enum`.
2. Create `resume_processing_batches` with all checks and indexes.
3. Add nullable `resume_documents.processing_batch_id`.
4. Add its foreign key and index.
5. Reverse those operations in downgrade order.

Use revision `20260709_0017` and `down_revision = "20260708_0016"`.

- [ ] **Step 5: Run model and migration checks**

Run:

```powershell
python -m pytest backend/tests/test_resume_batch_service.py -q -p no:cacheprovider
docker compose run --rm backend alembic -c alembic.ini upgrade head
docker compose run --rm backend alembic -c alembic.ini current
```

Expected: model test passes and Alembic reports `20260709_0017 (head)`.

- [ ] **Step 6: Commit persistence**

```powershell
git add backend/src/models backend/migrations/versions/20260709_0017_add_resume_processing_batches.py backend/tests/test_resume_batch_service.py
git commit -m "feat: add resume processing batch persistence"
```

### Task 2: Implement Idempotent Batch Lifecycle and Dispatch

**Files:**

- Create: `backend/src/services/resume_batch_service.py`
- Modify: `backend/tests/test_resume_batch_service.py`

- [ ] **Step 1: Add failing lifecycle tests**

Cover these cases with committed DB state:

```python
def test_reconcile_moves_terminal_batch_to_evaluation_pending(db, parsed_batch):
    mark_resume_statuses(parsed_batch, ["processed", "processed", "failed"])

    transition = reconcile_batch_after_parse(db, parsed_batch.id)

    assert transition.should_dispatch is True
    assert transition.processed_candidate_count == 2
    assert parsed_batch.status == ResumeProcessingBatchStatus.EVALUATION_PENDING
    assert parsed_batch.terminal_count == 3
    assert parsed_batch.failed_count == 1


def test_reconcile_duplicate_completion_does_not_reopen_batch(db, parsed_batch):
    mark_resume_statuses(parsed_batch, ["processed", "processed"])
    first = reconcile_batch_after_parse(db, parsed_batch.id)
    second = reconcile_batch_after_parse(db, parsed_batch.id)

    assert first.should_dispatch is True
    assert second.should_dispatch is False


def test_claim_dispatch_allows_stale_recovery(db, pending_batch, frozen_time):
    assert claim_evaluation_dispatch(db, pending_batch.id, stale_after_seconds=15)
    assert not claim_evaluation_dispatch(db, pending_batch.id, stale_after_seconds=15)

    frozen_time.tick(16)

    assert claim_evaluation_dispatch(db, pending_batch.id, stale_after_seconds=15)
```

- [ ] **Step 2: Run lifecycle tests and verify failure**

Run:

```powershell
python -m pytest backend/tests/test_resume_batch_service.py -q -p no:cacheprovider
```

Expected: failures for missing lifecycle functions.

- [ ] **Step 3: Implement focused service interfaces**

Implement these public interfaces:

```python
@dataclass(frozen=True)
class BatchParseTransition:
    batch_id: uuid.UUID
    should_dispatch: bool
    processed_candidate_count: int


def create_processing_batch(
    *,
    db: Session,
    job_id: uuid.UUID,
    total_count: int,
) -> ResumeProcessingBatch:
    if total_count < 1:
        raise ValueError("Resume processing batch must contain at least one file")
    batch = ResumeProcessingBatch(job_id=job_id, total_count=total_count)
    db.add(batch)
    db.flush()
    return batch


def reconcile_batch_after_parse(
    db: Session,
    processing_batch_id: uuid.UUID,
) -> BatchParseTransition:
    batch = (
        db.query(ResumeProcessingBatch)
        .filter(ResumeProcessingBatch.id == processing_batch_id)
        .with_for_update()
        .one()
    )
    statuses = [
        _status_value(status)
        for (status,) in db.query(ResumeDocument.upload_status)
        .filter(ResumeDocument.processing_batch_id == batch.id)
        .all()
    ]
    batch.processed_count = statuses.count(UploadStatus.PROCESSED.value)
    batch.failed_count = statuses.count(UploadStatus.FAILED.value)
    batch.terminal_count = batch.processed_count + batch.failed_count
    should_dispatch = (
        batch.status == ResumeProcessingBatchStatus.PARSING
        and batch.terminal_count == batch.total_count
        and batch.processed_count > 0
    )
    if should_dispatch:
        batch.status = ResumeProcessingBatchStatus.EVALUATION_PENDING
    elif batch.terminal_count == batch.total_count and batch.processed_count == 0:
        batch.status = ResumeProcessingBatchStatus.FAILED
        batch.completed_at = datetime.now(timezone.utc)
    db.commit()
    return BatchParseTransition(batch.id, should_dispatch, batch.processed_count)
```

Implement `claim_evaluation_dispatch`, `record_evaluation_task_id`, and
`list_recoverable_evaluation_batches`. A dispatch claim sets
`evaluation_dispatch_attempted_at`; it succeeds only when the batch is pending
and either never attempted or older than `stale_after_seconds`.

- [ ] **Step 4: Run lifecycle tests**

Run:

```powershell
python -m pytest backend/tests/test_resume_batch_service.py -q -p no:cacheprovider
```

Expected: all lifecycle tests pass.

- [ ] **Step 5: Commit lifecycle service**

```powershell
git add backend/src/services/resume_batch_service.py backend/tests/test_resume_batch_service.py
git commit -m "feat: add idempotent resume batch lifecycle"
```

### Task 3: Create One Batch Per Upload Request

**Files:**

- Modify: `backend/src/core/config.py`
- Modify: `.env.example`
- Modify: `backend/src/services/resume_service.py`
- Modify: `backend/src/api/v1/endpoints/jobs.py`
- Modify: `backend/src/api/v1/endpoints/public_jobs.py`
- Modify: `frontend/src/api/types.ts`
- Create: `backend/tests/test_resume_batch_upload.py`
- Modify: `backend/tests/test_job_resume_endpoints.py`
- Modify: `backend/tests/test_public_job_endpoints.py`

- [ ] **Step 1: Write failing upload contract tests**

For a three-file authenticated upload, capture `process_resume.delay` calls and
assert:

```python
assert response.status_code == 202
payload = response.json()
assert payload["total_files"] == 3
assert payload["queued_files"] == 3
assert uuid.UUID(payload["processing_batch_id"])
assert len({call.kwargs["processing_batch_id"] for call in queued_calls}) == 1
assert queued_calls[0].kwargs["processing_batch_id"] == payload["processing_batch_id"]
```

Add a public application test asserting one batch with `total_count == 1`.

- [ ] **Step 2: Run upload tests and confirm failure**

Run:

```powershell
python -m pytest backend/tests/test_resume_batch_upload.py backend/tests/test_job_resume_endpoints.py backend/tests/test_public_job_endpoints.py -q -p no:cacheprovider
```

Expected: response lacks `processing_batch_id` and tasks lack the batch argument.

- [ ] **Step 3: Add configuration and resume creation support**

Add:

```python
BATCH_RESUME_PIPELINE_ENABLED: bool = os.getenv(
    "BATCH_RESUME_PIPELINE_ENABLED", "true"
).lower() in {"1", "true", "yes", "on"}
RESUME_BATCH_DISPATCH_STALE_SECONDS: int = int(
    os.getenv("RESUME_BATCH_DISPATCH_STALE_SECONDS", "15")
)
```

Extend `create_resume_document`:

```python
def create_resume_document(
    *,
    db: Session,
    storage_uri: str,
    original_file_name: str,
    job_id: uuid.UUID,
    uploaded_by_user_id: uuid.UUID,
    retention_days: int = 365,
    processing_batch_id: uuid.UUID | None = None,
) -> ResumeDocument:
    resume = ResumeDocument(
        original_file_name=original_file_name,
        storage_uri=storage_uri,
        job_id=job_id,
        uploaded_by_user_id=uploaded_by_user_id,
        retention_expires_at=datetime.now(timezone.utc) + timedelta(days=retention_days),
        processing_batch_id=processing_batch_id,
    )
    db.add(resume)
    db.commit()
    db.refresh(resume)
    return resume
```

- [ ] **Step 4: Wire authenticated and public upload paths**

When the flag is enabled:

1. Create the batch before creating resume rows.
2. Pass `processing_batch_id` into each resume.
3. Pass its string value to `process_resume.delay`.
4. Return additive `processing_batch_id`.

When disabled, preserve the existing call signature and per-candidate evaluation
behavior.

Update the frontend response type:

```typescript
export interface ResumeBatchParseResponse {
  processing_batch_id?: string | null;
  total_files: number;
  queued_files: number;
  items: ResumeBatchParseItem[];
}
```

- [ ] **Step 5: Run endpoint contract tests**

Run:

```powershell
python -m pytest backend/tests/test_resume_batch_upload.py backend/tests/test_job_resume_endpoints.py backend/tests/test_public_job_endpoints.py -q -p no:cacheprovider
npm --prefix frontend run typecheck
```

Expected: tests and typecheck pass.

- [ ] **Step 6: Commit upload wiring**

```powershell
git add .env.example backend/src/core/config.py backend/src/services/resume_service.py backend/src/api/v1/endpoints/jobs.py backend/src/api/v1/endpoints/public_jobs.py frontend/src/api/types.ts backend/tests/test_resume_batch_upload.py backend/tests/test_job_resume_endpoints.py backend/tests/test_public_job_endpoints.py
git commit -m "feat: group resume uploads into processing batches"
```

### Task 4: Extract the Pure Multi-Candidate Scoring Engine

**Files:**

- Modify: `backend/src/services/score_candidate.py`
- Create: `backend/tests/test_resume_batch_evaluation.py`
- Modify: `backend/tests/test_score_candidate_error_handling.py`

- [ ] **Step 1: Add a failing rubric-reuse test**

Stub rubric extraction and semantic scoring, then call the new engine with ten
candidates:

```python
results = evaluate_candidate_profiles_raw(
    candidates=ten_candidates,
    job_description_text="Need Python and five years of experience",
)

assert extract_rubric.call_count == 1
assert set(results) == {candidate["id"] for candidate in ten_candidates}
assert 1 <= semantic_generate.call_count <= 3
```

Also assert a rubric containing only measurable criteria causes zero semantic
LLM calls.

- [ ] **Step 2: Run scoring tests and verify failure**

Run:

```powershell
python -m pytest backend/tests/test_resume_batch_evaluation.py backend/tests/test_score_candidate_error_handling.py -q -p no:cacheprovider
```

Expected: import failure for `evaluate_candidate_profiles_raw`.

- [ ] **Step 3: Implement the batch engine**

Add:

```python
def evaluate_candidate_profiles_raw(
    *,
    candidates: list[dict[str, Any]],
    job_description_text: str,
    section_weights: Optional[dict[str, float]] = None,
    debug_logger: Optional[ScoringDebugLogger] = None,
) -> dict[str, dict[str, Any]]:
    if not candidates:
        return {}

    llm = _scoring_llm_provider()
    rubric = _extract_locked_rubric(
        llm=llm,
        job_description_text=job_description_text,
        section_weights=section_weights,
        debug_logger=debug_logger,
    )
    if not rubric or not rubric.get("criteria"):
        return {
            str(candidate["id"]): {
                "rubricPayload": {"criteria": [], "sectionWeights": {}},
                "rawComponentScores": [],
                "rationaleSummary": _build_rationale_summary(0.0, []),
            }
            for candidate in candidates
        }

    semantic_criteria = [
        criterion
        for criterion in rubric["criteria"]
        if not criterion.get("measurable")
    ]
    window = BudgetWindow(
        context_window=settings.SCORING_CONTEXT_WINDOW_TOKENS,
        output_budget=settings.SCORING_OUTPUT_TOKEN_BUDGET,
        reserve=settings.SCORING_CONTEXT_RESERVE_TOKENS,
    )
    static_tokens = estimate_tokens(job_description_text) + estimate_json_tokens(rubric)
    plan = build_scoring_batch_plan(
        candidates=candidates,
        semantic_criteria=semantic_criteria,
        static_prompt_tokens=static_tokens,
        window=window,
        max_candidates_per_batch=settings.SCORING_MAX_CANDIDATES_PER_BATCH,
        max_criteria_per_call=settings.SCORING_MAX_SEMANTIC_CRITERIA_PER_CALL,
    )
    semantic_by_candidate: dict[str, dict[str, Any]] = {}
    for candidate_batch in plan.candidate_batches:
        for criterion_batch in plan.criterion_batches:
            update = _generate_semantic_scores_with_retries(
                llm=llm,
                prompt=build_prompts.build_locked_rubric_semantic_scoring_prompt(
                    candidates=candidate_batch.candidates,
                    rubric={"criteria": criterion_batch},
                ),
                debug_logger=debug_logger,
            )
            _merge_semantic_scores(semantic_by_candidate, update)

    return {
        str(candidate["id"]): _raw_evaluation_payload(
            candidate=candidate,
            rubric=rubric,
            semantic_result=semantic_by_candidate.get(str(candidate["id"]), {}),
            debug_logger=debug_logger,
        )
        for candidate in candidates
    }
```

Extract `_raw_evaluation_payload` from the tail of
`evaluate_candidate_profile_raw`. Make the single-candidate function delegate to
the batch function and return its one result, preserving its public contract.

- [ ] **Step 4: Run scoring tests**

Run:

```powershell
python -m pytest backend/tests/test_resume_batch_evaluation.py backend/tests/test_score_candidate_error_handling.py backend/tests/test_candidate_evaluations.py -q -p no:cacheprovider
```

Expected: one rubric extraction and all tests pass.

- [ ] **Step 5: Commit scoring engine**

```powershell
git add backend/src/services/score_candidate.py backend/tests/test_resume_batch_evaluation.py backend/tests/test_score_candidate_error_handling.py backend/tests/test_candidate_evaluations.py
git commit -m "perf: reuse scoring rubric across candidate batches"
```

### Task 5: Persist Batch Candidate Evaluations

**Files:**

- Modify: `backend/src/services/candidate_evaluation_service.py`
- Modify: `backend/src/services/resume_batch_service.py`
- Modify: `backend/tests/test_resume_batch_evaluation.py`

- [ ] **Step 1: Add failing persistence and retry tests**

Test:

```python
result = evaluate_processing_batch(
    db=db,
    processing_batch_id=batch.id,
    worker_task_id="task-1",
)

assert result.completed == 8
assert result.failed == 0
assert batch.status == ResumeProcessingBatchStatus.COMPLETED_WITH_ERRORS
assert raw_batch_engine.call_count == 1
assert len(db.query(CandidateEvaluation).all()) == 8
```

The fixture contains ten resumes: eight processed with profiles and two failed.
Add another test with two pre-existing completed evaluations; assert those IDs
are omitted from the batch engine input and remain unchanged.

- [ ] **Step 2: Run the tests and confirm failure**

Run:

```powershell
python -m pytest backend/tests/test_resume_batch_evaluation.py -q -p no:cacheprovider
```

Expected: `evaluate_processing_batch` is absent.

- [ ] **Step 3: Implement claim, scoring, and persistence**

Add result type:

```python
@dataclass(frozen=True)
class ProcessingBatchEvaluationResult:
    batch_id: uuid.UUID
    completed: int
    failed: int
    skipped: int
```

Implement `evaluate_processing_batch` in three transaction phases:

1. Lock and claim `evaluation_pending -> evaluating`; allow re-entry only for
   the same `worker_task_id`.
2. Commit before the LLM call, load parsed profiles and skip completed snapshots.
3. Persist each returned result independently, then set the final batch status.

Use:

```python
raw_results = score_candidate.evaluate_candidate_profiles_raw(
    candidates=[
        score_candidate._profile_to_candidate_dict(profile)
        for profile in profiles_to_evaluate
    ],
    job_description_text=score_candidate._build_scoring_job_description_text(
        public_job_description=jd.jd_text,
        hidden_text=jd.hidden_text,
    ),
)
```

For every profile, upsert by `(job_description_id, candidate_profile_id,
scoring_signature)`. Set successful rows to `completed`; if a candidate ID is
missing from `raw_results`, set only that row to `failed`.

Final status:

```python
batch.status = (
    ResumeProcessingBatchStatus.COMPLETED_WITH_ERRORS
    if batch.failed_count or failed_evaluations
    else ResumeProcessingBatchStatus.COMPLETED
)
batch.completed_at = datetime.now(timezone.utc)
```

- [ ] **Step 4: Run persistence tests**

Run:

```powershell
python -m pytest backend/tests/test_resume_batch_evaluation.py backend/tests/test_candidate_evaluations.py -q -p no:cacheprovider
```

Expected: all tests pass and completed snapshots are skipped.

- [ ] **Step 5: Commit evaluation persistence**

```powershell
git add backend/src/services/candidate_evaluation_service.py backend/src/services/resume_batch_service.py backend/tests/test_resume_batch_evaluation.py
git commit -m "feat: persist batched candidate evaluations"
```

### Task 6: Orchestrate Parse Completion, Evaluation, and Recovery

**Files:**

- Modify: `backend/worker/tasks.py`
- Modify: `backend/tests/test_resume_scoring_trigger.py`
- Modify: `backend/tests/conftest.py`

- [ ] **Step 1: Write failing worker orchestration tests**

Cover:

```python
def test_process_resume_reconciles_batch_instead_of_queueing_single_candidate(...):
    result = process_resume.run(resume_id, processing_batch_id=batch_id)
    assert result["status"] == "completed"
    reconcile.assert_called_once_with(batch_id)
    legacy_queue.assert_not_called()


def test_evaluate_resume_batch_retries_provider_failure(...):
    with pytest.raises(ExpectedRetry):
        evaluate_resume_batch.run(batch_id)
    mark_terminal_failure.assert_not_called()


def test_recovery_republishes_only_stale_pending_batches(...):
    result = recover_pending_resume_batches.run()
    assert result == {"dispatched": 2}
```

Retain tests proving the feature-flag-disabled path still queues the legacy
single-candidate task.

- [ ] **Step 2: Run worker tests and verify failure**

Run:

```powershell
python -m pytest backend/tests/test_resume_scoring_trigger.py -q -p no:cacheprovider
```

Expected: task signatures and recovery task are missing.

- [ ] **Step 3: Extend `process_resume` safely**

Add optional `processing_batch_id`. On successful or returned-failed parse,
reconcile the batch. On an exception, only reconcile terminal failure when
`self.request.retries >= self.max_retries`; otherwise retry without closing the
batch.

Dispatch after the reconciliation transaction commits:

```python
transition = reconcile_batch_after_parse(db, uuid.UUID(processing_batch_id))
if transition.should_dispatch:
    dispatch_evaluation_batch(uuid.UUID(processing_batch_id))
```

If no batch ID is supplied or the feature flag is disabled, keep
`queue_candidate_evaluation_for_current_jd`.

- [ ] **Step 4: Add evaluation and recovery tasks**

Add:

```python
@celery_app.task(
    bind=True,
    name="worker.tasks.evaluate_resume_batch",
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
)
def evaluate_resume_batch(self, processing_batch_id: str):
    with SessionLocal() as db:
        return evaluate_processing_batch(
            db=db,
            processing_batch_id=uuid.UUID(processing_batch_id),
            worker_task_id=str(self.request.id or ""),
        ).__dict__


@celery_app.task(name="worker.tasks.recover_pending_resume_batches")
def recover_pending_resume_batches():
    with SessionLocal() as db:
        batch_ids = list_recoverable_evaluation_batches(
            db,
            stale_after_seconds=settings.RESUME_BATCH_DISPATCH_STALE_SECONDS,
        )
    dispatched = sum(dispatch_evaluation_batch(batch_id) for batch_id in batch_ids)
    return {"dispatched": dispatched}
```

Wrap `evaluate_resume_batch` with the existing retry style. On the final failed
attempt, call `mark_processing_batch_failed` so candidate rows and batch status
do not remain running forever.

- [ ] **Step 5: Run worker tests**

Run:

```powershell
python -m pytest backend/tests/test_resume_scoring_trigger.py backend/tests/test_resume_batch_service.py backend/tests/test_resume_batch_evaluation.py -q -p no:cacheprovider
```

Expected: all task, recovery, and idempotency tests pass.

- [ ] **Step 6: Commit worker orchestration**

```powershell
git add backend/worker/tasks.py backend/tests/test_resume_scoring_trigger.py backend/tests/conftest.py
git commit -m "feat: orchestrate resume batch evaluation tasks"
```

### Task 7: Split Celery Queues and Worker Topology

**Files:**

- Modify: `backend/worker/celery_app.py`
- Modify: `docker-compose.yml`
- Modify: `docker-compose.prod.yml`
- Modify: `.env.example`
- Create: `backend/tests/test_celery_routing.py`

- [ ] **Step 1: Add failing routing tests**

Assert:

```python
assert routes["worker.tasks.process_resume"]["queue"] == "resume_parse"
assert routes["worker.tasks.evaluate_resume_batch"]["queue"] == "candidate_evaluation"
assert routes["worker.tasks.evaluate_candidate"]["queue"] == "candidate_evaluation"
assert routes["worker.tasks.send_outreach_email"]["queue"] == "default"
assert beat["recover-pending-resume-batches"]["task"] == (
    "worker.tasks.recover_pending_resume_batches"
)
```

- [ ] **Step 2: Run routing test and confirm failure**

Run:

```powershell
python -m pytest backend/tests/test_celery_routing.py -q -p no:cacheprovider
```

Expected: all tasks currently route to `default`.

- [ ] **Step 3: Configure routes and recovery schedule**

Use explicit routes before the wildcard:

```python
task_routes={
    "worker.tasks.process_resume": {"queue": "resume_parse"},
    "worker.tasks.evaluate_resume_batch": {"queue": "candidate_evaluation"},
    "worker.tasks.evaluate_candidate": {"queue": "candidate_evaluation"},
    "worker.tasks.*": {"queue": "default"},
},
beat_schedule={
    "recover-pending-resume-batches": {
        "task": "worker.tasks.recover_pending_resume_batches",
        "schedule": 15.0,
    },
},
```

- [ ] **Step 4: Add Compose processes**

Development and production need:

```yaml
worker:
  command: celery -A worker.celery_app worker -Q default --concurrency=2 --loglevel=info

resume-worker:
  command: celery -A worker.celery_app worker -Q resume_parse --concurrency=${RESUME_PARSE_WORKER_CONCURRENCY:-5} --loglevel=info

evaluation-worker:
  command: celery -A worker.celery_app worker -Q candidate_evaluation --concurrency=${CANDIDATE_EVALUATION_WORKER_CONCURRENCY:-1} --loglevel=info

beat:
  command: celery -A worker.celery_app beat --loglevel=info
```

Reuse the existing build, environment, dependency, and volume blocks. Do not
delete current interview/audio environment values from the dirty Compose files.

Document:

```dotenv
BATCH_RESUME_PIPELINE_ENABLED=true
RESUME_PARSE_WORKER_CONCURRENCY=5
CANDIDATE_EVALUATION_WORKER_CONCURRENCY=1
RESUME_BATCH_DISPATCH_STALE_SECONDS=15
RESUME_PARSE_MODEL_NAME=gpt-4.1-mini
```

- [ ] **Step 5: Validate routing and Compose**

Run:

```powershell
python -m pytest backend/tests/test_celery_routing.py -q -p no:cacheprovider
docker compose config --quiet
docker compose -f docker-compose.prod.yml config --quiet
```

Expected: routing test passes and both Compose files validate.

- [ ] **Step 6: Commit queue topology**

```powershell
git add backend/worker/celery_app.py docker-compose.yml docker-compose.prod.yml .env.example backend/tests/test_celery_routing.py
git commit -m "perf: isolate resume parsing and evaluation queues"
```

### Task 8: Add Observability and End-to-End Verification

**Files:**

- Modify: `backend/src/services/resume_batch_service.py`
- Modify: `backend/src/services/score_candidate.py`
- Modify: `backend/worker/tasks.py`
- Modify: `backend/tests/test_resume_batch_evaluation.py`
- Modify: `backend/tests/test_resume_batch_service.py`

- [ ] **Step 1: Add failing trace assertions**

Capture logs/events and assert the presence of:

```python
assert event["processing_batch_id"] == str(batch.id)
assert event["processed_count"] == 10
assert event["rubric_extraction_count"] == 1
assert 1 <= event["semantic_batch_count"] <= 3
assert event["total_duration_ms"] > 0
```

Tests must assert metadata only; CV text and hidden JD text must not appear.

- [ ] **Step 2: Run trace tests and confirm failure**

Run:

```powershell
python -m pytest backend/tests/test_resume_batch_service.py backend/tests/test_resume_batch_evaluation.py -q -p no:cacheprovider
```

Expected: new batch metrics are absent.

- [ ] **Step 3: Add structured timing**

Record:

- Upload-to-terminal parse duration.
- Evaluation duration.
- Rubric extraction count.
- Candidate batch count.
- Semantic criteria batch count.
- Completed and failed counts.

Use IDs, counts, and durations only. Reuse `ScoringDebugLogger.record_event` for
scoring details and standard structured logger fields for worker lifecycle.

- [ ] **Step 4: Run the focused backend suite**

Run:

```powershell
$env:TMP="$PWD\.codex-tmp"
$env:TEMP="$PWD\.codex-tmp"
python -m pytest backend/tests/test_resume_batch_service.py backend/tests/test_resume_batch_upload.py backend/tests/test_resume_batch_evaluation.py backend/tests/test_resume_scoring_trigger.py backend/tests/test_candidate_evaluations.py backend/tests/test_score_candidate_error_handling.py backend/tests/test_celery_routing.py -q -p no:cacheprovider
```

Expected: all focused tests pass.

- [ ] **Step 5: Run static and deployment verification**

Run:

```powershell
npm --prefix frontend run typecheck
docker compose config --quiet
docker compose -f docker-compose.prod.yml config --quiet
docker compose up -d --build redis db backend worker resume-worker evaluation-worker beat
docker compose ps
```

Expected: typecheck passes; every listed service is running.

- [ ] **Step 6: Run the 5- and 10-CV demo benchmark**

Upload fixed local CV fixtures through `POST /jobs/{job_id}/resumes`, then inspect
worker logs and `logs/scoring`. Record:

```text
5 CV: first parse, all parse terminal, all evaluation terminal, LLM call counts
10 CV: first parse, all parse terminal, all evaluation terminal, LLM call counts
```

Acceptance:

- At most five concurrent parse requests.
- Ten CVs run in no more than two parse scheduling waves.
- One rubric extraction per JD signature.
- Semantic scoring uses adaptive batches rather than ten independent calls.
- Partial failures do not block successful candidates.
- No duplicate `CandidateEvaluation` rows.

- [ ] **Step 7: Commit observability**

```powershell
git add backend/src/services/resume_batch_service.py backend/src/services/score_candidate.py backend/worker/tasks.py backend/tests/test_resume_batch_service.py backend/tests/test_resume_batch_evaluation.py
git commit -m "chore: trace resume batch throughput"
```

## Final Verification

- [ ] Run `git status --short` and confirm only pre-existing user changes remain.
- [ ] Run the complete backend suite with workspace-local temp directories:

```powershell
$env:TMP="$PWD\.codex-tmp"
$env:TEMP="$PWD\.codex-tmp"
python -m pytest backend/tests -q -p no:cacheprovider
```

- [ ] Run frontend verification:

```powershell
npm --prefix frontend run typecheck
npm --prefix frontend run lint
```

- [ ] Inspect runtime queues:

```powershell
docker compose exec redis redis-cli LLEN resume_parse
docker compose exec redis redis-cli LLEN candidate_evaluation
docker compose logs --tail 200 resume-worker evaluation-worker beat
```

- [ ] Compare the final implementation against
`docs/superpowers/specs/2026-07-09-resume-processing-batch-performance-design.md`
and confirm every acceptance criterion is represented by a passing test or
runtime observation.
