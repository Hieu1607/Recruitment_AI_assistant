# Scoring Evaluation Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build candidate evaluation snapshots so LLM scoring runs once per candidate per JD signature, while job-level weights recalculate displayed scores without calling the LLM.

**Architecture:** Add persistent `CandidateEvaluation` snapshots and `JobScoringPreference` records, then move display scoring into a pure backend calculation function. Resume parsing and public applications enqueue per-candidate evaluation tasks; JD edits mark evaluations outdated and expose `Score again` actions through new job-scoped endpoints. Frontend scoring becomes a results dashboard backed by evaluation endpoints instead of a three-step scoring wizard.

**Tech Stack:** FastAPI, SQLAlchemy, Alembic, Celery, PostgreSQL JSONB, pytest, React, TypeScript, TanStack Query, Vite.

---

## File Structure

- Create `backend/src/models/scoring_evaluation.py`: SQLAlchemy models for `CandidateEvaluation` and `JobScoringPreference`.
- Modify `backend/src/models/enums.py`: add `CandidateEvaluationStatus`.
- Modify `backend/src/models/entities.py` and `backend/src/models/__init__.py`: export new models and enum.
- Create `backend/migrations/versions/20260707_0015_add_candidate_evaluations.py`: create new persistence tables.
- Create `backend/src/services/scoring_signature.py`: deterministic signature for JD scoring inputs and prompt/rule versions.
- Create `backend/src/services/scoring_preferences.py`: normalize job weights and compute weighted display scores.
- Create `backend/src/services/candidate_evaluation_service.py`: evaluation lifecycle, status transitions, score-again queueing, and serialization helpers.
- Modify `backend/src/services/score_candidate.py`: expose raw rubric evaluation helpers and update semantic prompt flow to store `scorePercent`.
- Modify `backend/src/prompts/build_prompts.py`: prompt asks LLM for every semantic criterion and no total score.
- Modify `backend/worker/tasks.py`: add `evaluate_candidate` Celery task and trigger it after successful `process_resume`.
- Modify `backend/src/api/v1/endpoints/jobs.py`: add evaluation/preference endpoints and mark outdated on JD changes.
- Modify `backend/tests/conftest.py`: stub `evaluate_candidate.delay`.
- Add backend tests under `backend/tests/test_candidate_evaluations.py`, `backend/tests/test_scoring_preferences.py`, and update scoring prompt tests.
- Modify `frontend/src/api/types.ts`: add evaluation and preference types.
- Modify `frontend/src/api/endpoints/jobs.ts`: add evaluations and scoring preference API methods.
- Rewrite `frontend/src/routes/scoring/setup.tsx`: results dashboard with weights and score-again.
- Modify `frontend/src/routes/candidates/detail.tsx`: add evaluation tab/section.
- Modify JD UI component, most likely `frontend/src/components/jobs/WorkspaceJobDescriptionEditor.tsx`: own hidden info and show outdated scoring status.

---

### Task 1: Add Evaluation Persistence Models

**Files:**
- Modify: `backend/src/models/enums.py`
- Create: `backend/src/models/scoring_evaluation.py`
- Modify: `backend/src/models/entities.py`
- Modify: `backend/src/models/__init__.py`
- Create: `backend/migrations/versions/20260707_0015_add_candidate_evaluations.py`
- Test: `backend/tests/test_candidate_evaluations.py`

- [ ] **Step 1: Write model tests first**

Create `backend/tests/test_candidate_evaluations.py` with a SQLite-safe table creation fixture and tests that prove the new models persist JSON payloads and job-scoped preferences.

```python
import uuid
from decimal import Decimal

from src.models.base import Base
from src.models.enums import CandidateEvaluationStatus
from src.models.scoring_evaluation import CandidateEvaluation, JobScoringPreference


def test_candidate_evaluation_persists_raw_scores(db):
    Base.metadata.create_all(bind=db.get_bind())
    evaluation = CandidateEvaluation(
        job_id=uuid.uuid4(),
        job_description_id=uuid.uuid4(),
        candidate_profile_id=uuid.uuid4(),
        scoring_signature="sig-a",
        rubric_payload={"criteria": [{"key": "skills.python"}]},
        raw_component_scores=[
            {
                "criterionKey": "skills.python",
                "section": "skills",
                "evaluationMode": "semantic",
                "scorePercent": 85,
                "evidenceSummary": "Python appears in projects.",
            }
        ],
        rationale_summary="Strong Python match.",
        status=CandidateEvaluationStatus.COMPLETED.value,
    )
    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)

    assert evaluation.status == CandidateEvaluationStatus.COMPLETED.value
    assert evaluation.raw_component_scores[0]["scorePercent"] == 85
    assert evaluation.rationale_summary == "Strong Python match."


def test_job_scoring_preference_persists_weights(db):
    Base.metadata.create_all(bind=db.get_bind())
    preference = JobScoringPreference(
        job_id=uuid.uuid4(),
        section_weights={"skills": 60, "experience": 40},
        score_threshold=Decimal("70.00"),
        updated_by_user_id=uuid.uuid4(),
    )
    db.add(preference)
    db.commit()
    db.refresh(preference)

    assert preference.section_weights == {"skills": 60, "experience": 40}
    assert float(preference.score_threshold) == 70.0
```

- [ ] **Step 2: Run model tests to verify they fail**

Run:

```powershell
cd backend
pytest tests/test_candidate_evaluations.py -q
```

Expected: FAIL because `CandidateEvaluationStatus` and `src.models.scoring_evaluation` do not exist.

- [ ] **Step 3: Add enum and models**

In `backend/src/models/enums.py`, add:

```python
class CandidateEvaluationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    OUTDATED = "outdated"
```

Create `backend/src/models/scoring_evaluation.py`:

```python
from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum as SqlEnum, ForeignKey, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.enums import CandidateEvaluationStatus

_ENUM_VALUES = lambda enum_cls: [item.value for item in enum_cls]

if TYPE_CHECKING:
    from src.models.candidate_profile import CandidateProfile
    from src.models.job import Job
    from src.models.job_matching import JobDescription, MatchRun
    from src.models.user_account import UserAccount


class CandidateEvaluation(Base):
    __tablename__ = "candidate_evaluations"
    __table_args__ = (
        UniqueConstraint(
            "job_description_id",
            "candidate_profile_id",
            "scoring_signature",
            name="uq_candidate_evaluations_jd_candidate_signature",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    job_description_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("job_descriptions.id", ondelete="CASCADE"), nullable=False, index=True)
    candidate_profile_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("candidate_profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    scoring_signature: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    rubric_payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    raw_component_scores: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)
    rationale_summary: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    status: Mapped[CandidateEvaluationStatus] = mapped_column(
        SqlEnum(CandidateEvaluationStatus, name="candidate_evaluation_status_enum", values_callable=_ENUM_VALUES),
        nullable=False,
        default=CandidateEvaluationStatus.PENDING,
        server_default=CandidateEvaluationStatus.PENDING.value,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_match_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("match_runs.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    scored_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    job: Mapped["Job"] = relationship()
    job_description: Mapped["JobDescription"] = relationship()
    candidate_profile: Mapped["CandidateProfile"] = relationship()
    source_match_run: Mapped["MatchRun | None"] = relationship()


class JobScoringPreference(Base):
    __tablename__ = "job_scoring_preferences"

    job_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True)
    section_weights: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    score_threshold: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False, default=Decimal("50.00"), server_default="50.00")
    updated_by_user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("user_accounts.id", ondelete="SET NULL"), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    job: Mapped["Job"] = relationship()
    updated_by_user: Mapped["UserAccount | None"] = relationship()
```

- [ ] **Step 4: Export models**

Update `backend/src/models/entities.py` imports and `__all__`:

```python
from src.models.scoring_evaluation import CandidateEvaluation, JobScoringPreference
```

Add `"CandidateEvaluation"` and `"JobScoringPreference"` to `__all__`.

Update `backend/src/models/__init__.py` imports and `__all__` similarly, and import `CandidateEvaluationStatus` from `src.models.enums`.

- [ ] **Step 5: Add Alembic migration**

Create `backend/migrations/versions/20260707_0015_add_candidate_evaluations.py`:

```python
"""Add candidate evaluations.

Revision ID: 20260707_0015
Revises: 20260622_0014
Create Date: 2026-07-07 19:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260707_0015"
down_revision = "20260622_0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    evaluation_status = postgresql.ENUM(
        "pending",
        "running",
        "completed",
        "failed",
        "outdated",
        name="candidate_evaluation_status_enum",
    )
    evaluation_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "candidate_evaluations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_description_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("candidate_profile_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("scoring_signature", sa.String(length=128), nullable=False),
        sa.Column("rubric_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("raw_component_scores", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("rationale_summary", sa.Text(), server_default="", nullable=False),
        sa.Column("status", evaluation_status, server_default="pending", nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("source_match_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("scored_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["candidate_profile_id"], ["candidate_profiles.id"], name="fk_candidate_evaluations_candidate_profiles", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["job_description_id"], ["job_descriptions.id"], name="fk_candidate_evaluations_job_descriptions", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], name="fk_candidate_evaluations_jobs", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_match_run_id"], ["match_runs.id"], name="fk_candidate_evaluations_match_runs", ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id", name="pk_candidate_evaluations"),
        sa.UniqueConstraint("job_description_id", "candidate_profile_id", "scoring_signature", name="uq_candidate_evaluations_jd_candidate_signature"),
    )
    op.create_index("ix_candidate_evaluations_job_id", "candidate_evaluations", ["job_id"], unique=False)
    op.create_index("ix_candidate_evaluations_job_description_id", "candidate_evaluations", ["job_description_id"], unique=False)
    op.create_index("ix_candidate_evaluations_candidate_profile_id", "candidate_evaluations", ["candidate_profile_id"], unique=False)
    op.create_index("ix_candidate_evaluations_scoring_signature", "candidate_evaluations", ["scoring_signature"], unique=False)

    op.create_table(
        "job_scoring_preferences",
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("section_weights", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("score_threshold", sa.Numeric(5, 2), server_default="50.00", nullable=False),
        sa.Column("updated_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], name="fk_job_scoring_preferences_jobs", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["updated_by_user_id"], ["user_accounts.id"], name="fk_job_scoring_preferences_user_accounts", ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("job_id", name="pk_job_scoring_preferences"),
    )


def downgrade() -> None:
    op.drop_table("job_scoring_preferences")
    op.drop_index("ix_candidate_evaluations_scoring_signature", table_name="candidate_evaluations")
    op.drop_index("ix_candidate_evaluations_candidate_profile_id", table_name="candidate_evaluations")
    op.drop_index("ix_candidate_evaluations_job_description_id", table_name="candidate_evaluations")
    op.drop_index("ix_candidate_evaluations_job_id", table_name="candidate_evaluations")
    op.drop_table("candidate_evaluations")
    postgresql.ENUM(name="candidate_evaluation_status_enum").drop(op.get_bind(), checkfirst=True)
```

- [ ] **Step 6: Run tests and migration smoke check**

Run:

```powershell
cd backend
pytest tests/test_candidate_evaluations.py -q
alembic upgrade head
alembic downgrade 20260622_0014
alembic upgrade head
```

Expected: tests PASS and Alembic upgrade/downgrade succeeds against the configured dev database.

- [ ] **Step 7: Commit**

```powershell
git add backend/src/models/enums.py backend/src/models/scoring_evaluation.py backend/src/models/entities.py backend/src/models/__init__.py backend/migrations/versions/20260707_0015_add_candidate_evaluations.py backend/tests/test_candidate_evaluations.py
git commit -m "feat: add candidate evaluation persistence"
```

---

### Task 2: Add Signature and Weight Calculation Services

**Files:**
- Create: `backend/src/services/scoring_signature.py`
- Create: `backend/src/services/scoring_preferences.py`
- Test: `backend/tests/test_scoring_preferences.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_scoring_preferences.py`:

```python
from decimal import Decimal

from src.services.scoring_preferences import calculate_weighted_score, normalize_section_weights
from src.services.scoring_signature import SCORING_SIGNATURE_VERSION, compute_scoring_signature


def test_scoring_signature_changes_when_hidden_text_changes():
    first = compute_scoring_signature(
        job_description_id="jd-1",
        jd_text="Need Python",
        hidden_text="Prefer RAG",
    )
    second = compute_scoring_signature(
        job_description_id="jd-1",
        jd_text="Need Python",
        hidden_text="Prefer MLOps",
    )
    assert first != second
    assert first.startswith(f"{SCORING_SIGNATURE_VERSION}:")


def test_normalize_section_weights_rejects_zero_total():
    try:
        normalize_section_weights({"skills": 0, "experience": -5})
    except ValueError as exc:
        assert "total" in str(exc).lower()
    else:
        raise AssertionError("Expected zero-total weights to fail")


def test_calculate_weighted_score_uses_raw_percentages_without_llm():
    result = calculate_weighted_score(
        raw_component_scores=[
            {
                "criterionKey": "skills.python",
                "section": "skills",
                "criterionType": "must_have",
                "evaluationMode": "semantic",
                "requirementText": "Python",
                "scorePercent": 80,
                "evidenceSummary": "Python project.",
            },
            {
                "criterionKey": "experience.years",
                "section": "experience",
                "criterionType": "must_have",
                "evaluationMode": "measurable",
                "requirementText": "2+ years",
                "scorePercent": 50,
                "evidenceSummary": "One year listed.",
            },
        ],
        section_weights={"skills": 75, "experience": 25},
        score_threshold=Decimal("70"),
    )

    assert result["totalScore"] == 72.5
    assert result["passedThreshold"] is True
    assert result["componentScores"][0]["scorePercent"] == 80
    assert result["componentScores"][0]["weightedScore"] == 60.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
cd backend
pytest tests/test_scoring_preferences.py -q
```

Expected: FAIL because the service modules do not exist.

- [ ] **Step 3: Implement scoring signature**

Create `backend/src/services/scoring_signature.py`:

```python
from __future__ import annotations

import hashlib
import json
from typing import Any

SCORING_SIGNATURE_VERSION = "scoring-v1"
RUBRIC_PROMPT_VERSION = "rubric-prompt-v1"
SEMANTIC_PROMPT_VERSION = "semantic-prompt-v1"
MEASURABLE_RULE_VERSION = "measurable-rules-v1"


def compute_scoring_signature(
    *,
    job_description_id: Any,
    jd_text: str,
    hidden_text: str,
) -> str:
    payload = {
        "job_description_id": str(job_description_id),
        "jd_text": (jd_text or "").strip(),
        "hidden_text": (hidden_text or "").strip(),
        "rubric_prompt_version": RUBRIC_PROMPT_VERSION,
        "semantic_prompt_version": SEMANTIC_PROMPT_VERSION,
        "measurable_rule_version": MEASURABLE_RULE_VERSION,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{SCORING_SIGNATURE_VERSION}:{digest}"
```

- [ ] **Step 4: Implement weighting service**

Create `backend/src/services/scoring_preferences.py`:

```python
from __future__ import annotations

from decimal import Decimal
from typing import Any

DEFAULT_SECTION_WEIGHTS = {
    "skills": 25.0,
    "experience": 25.0,
    "education": 20.0,
    "projects": 20.0,
    "summary": 10.0,
}


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(100.0, score))


def normalize_section_weights(section_weights: dict[str, float] | None) -> dict[str, float]:
    source = section_weights if section_weights is not None else DEFAULT_SECTION_WEIGHTS
    cleaned = {
        str(key): max(0.0, float(value))
        for key, value in source.items()
        if value is not None
    }
    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError("section_weights total must be > 0")
    return {key: value / total for key, value in cleaned.items()}


def _component_section(component: dict[str, Any]) -> str:
    section = str(component.get("section") or "").strip()
    if section:
        return section
    criterion_key = str(component.get("criterionKey") or "")
    return criterion_key.split(".", 1)[0] if "." in criterion_key else criterion_key


def calculate_weighted_score(
    *,
    raw_component_scores: list[dict[str, Any]],
    section_weights: dict[str, float] | None,
    score_threshold: Decimal | float,
) -> dict[str, Any]:
    normalized_weights = normalize_section_weights(section_weights)
    component_scores: list[dict[str, Any]] = []

    for component in raw_component_scores:
        section = _component_section(component)
        effective_weight = normalized_weights.get(section, 0.0)
        score_percent = _clamp_score(component.get("scorePercent", component.get("score", 0)))
        weighted_score = round(score_percent * effective_weight, 2)
        component_scores.append(
            {
                **component,
                "section": section,
                "scorePercent": score_percent,
                "score": score_percent,
                "effectiveWeight": round(effective_weight, 4),
                "weight": round(effective_weight, 4),
                "weightedScore": weighted_score,
            }
        )

    total_score = round(sum(component["weightedScore"] for component in component_scores), 2)
    total_score = _clamp_score(total_score)
    return {
        "totalScore": total_score,
        "passedThreshold": total_score >= float(score_threshold),
        "componentScores": component_scores,
    }
```

- [ ] **Step 5: Run tests**

Run:

```powershell
cd backend
pytest tests/test_scoring_preferences.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add backend/src/services/scoring_signature.py backend/src/services/scoring_preferences.py backend/tests/test_scoring_preferences.py
git commit -m "feat: add scoring signature and weight calculation"
```

---

### Task 3: Refactor Raw Evaluation and Prompt Contract

**Files:**
- Modify: `backend/src/prompts/build_prompts.py`
- Modify: `backend/src/services/score_candidate.py`
- Modify: `backend/tests/test_build_prompts.py`
- Modify: `backend/tests/test_score_candidate_service.py`

- [ ] **Step 1: Update prompt tests first**

In `backend/tests/test_build_prompts.py`, update or add:

```python
from src.prompts.build_prompts import BuildPrompts


def test_locked_rubric_semantic_prompt_scores_every_criterion_without_total_score():
    prompt = BuildPrompts().build_locked_rubric_semantic_scoring_prompt(
        candidates=[{"id": "candidate-1", "skills_text": "Python"}],
        rubric={
            "criteria": [
                {
                    "key": "skills.python",
                    "section": "skills",
                    "type": "must_have",
                    "requirementText": "Python",
                }
            ]
        },
    )

    assert "Score every listed semantic criterion" in prompt
    assert "Does not calculate totalScore" not in prompt
    assert "Do not calculate totalScore" in prompt
    assert '"scorePercent"' in prompt
    assert '"totalScore"' not in prompt
```

- [ ] **Step 2: Add raw evaluation tests**

In `backend/tests/test_score_candidate_service.py`, add a test for raw component output:

```python
from decimal import Decimal

from src.services.score_candidate import _build_candidate_raw_evaluation


def test_build_candidate_raw_evaluation_keeps_measurable_rule_based():
    candidate = {
        "id": "candidate-1",
        "full_name": "An",
        "experience_years": 3,
    }
    rubric = {
        "criteria": [
            {
                "key": "experience.years",
                "section": "experience",
                "type": "must_have",
                "requirementText": "2+ years",
                "measurable": {"field": "experience_years", "operator": ">=", "value": 2},
            },
            {
                "key": "skills.python",
                "section": "skills",
                "type": "semantic",
                "requirementText": "Python",
            },
        ]
    }
    semantic_result = {
        "criteria": {
            "skills.python": {
                "score": 75,
                "evidenceSummary": "Python project listed.",
            }
        }
    }

    raw = _build_candidate_raw_evaluation(
        candidate=candidate,
        rubric=rubric,
        semantic_result=semantic_result,
    )

    assert raw["rawComponentScores"][0]["evaluationMode"] == "measurable"
    assert raw["rawComponentScores"][0]["scorePercent"] == 100.0
    assert raw["rawComponentScores"][1]["evaluationMode"] == "semantic"
    assert raw["rawComponentScores"][1]["scorePercent"] == 75.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```powershell
cd backend
pytest tests/test_build_prompts.py::test_locked_rubric_semantic_prompt_scores_every_criterion_without_total_score tests/test_score_candidate_service.py::test_build_candidate_raw_evaluation_keeps_measurable_rule_based -q
```

Expected: FAIL because prompt still uses `score` and `_build_candidate_raw_evaluation` does not exist.

- [ ] **Step 4: Update prompt response format**

In `BuildPrompts.build_locked_rubric_semantic_scoring_prompt`, change response format criteria rows to:

```python
{
    "criterionKey": "skills.python",
    "scorePercent": 85,
    "evidenceSummary": "string",
}
```

Update instruction string to include:

```python
"Score every listed semantic criterion. "
"Do not calculate totalScore, weightedScore, passedThreshold, or section totals. "
"Return scorePercent as a number from 0 to 100, where 100 is a clear full match, 70 is a strong partial match, "
"40 is weak or indirect evidence, and 0 is no evidence. "
```

- [ ] **Step 5: Update semantic parser**

In `_parse_semantic_scores`, accept `scorePercent` before fallback `score`:

```python
"score": _normalize_llm_score(row.get("scorePercent", row.get("score", 0))),
"scorePercent": _normalize_llm_score(row.get("scorePercent", row.get("score", 0))),
```

- [ ] **Step 6: Add raw evaluation builder**

Add this function near `_build_candidate_score` in `backend/src/services/score_candidate.py`:

```python
def _build_candidate_raw_evaluation(
    *,
    candidate: Dict[str, Any],
    rubric: Dict[str, Any],
    semantic_result: Dict[str, Any],
    debug_logger: Optional[ScoringDebugLogger] = None,
) -> Dict[str, Any]:
    semantic_criteria = semantic_result.get("criteria", {}) if isinstance(semantic_result, dict) else {}
    raw_component_scores: List[Dict[str, Any]] = []

    for criterion in rubric.get("criteria", []):
        if criterion.get("measurable"):
            score, evidence, measurable_detail = _score_measurable_criterion(candidate, criterion)
            evaluation_mode = "measurable"
        else:
            semantic_detail = semantic_criteria.get(criterion["key"], {})
            score = _normalize_llm_score(
                semantic_detail.get("scorePercent", semantic_detail.get("score", 0))
            )
            evidence = str(semantic_detail.get("evidenceSummary") or "").strip()
            evaluation_mode = "semantic"
            measurable_detail = None

        component = {
            "criterionKey": criterion["key"],
            "criterionType": criterion["type"],
            "section": str(criterion.get("section") or "").strip(),
            "evaluationMode": evaluation_mode,
            "requirementText": criterion["requirementText"],
            "scorePercent": score,
            "evidenceSummary": evidence,
        }
        if measurable_detail is not None:
            component["measurable"] = measurable_detail
        raw_component_scores.append(component)

    rationale = _build_rationale_summary(
        round(sum(float(component.get("scorePercent") or 0) for component in raw_component_scores) / max(1, len(raw_component_scores)), 2),
        [
            {
                **component,
                "score": component.get("scorePercent", 0),
                "weightedScore": component.get("scorePercent", 0),
            }
            for component in raw_component_scores
        ],
    )

    return {
        "candidateId": str(candidate.get("id") or candidate.get("candidateId") or ""),
        "candidateName": str(candidate.get("full_name") or "").strip(),
        "resumeFileName": str(candidate.get("resume_file_name") or "").strip(),
        "candidateDisplayName": str(
            candidate.get("display_name") or candidate.get("full_name") or candidate.get("resume_file_name") or ""
        ).strip(),
        "rationale": rationale,
        "rawComponentScores": raw_component_scores,
    }
```

- [ ] **Step 7: Keep legacy `_build_candidate_score` compatible**

Do not import `calculate_weighted_score` into `score_candidate.py`; that creates unnecessary coupling between the legacy match-run endpoint and the new evaluation display calculation. Keep `_build_candidate_score` returning the same response shape for existing `/score` and `/jobs/{job_id}/score` endpoints, but update its component dictionaries to include `section` and `scorePercent` alongside existing `score`.

```python
"section": str(criterion.get("section") or "").strip(),
"scorePercent": score,
```

The new evaluation service in Task 4 will use `_build_candidate_raw_evaluation` plus `calculate_weighted_score`. The legacy scoring endpoint continues to use rubric weights exactly as it does today until it is removed in a separate migration.

- [ ] **Step 8: Run focused tests**

Run:

```powershell
cd backend
pytest tests/test_build_prompts.py tests/test_score_candidate_service.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```powershell
git add backend/src/prompts/build_prompts.py backend/src/services/score_candidate.py backend/tests/test_build_prompts.py backend/tests/test_score_candidate_service.py
git commit -m "feat: store raw criterion scoring percentages"
```

---

### Task 4: Implement Candidate Evaluation Service

**Files:**
- Create: `backend/src/services/candidate_evaluation_service.py`
- Modify: `backend/tests/test_candidate_evaluations.py`

- [ ] **Step 1: Add service tests**

Append to `backend/tests/test_candidate_evaluations.py`:

```python
from src.models.enums import CandidateEvaluationStatus
from src.services.candidate_evaluation_service import mark_job_evaluations_outdated


def test_mark_job_evaluations_outdated_only_changes_old_signature(db):
    Base.metadata.create_all(bind=db.get_bind())
    job_id = uuid.uuid4()
    jd_id = uuid.uuid4()
    old = CandidateEvaluation(
        job_id=job_id,
        job_description_id=jd_id,
        candidate_profile_id=uuid.uuid4(),
        scoring_signature="old",
        rubric_payload={},
        raw_component_scores=[],
        status=CandidateEvaluationStatus.COMPLETED.value,
    )
    current = CandidateEvaluation(
        job_id=job_id,
        job_description_id=jd_id,
        candidate_profile_id=uuid.uuid4(),
        scoring_signature="current",
        rubric_payload={},
        raw_component_scores=[],
        status=CandidateEvaluationStatus.COMPLETED.value,
    )
    db.add_all([old, current])
    db.commit()

    changed = mark_job_evaluations_outdated(
        db=db,
        job_id=job_id,
        current_scoring_signature="current",
    )
    db.refresh(old)
    db.refresh(current)

    assert changed == 1
    assert old.status == CandidateEvaluationStatus.OUTDATED.value
    assert current.status == CandidateEvaluationStatus.COMPLETED.value
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
cd backend
pytest tests/test_candidate_evaluations.py::test_mark_job_evaluations_outdated_only_changes_old_signature -q
```

Expected: FAIL because the service module does not exist.

- [ ] **Step 3: Implement lifecycle helpers**

Create `backend/src/services/candidate_evaluation_service.py`:

```python
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy.orm import Session, joinedload

from src.models.candidate_profile import CandidateProfile
from src.models.enums import CandidateEvaluationStatus
from src.models.job_matching import JobDescription
from src.models.resume_document import ResumeDocument
from src.models.scoring_evaluation import CandidateEvaluation, JobScoringPreference
from src.services.score_candidate import (
    _build_candidate_raw_evaluation,
    _build_scoring_job_description_text,
    _extract_locked_rubric,
    _generate_semantic_scores_with_retries,
    _merge_semantic_scores,
    _profile_to_candidate_dict,
    _scoring_llm_provider,
    build_prompts,
)
from src.services.scoring_preferences import calculate_weighted_score
from src.services.scoring_signature import compute_scoring_signature


def get_job_scoring_preference(db: Session, job_id: uuid.UUID) -> JobScoringPreference | None:
    return db.get(JobScoringPreference, job_id)


def current_signature_for_jd(jd: JobDescription) -> str:
    return compute_scoring_signature(
        job_description_id=jd.id,
        jd_text=jd.jd_text,
        hidden_text=getattr(jd, "hidden_text", "") or "",
    )


def mark_job_evaluations_outdated(
    *,
    db: Session,
    job_id: uuid.UUID,
    current_scoring_signature: str,
) -> int:
    rows = (
        db.query(CandidateEvaluation)
        .filter(
            CandidateEvaluation.job_id == job_id,
            CandidateEvaluation.scoring_signature != current_scoring_signature,
            CandidateEvaluation.status == CandidateEvaluationStatus.COMPLETED.value,
        )
        .all()
    )
    for row in rows:
        row.status = CandidateEvaluationStatus.OUTDATED.value
    return len(rows)


def serialize_evaluation(
    *,
    evaluation: CandidateEvaluation,
    preference: JobScoringPreference | None,
) -> dict[str, Any]:
    threshold = preference.score_threshold if preference is not None else Decimal("50.00")
    weights = preference.section_weights if preference is not None else None
    weighted = calculate_weighted_score(
        raw_component_scores=evaluation.raw_component_scores or [],
        section_weights=weights,
        score_threshold=threshold,
    )
    return {
        "id": str(evaluation.id),
        "job_id": str(evaluation.job_id),
        "job_description_id": str(evaluation.job_description_id),
        "candidate_profile_id": str(evaluation.candidate_profile_id),
        "scoring_signature": evaluation.scoring_signature,
        "status": str(evaluation.status),
        "rationale": evaluation.rationale_summary,
        "error_message": evaluation.error_message,
        "scored_at": evaluation.scored_at,
        **weighted,
    }
```

Then add `evaluate_candidate_for_current_jd` in the same file. Use the existing scoring helpers to avoid a second prompt stack:

```python
def evaluate_candidate_for_current_jd(
    *,
    db: Session,
    candidate_profile_id: uuid.UUID,
) -> dict[str, Any]:
    profile = (
        db.query(CandidateProfile)
        .options(joinedload(CandidateProfile.resume_document))
        .filter(CandidateProfile.id == candidate_profile_id)
        .first()
    )
    if profile is None or profile.resume_document is None:
        raise ValueError(f"CandidateProfile {candidate_profile_id} not found")

    jd = (
        db.query(JobDescription)
        .filter(
            JobDescription.job_id == profile.resume_document.job_id,
            JobDescription.is_active.is_(True),
        )
        .order_by(JobDescription.created_at.desc())
        .first()
    )
    if jd is None:
        return {"status": "skipped", "reason": "No active job description"}

    signature = current_signature_for_jd(jd)
    existing = (
        db.query(CandidateEvaluation)
        .filter(
            CandidateEvaluation.job_description_id == jd.id,
            CandidateEvaluation.candidate_profile_id == profile.id,
            CandidateEvaluation.scoring_signature == signature,
            CandidateEvaluation.status == CandidateEvaluationStatus.COMPLETED.value,
        )
        .first()
    )
    if existing is not None:
        return {"status": "skipped", "reason": "Evaluation already completed", "evaluation_id": str(existing.id)}

    evaluation = CandidateEvaluation(
        job_id=profile.resume_document.job_id,
        job_description_id=jd.id,
        candidate_profile_id=profile.id,
        scoring_signature=signature,
        rubric_payload={},
        raw_component_scores=[],
        status=CandidateEvaluationStatus.RUNNING.value,
    )
    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)

    try:
        llm = _scoring_llm_provider()
        scoring_jd_text = _build_scoring_job_description_text(
            public_job_description=jd.jd_text,
            hidden_text=getattr(jd, "hidden_text", ""),
        )
        rubric = _extract_locked_rubric(
            llm=llm,
            job_description_text=scoring_jd_text,
            section_weights=None,
            debug_logger=None,
        )
        if rubric is None:
            raise ValueError("Locked rubric extraction failed")

        candidate = _profile_to_candidate_dict(profile)
        semantic_criteria = [criterion for criterion in rubric["criteria"] if criterion.get("measurable") is None]
        semantic_by_candidate: dict[str, dict[str, Any]] = {}
        if semantic_criteria:
            semantic_update = _generate_semantic_scores_with_retries(
                llm=llm,
                prompt=build_prompts.build_locked_rubric_semantic_scoring_prompt(
                    candidates=[candidate],
                    rubric={"criteria": semantic_criteria},
                ),
                debug_logger=None,
            )
            _merge_semantic_scores(semantic_by_candidate, semantic_update)

        raw = _build_candidate_raw_evaluation(
            candidate=candidate,
            rubric=rubric,
            semantic_result=semantic_by_candidate.get(str(candidate["id"]), {}),
        )
        evaluation.rubric_payload = rubric
        evaluation.raw_component_scores = raw["rawComponentScores"]
        evaluation.rationale_summary = raw["rationale"]
        evaluation.status = CandidateEvaluationStatus.COMPLETED.value
        evaluation.error_message = None
        evaluation.scored_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(evaluation)
        return {"status": "completed", "evaluation_id": str(evaluation.id)}
    except Exception as exc:
        evaluation.status = CandidateEvaluationStatus.FAILED.value
        evaluation.error_message = str(exc)
        db.commit()
        raise
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
cd backend
pytest tests/test_candidate_evaluations.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add backend/src/services/candidate_evaluation_service.py backend/tests/test_candidate_evaluations.py
git commit -m "feat: add candidate evaluation service"
```

---

### Task 5: Add Worker Task and Auto Trigger After Resume Parse

**Files:**
- Modify: `backend/worker/tasks.py`
- Modify: `backend/tests/conftest.py`
- Add or modify: `backend/tests/test_resume_scoring_trigger.py`

- [ ] **Step 1: Write worker trigger test**

Create `backend/tests/test_resume_scoring_trigger.py`:

```python
import types

import worker.tasks as tasks


def test_process_resume_enqueues_evaluation_after_success(monkeypatch):
    calls = []

    def fake_process_single_resume(*args, **kwargs):
        return {
            "status": "processed",
            "candidate_profile_id": "11111111-1111-1111-1111-111111111111",
            "extraction_mode": "text",
        }

    monkeypatch.setitem(
        __import__("sys").modules,
        "src.services.resume_service",
        types.SimpleNamespace(process_single_resume=fake_process_single_resume),
    )
    monkeypatch.setattr(
        tasks.evaluate_candidate,
        "delay",
        lambda candidate_profile_id: calls.append(candidate_profile_id) or types.SimpleNamespace(id="eval-task-id"),
    )

    result = tasks.process_resume.run("22222222-2222-2222-2222-222222222222")

    assert result["status"] == "processed"
    assert calls == ["11111111-1111-1111-1111-111111111111"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cd backend
pytest tests/test_resume_scoring_trigger.py -q
```

Expected: FAIL because `evaluate_candidate` task is not defined.

- [ ] **Step 3: Add worker task**

In `backend/worker/tasks.py`, add after `process_resume`:

```python
@celery_app.task(
    name="worker.tasks.evaluate_candidate",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
)
def evaluate_candidate(self, candidate_profile_id: str):
    logger.info("evaluate_candidate started for %s", candidate_profile_id)
    try:
        from src.models.deps import SessionLocal
        from src.services.candidate_evaluation_service import evaluate_candidate_for_current_jd

        with SessionLocal() as db:
            result = evaluate_candidate_for_current_jd(
                db=db,
                candidate_profile_id=uuid.UUID(candidate_profile_id),
            )
        logger.info("evaluate_candidate finished for %s with %s", candidate_profile_id, result.get("status"))
        return result
    except Exception as exc:
        logger.exception("evaluate_candidate crashed for %s", candidate_profile_id)
        raise self.retry(exc=exc)
```

In `process_resume`, after a successful result:

```python
candidate_profile_id = result.get("candidate_profile_id")
if result.get("status") != "failed" and candidate_profile_id:
    evaluate_candidate.delay(str(candidate_profile_id))
```

- [ ] **Step 4: Update test stubs**

In `backend/tests/conftest.py`, add:

```python
tasks_stub.evaluate_candidate = types.SimpleNamespace(
    delay=lambda *args, **kwargs: types.SimpleNamespace(id="test-evaluation-task-id")
)
```

- [ ] **Step 5: Ensure resume service returns candidate_profile_id**

Inspect `process_single_resume` return payload in `backend/src/services/resume_service.py`. If it does not include `candidate_profile_id`, update the success return to include:

```python
"candidate_profile_id": str(profile.id),
```

- [ ] **Step 6: Run focused tests**

Run:

```powershell
cd backend
pytest tests/test_resume_scoring_trigger.py tests/test_public_job_endpoints.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add backend/worker/tasks.py backend/tests/conftest.py backend/tests/test_resume_scoring_trigger.py backend/src/services/resume_service.py
git commit -m "feat: enqueue candidate evaluation after resume parsing"
```

---

### Task 6: Add Job Evaluation API Endpoints

**Files:**
- Modify: `backend/src/api/v1/endpoints/jobs.py`
- Modify: `backend/src/services/candidate_evaluation_service.py`
- Test: `backend/tests/test_jobs_evaluation_endpoints.py`

- [ ] **Step 1: Write endpoint tests**

Create `backend/tests/test_jobs_evaluation_endpoints.py` using existing endpoint test patterns from `backend/tests/test_jobs_score_endpoint.py`. Include these tests:

```python
def test_patch_job_description_marks_evaluations_outdated(monkeypatch, db, owner, owned_job_with_jd_and_evaluation):
    response = client.patch(
        f"/api/v1/jobs/{owned_job_with_jd_and_evaluation.job_id}/job-description",
        headers=auth_headers(owner),
        json={"hidden_text": "new hidden criteria"},
    )
    assert response.status_code == 200
    evaluation = db.get(CandidateEvaluation, owned_job_with_jd_and_evaluation.evaluation_id)
    assert evaluation.status == CandidateEvaluationStatus.OUTDATED.value


def test_get_job_evaluations_returns_weighted_scores(client, owner_headers, job_with_completed_evaluation):
    response = client.get(f"/api/v1/jobs/{job_with_completed_evaluation.job_id}/evaluations", headers=owner_headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload["completed_count"] == 1
    assert payload["items"][0]["totalScore"] >= 0


def test_put_scoring_preferences_recalculates_without_llm(client, owner_headers, job_with_completed_evaluation):
    response = client.put(
        f"/api/v1/jobs/{job_with_completed_evaluation.job_id}/scoring-preferences",
        headers=owner_headers,
        json={"section_weights": {"skills": 100}, "score_threshold": 60},
    )
    assert response.status_code == 200
    assert response.json()["section_weights"] == {"skills": 100}
```

Use concrete fixtures modeled after `owned_score_run` in `backend/tests/test_jobs_score_endpoint.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
cd backend
pytest tests/test_jobs_evaluation_endpoints.py -q
```

Expected: FAIL because endpoints do not exist.

- [ ] **Step 3: Add Pydantic schemas in `jobs.py`**

Add these classes near scoring response schemas:

```python
class JobScoringPreferenceRequest(BaseModel):
    section_weights: dict[str, float]
    score_threshold: float = Field(50.0, ge=0, le=100)


class JobScoringPreferenceResponse(BaseModel):
    job_id: str
    section_weights: dict[str, float]
    score_threshold: float
    updated_at: datetime


class CandidateEvaluationResponse(BaseModel):
    id: str
    job_id: str
    job_description_id: str
    candidate_profile_id: str
    scoring_signature: str
    status: str
    totalScore: float
    passedThreshold: bool
    rationale: str
    error_message: Optional[str] = None
    scored_at: Optional[datetime] = None
    componentScores: list[ComponentScoreResponse]


class JobEvaluationListResponse(BaseModel):
    job_id: str
    total_candidates: int
    completed_count: int
    pending_count: int
    running_count: int
    failed_count: int
    outdated_count: int
    average_score: float
    highest_score: float
    items: list[CandidateEvaluationResponse]
```

- [ ] **Step 4: Mark outdated on JD patch/upsert**

After JD commit in `patch_job_description` and the active-JD upsert path, call:

```python
from src.services.candidate_evaluation_service import current_signature_for_jd, mark_job_evaluations_outdated

signature = current_signature_for_jd(jd)
mark_job_evaluations_outdated(db=db, job_id=job_id, current_scoring_signature=signature)
db.commit()
```

Only do this when `jd_text` or `hidden_text` changed.

- [ ] **Step 5: Add endpoints**

Add:

```python
@router.get("/{job_id}/evaluations", response_model=JobEvaluationListResponse)
def list_job_evaluations(...):
    ...

@router.get("/{job_id}/candidates/{candidate_profile_id}/evaluation", response_model=CandidateEvaluationResponse)
def get_candidate_evaluation(...):
    ...

@router.put("/{job_id}/scoring-preferences", response_model=JobScoringPreferenceResponse)
def update_job_scoring_preferences(...):
    ...

@router.post("/{job_id}/evaluations/score-again")
def score_again_job_evaluations(...):
    ...
```

Implementation should call service helpers instead of embedding scoring logic in endpoints.

- [ ] **Step 6: Add service helpers for listing and queueing**

In `candidate_evaluation_service.py`, add:

```python
def list_latest_job_evaluations(db: Session, job_id: uuid.UUID) -> list[CandidateEvaluation]:
    return (
        db.query(CandidateEvaluation)
        .filter(CandidateEvaluation.job_id == job_id)
        .order_by(CandidateEvaluation.updated_at.desc())
        .all()
    )
```

And a queue helper:

```python
def enqueue_missing_current_evaluations(*, db: Session, job_id: uuid.UUID, jd: JobDescription) -> dict[str, Any]:
    from worker.tasks import evaluate_candidate

    signature = current_signature_for_jd(jd)
    profiles = (
        db.query(CandidateProfile)
        .join(ResumeDocument, ResumeDocument.id == CandidateProfile.resume_document_id)
        .filter(ResumeDocument.job_id == job_id)
        .all()
    )
    queued = 0
    for profile in profiles:
        exists = (
            db.query(CandidateEvaluation)
            .filter(
                CandidateEvaluation.job_description_id == jd.id,
                CandidateEvaluation.candidate_profile_id == profile.id,
                CandidateEvaluation.scoring_signature == signature,
                CandidateEvaluation.status == CandidateEvaluationStatus.COMPLETED.value,
            )
            .first()
        )
        if exists is None:
            evaluate_candidate.delay(str(profile.id))
            queued += 1
    return {"queued": queued, "total_candidates": len(profiles)}
```

- [ ] **Step 7: Run endpoint tests**

Run:

```powershell
cd backend
pytest tests/test_jobs_evaluation_endpoints.py tests/test_jobs_score_endpoint.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add backend/src/api/v1/endpoints/jobs.py backend/src/services/candidate_evaluation_service.py backend/tests/test_jobs_evaluation_endpoints.py
git commit -m "feat: expose candidate evaluation endpoints"
```

---

### Task 7: Update Frontend Types and API Client

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/endpoints/jobs.ts`

- [ ] **Step 1: Add TypeScript types**

In `frontend/src/api/types.ts`, add:

```ts
export interface CandidateEvaluationComponentScore extends ComponentScore {
  section?: string | null;
  scorePercent: number;
  effectiveWeight: number;
}

export interface CandidateEvaluationResponse {
  id: string;
  job_id: string;
  job_description_id: string;
  candidate_profile_id: string;
  scoring_signature: string;
  status: "pending" | "running" | "completed" | "failed" | "outdated";
  totalScore: number;
  passedThreshold: boolean;
  rationale: string;
  error_message?: string | null;
  scored_at?: string | null;
  componentScores: CandidateEvaluationComponentScore[];
}

export interface JobEvaluationListResponse {
  job_id: string;
  total_candidates: number;
  completed_count: number;
  pending_count: number;
  running_count: number;
  failed_count: number;
  outdated_count: number;
  average_score: number;
  highest_score: number;
  items: CandidateEvaluationResponse[];
}

export interface JobScoringPreferenceResponse {
  job_id: string;
  section_weights: Record<string, number>;
  score_threshold: number;
  updated_at: string;
}
```

- [ ] **Step 2: Add API methods**

In `frontend/src/api/endpoints/jobs.ts`, import the new types and add:

```ts
  evaluations: {
    async list(jobId: string): Promise<JobEvaluationListResponse> {
      const { data } = await client.get<JobEvaluationListResponse>(`/jobs/${jobId}/evaluations`);
      return data;
    },
    async getCandidate(jobId: string, candidateProfileId: string): Promise<CandidateEvaluationResponse> {
      const { data } = await client.get<CandidateEvaluationResponse>(`/jobs/${jobId}/candidates/${candidateProfileId}/evaluation`);
      return data;
    },
    async scoreAgain(jobId: string): Promise<{ queued: number; total_candidates: number }> {
      const { data } = await client.post<{ queued: number; total_candidates: number }>(`/jobs/${jobId}/evaluations/score-again`);
      return data;
    },
  },

  scoringPreferences: {
    async update(jobId: string, body: { section_weights: Record<string, number>; score_threshold: number }): Promise<JobScoringPreferenceResponse> {
      const { data } = await client.put<JobScoringPreferenceResponse>(`/jobs/${jobId}/scoring-preferences`, body);
      return data;
    },
  },
```

- [ ] **Step 3: Run typecheck**

Run:

```powershell
cd frontend
npm run typecheck
```

Expected: PASS.

- [ ] **Step 4: Commit**

```powershell
git add frontend/src/api/types.ts frontend/src/api/endpoints/jobs.ts
git commit -m "feat: add evaluation api client types"
```

---

### Task 8: Rewrite Scoring Page as Results Dashboard

**Files:**
- Modify: `frontend/src/routes/scoring/setup.tsx`

- [ ] **Step 1: Replace wizard data flow**

Remove `step`, candidate selection state, hidden text state, processing timer state, and `startScoring`. Fetch:

```ts
const { data: evaluations, isLoading } = useQuery({
  queryKey: ["jobs", selectedJobId, "evaluations"],
  queryFn: () => selectedJobId ? api.jobs.evaluations.list(selectedJobId) : Promise.resolve(null),
  enabled: !!selectedJobId,
  refetchInterval: (query) => {
    const data = query.state.data;
    return data && (data.pending_count > 0 || data.running_count > 0) ? 3000 : false;
  },
});
```

- [ ] **Step 2: Add job-level weight mutation**

Use `useMutation`:

```ts
const savePreferences = useMutation({
  mutationFn: (body: { section_weights: Record<string, number>; score_threshold: number }) =>
    api.jobs.scoringPreferences.update(selectedJobId!, body),
  onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: ["jobs", selectedJobId, "evaluations"] });
  },
});
```

- [ ] **Step 3: Add Score again mutation**

```ts
const scoreAgain = useMutation({
  mutationFn: () => api.jobs.evaluations.scoreAgain(selectedJobId!),
  onSuccess: () => {
    toast.success("Scoring queued");
    queryClient.invalidateQueries({ queryKey: ["jobs", selectedJobId, "evaluations"] });
    queryClient.invalidateQueries({ queryKey: ["jobs", selectedJobId, "setup-status"] });
  },
});
```

- [ ] **Step 4: Render dashboard states**

Keep summary cards and table, but source from `evaluations.items`.

Required visible states:

- No selected job.
- No evaluations yet with `Score again`.
- Loading skeleton.
- Outdated count greater than zero with `Score again`.
- Candidate status badge for `pending`, `running`, `completed`, `failed`, `outdated`.

- [ ] **Step 5: Run frontend verification**

Run:

```powershell
cd frontend
npm run typecheck
npm run lint
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add frontend/src/routes/scoring/setup.tsx
git commit -m "feat: show scoring evaluations dashboard"
```

---

### Task 9: Add Candidate Detail Evaluation View

**Files:**
- Modify: `frontend/src/routes/candidates/detail.tsx`

- [ ] **Step 1: Add tab**

Extend:

```ts
type Tab = "overview" | "resume" | "evaluation" | "outreach" | "interview";
```

Add `Evaluation` tab after `Resume PDF`.

- [ ] **Step 2: Fetch candidate evaluation**

Use profile ID, not resume ID:

```ts
const { data: evaluation, isLoading: evaluationLoading } = useQuery({
  queryKey: ["jobs", selectedJobId, "candidate-evaluation", profile?.id],
  queryFn: () => api.jobs.evaluations.getCandidate(selectedJobId!, profile!.id),
  enabled: !!selectedJobId && !!profile?.id,
  refetchInterval: (query) => {
    const status = query.state.data?.status;
    return status === "pending" || status === "running" ? 3000 : false;
  },
});
```

- [ ] **Step 3: Render evaluation tab**

Add `EvaluationTab` component in the same file:

```tsx
function EvaluationTab({ evaluation, loading }: { evaluation?: CandidateEvaluationResponse; loading: boolean }) {
  if (loading) return <Skeleton className="h-64 w-full rounded-[var(--radius-lg)]" />;
  if (!evaluation) {
    return <EmptyState heading="Evaluation unavailable" body="Scoring will appear after this candidate is evaluated against the current job description." />;
  }
  return (
    <div className="space-y-5">
      <section className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-5">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h2 className="font-display text-xl font-medium text-fg">JD Match</h2>
            <p className="mt-2 text-sm text-fg-muted">{evaluation.rationale}</p>
          </div>
          <div className="text-right">
            <p className="font-display text-4xl font-medium text-fg tabular-nums">{evaluation.totalScore}</p>
            <Badge variant={evaluation.status === "completed" ? "success" : evaluation.status === "failed" ? "danger" : "warning"}>
              {evaluation.status}
            </Badge>
          </div>
        </div>
      </section>
      {evaluation.componentScores.map((component) => (
        <section key={component.criterionKey} className="rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg-elevated p-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="text-sm font-medium text-fg">{component.requirementText || component.criterionKey}</h3>
              <p className="mt-2 text-sm leading-relaxed text-fg-muted">{component.evidenceSummary}</p>
            </div>
            <div className="text-right">
              <p className="font-mono text-lg text-fg tabular-nums">{component.scorePercent}%</p>
              <Badge variant={component.evaluationMode === "measurable" ? "success" : "neutral"} size="sm">
                {component.evaluationMode === "measurable" ? "Rule-based" : "Semantic"}
              </Badge>
            </div>
          </div>
        </section>
      ))}
    </div>
  );
}
```

- [ ] **Step 4: Run frontend verification**

Run:

```powershell
cd frontend
npm run typecheck
npm run lint
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add frontend/src/routes/candidates/detail.tsx
git commit -m "feat: show candidate evaluation detail"
```

---

### Task 10: Move Hidden Information Ownership to JD UI

**Files:**
- Modify: `frontend/src/components/jobs/WorkspaceJobDescriptionEditor.tsx`
- Modify: `frontend/src/routes/scoring/setup.tsx`
- Modify: `backend/src/api/v1/endpoints/jobs.py`

- [ ] **Step 1: Add JD editor hidden text field**

In `WorkspaceJobDescriptionEditor.tsx`, add state for `hiddenText`, load it from `api.jobs.jobDescription.get`, and include it in upsert/patch:

```ts
const [hiddenText, setHiddenText] = useState("");
```

Render a textarea below public JD:

```tsx
<textarea
  aria-label="Recruiter-only hidden information"
  value={hiddenText}
  onChange={(event) => setHiddenText(event.target.value)}
  className="w-full min-h-28 rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] bg-bg px-3 py-2 text-sm text-fg"
/>
```

- [ ] **Step 2: Show scoring status and Score again on JD page**

Fetch evaluations in JD editor or parent:

```ts
const { data: evaluations } = useQuery({
  queryKey: ["jobs", selectedJobId, "evaluations"],
  queryFn: () => api.jobs.evaluations.list(selectedJobId!),
  enabled: !!selectedJobId,
});
```

Status mapping:

```ts
const scoringStatus =
  !evaluations || evaluations.total_candidates === 0
    ? "Not scored"
    : evaluations.outdated_count > 0
      ? "Outdated"
      : evaluations.running_count > 0 || evaluations.pending_count > 0
        ? "Scoring"
        : "Current";
```

Show `Score again` when status is `Outdated` or `Not scored`.

- [ ] **Step 3: Ensure scoring page has no hidden editor**

Confirm `frontend/src/routes/scoring/setup.tsx` no longer calls:

```ts
api.jobs.jobDescription.patch(selectedJobId, { hidden_text: hiddenText })
```

- [ ] **Step 4: Run verification**

Run:

```powershell
cd frontend
npm run typecheck
npm run lint
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add frontend/src/components/jobs/WorkspaceJobDescriptionEditor.tsx frontend/src/routes/scoring/setup.tsx
git commit -m "feat: move hidden scoring input to job description"
```

---

### Task 11: Final Integration Verification

**Files:**
- Review: all files changed by Tasks 1-10

- [ ] **Step 1: Run backend focused tests**

Run:

```powershell
cd backend
pytest tests/test_candidate_evaluations.py tests/test_scoring_preferences.py tests/test_score_candidate_service.py tests/test_build_prompts.py tests/test_jobs_evaluation_endpoints.py tests/test_resume_scoring_trigger.py -q
```

Expected: PASS.

- [ ] **Step 2: Run existing scoring endpoint regression tests**

Run:

```powershell
cd backend
pytest tests/test_score_endpoint.py tests/test_jobs_score_endpoint.py tests/test_score_candidate_error_handling.py -q
```

Expected: PASS.

- [ ] **Step 3: Run frontend checks**

Run:

```powershell
cd frontend
npm run typecheck
npm run lint
```

Expected: PASS.

- [ ] **Step 4: Manual smoke test**

Run backend, worker, and frontend using the project’s normal dev commands. In the browser:

1. Select a job with active JD.
2. Upload a candidate PDF.
3. Wait until parsing completes.
4. Confirm evaluation appears on candidate detail.
5. Open Scoring page and confirm dashboard rows appear.
6. Change job weights and confirm scores recalculate without a new LLM call.
7. Edit `hidden_text` in JD page and confirm scoring status becomes `Outdated`.
8. Click `Score again` and confirm queued/running/completed statuses update.

- [ ] **Step 5: Commit final fixes**

```powershell
git status --short
git add backend frontend
git commit -m "test: verify scoring evaluation redesign"
```

Skip this commit if no files changed during final verification.

---

## Self-Review Checklist

- Spec coverage: candidate snapshots, JD hidden information ownership, outdated scoring, job-level weights, rule-based measurable criteria, auto trigger after candidate parse, Scoring page dashboard, Candidate detail evaluation, and score-again flow are each covered by at least one task.
- Placeholder scan: no task relies on unspecified implementation; every new module has concrete function names and test expectations.
- Type consistency: backend response names match frontend types: `scorePercent`, `effectiveWeight`, `weightedScore`, `totalScore`, `passedThreshold`, `componentScores`, and evaluation statuses.
- Scope check: existing `MatchRun` and `MatchResult` remain compatible during migration; the plan does not require deleting the old manual scoring endpoint.
