import sys
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.base import Base  # noqa: E402
from src.models.candidate_profile import CandidateProfile  # noqa: E402
from src.models.enums import (  # noqa: E402
    CandidateEvaluationStatus,
    ProfileStatus,
    ResumeProcessingBatchStatus,
    UploadStatus,
    UserStatus,
)
from src.models.job import Job  # noqa: E402
from src.models.job_matching import JobDescription  # noqa: E402
from src.models.resume_document import ResumeDocument  # noqa: E402
from src.models.resume_processing_batch import ResumeProcessingBatch  # noqa: E402
from src.models.scoring_evaluation import CandidateEvaluation  # noqa: E402
from src.models.user_account import UserAccount  # noqa: E402
from src.services import score_candidate  # noqa: E402
from src.services.candidate_evaluation_service import (  # noqa: E402
    current_signature_for_jd,
    evaluate_processing_batch,
)
from src.services.score_candidate import evaluate_candidate_profiles_raw  # noqa: E402
from src.services.resume_batch_service import mark_processing_batch_failed  # noqa: E402


def _candidates(count: int) -> list[dict]:
    return [
        {
            "id": f"candidate-{index}",
            "full_name": f"Candidate {index}",
            "display_name": f"Candidate {index}",
            "skills_text": "Python",
        }
        for index in range(count)
    ]


def _semantic_rubric() -> dict:
    return {
        "criteria": [
            {
                "key": "skills.python",
                "section": "skills",
                "requirementText": "Strong Python",
                "type": "semantic",
                "measurable": None,
                "weight": 1.0,
            }
        ],
        "sectionWeights": {"skills": 1.0},
    }


def test_batch_raw_evaluation_extracts_rubric_once_for_ten_candidates(monkeypatch):
    candidates = _candidates(10)
    calls = {"rubric": 0, "semantic": 0}

    monkeypatch.setattr(score_candidate, "_scoring_llm_provider", lambda: object())

    def extract_rubric(**_kwargs):
        calls["rubric"] += 1
        return _semantic_rubric()

    semantic_batches = [candidates[:8], candidates[8:]]

    def generate_semantic(**_kwargs):
        batch = semantic_batches[calls["semantic"]]
        calls["semantic"] += 1
        return {
            candidate["id"]: {
                "criteria": {
                    "skills.python": {
                        "scorePercent": 80,
                        "evidenceSummary": "Python project",
                    }
                }
            }
            for candidate in batch
        }

    monkeypatch.setattr(score_candidate, "_extract_locked_rubric", extract_rubric)
    monkeypatch.setattr(
        score_candidate,
        "_generate_semantic_scores_with_retries",
        generate_semantic,
    )
    monkeypatch.setattr(score_candidate.settings, "SCORING_MAX_CANDIDATES_PER_BATCH", 8)

    results = evaluate_candidate_profiles_raw(
        candidates=candidates,
        job_description_text="Need strong Python",
    )

    assert calls == {"rubric": 1, "semantic": 2}
    assert set(results) == {candidate["id"] for candidate in candidates}
    assert all(
        result["rawComponentScores"][0]["scorePercent"] == 80
        for result in results.values()
    )


def test_batch_raw_evaluation_skips_semantic_llm_for_measurable_only_rubric(monkeypatch):
    candidates = [
        {
            **candidate,
            "experience_years": 5,
        }
        for candidate in _candidates(3)
    ]
    rubric = {
        "criteria": [
            {
                "key": "experience.years",
                "section": "experience",
                "requirementText": "At least 3 years",
                "type": "must_have",
                "measurable": {
                    "field": "experience_years",
                    "operator": ">=",
                    "value": 3,
                },
                "weight": 1.0,
            }
        ],
        "sectionWeights": {"experience": 1.0},
    }
    semantic_calls = []

    monkeypatch.setattr(score_candidate, "_scoring_llm_provider", lambda: object())
    monkeypatch.setattr(
        score_candidate,
        "_extract_locked_rubric",
        lambda **_kwargs: rubric,
    )
    monkeypatch.setattr(
        score_candidate,
        "_generate_semantic_scores_with_retries",
        lambda **kwargs: semantic_calls.append(kwargs) or {},
    )

    results = evaluate_candidate_profiles_raw(
        candidates=candidates,
        job_description_text="Need at least 3 years",
    )

    assert semantic_calls == []
    assert all(
        result["rawComponentScores"][0]["scorePercent"] == 100
        for result in results.values()
    )


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    tables = [
        Base.metadata.tables["user_accounts"],
        Base.metadata.tables["jobs"],
        Base.metadata.tables["resume_processing_batches"],
        Base.metadata.tables["resume_documents"],
        Base.metadata.tables["candidate_profiles"],
        Base.metadata.tables["job_descriptions"],
        Base.metadata.tables["match_runs"],
        Base.metadata.tables["candidate_evaluations"],
    ]
    Base.metadata.create_all(engine, tables=tables)
    with Session(engine) as session:
        yield session


def _processing_batch_context(db, *, processed_count: int, failed_count: int = 0):
    owner = UserAccount(
        email=f"evaluation-{uuid.uuid4()}@example.com",
        display_name="Owner",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    db.add(owner)
    db.flush()
    job = Job(owner_user_id=owner.id, title="AI Engineer", status="active")
    db.add(job)
    db.flush()
    jd = JobDescription(
        job_id=job.id,
        title="AI Engineer JD",
        jd_text="Need strong Python",
        hidden_text="Prefer production systems",
        created_by_user_id=owner.id,
        is_active=True,
    )
    db.add(jd)
    batch = ResumeProcessingBatch(
        job_id=job.id,
        total_count=processed_count + failed_count,
        terminal_count=processed_count + failed_count,
        processed_count=processed_count,
        failed_count=failed_count,
        status=ResumeProcessingBatchStatus.EVALUATION_PENDING,
    )
    db.add(batch)
    db.flush()
    profiles = []
    for index in range(processed_count):
        resume = ResumeDocument(
            original_file_name=f"candidate-{index}.pdf",
            storage_uri=f"s3://bucket/candidate-{index}.pdf",
            upload_status=UploadStatus.PROCESSED,
            job_id=job.id,
            uploaded_by_user_id=owner.id,
            retention_expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
            processing_batch_id=batch.id,
        )
        db.add(resume)
        db.flush()
        profile = CandidateProfile(
            resume_document_id=resume.id,
            full_name=f"Candidate {index}",
            experience_years=Decimal("5"),
            profile_status=ProfileStatus.REVIEWED,
            skills_text="Python",
        )
        db.add(profile)
        profiles.append(profile)
    for index in range(failed_count):
        db.add(
            ResumeDocument(
                original_file_name=f"failed-{index}.pdf",
                storage_uri=f"s3://bucket/failed-{index}.pdf",
                upload_status=UploadStatus.FAILED,
                job_id=job.id,
                uploaded_by_user_id=owner.id,
                retention_expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
                processing_batch_id=batch.id,
            )
        )
    db.commit()
    return batch, jd, profiles


def _raw_result(candidate_id: str) -> dict:
    return {
        "rubricPayload": _semantic_rubric(),
        "rawComponentScores": [
            {
                "criterionKey": "skills.python",
                "section": "skills",
                "criterionType": "semantic",
                "evaluationMode": "semantic",
                "requirementText": "Strong Python",
                "scorePercent": 88,
                "evidenceSummary": "Python projects",
            }
        ],
        "rationaleSummary": f"Strong match for {candidate_id}",
    }


def test_evaluate_processing_batch_persists_successes_with_partial_parse_failure(
    db,
    monkeypatch,
):
    batch, _jd, profiles = _processing_batch_context(
        db,
        processed_count=3,
        failed_count=1,
    )
    engine_inputs = []

    def raw_batch_engine(*, candidates, **_kwargs):
        engine_inputs.extend(candidate["id"] for candidate in candidates)
        return {
            candidate["id"]: _raw_result(candidate["id"])
            for candidate in candidates
        }

    monkeypatch.setattr(
        score_candidate,
        "evaluate_candidate_profiles_raw",
        raw_batch_engine,
    )

    result = evaluate_processing_batch(
        db=db,
        processing_batch_id=batch.id,
        worker_task_id="task-1",
    )

    db.refresh(batch)
    evaluations = db.query(CandidateEvaluation).all()
    assert result.completed == 3
    assert result.failed == 0
    assert result.skipped == 0
    assert set(engine_inputs) == {str(profile.id) for profile in profiles}
    assert len(evaluations) == 3
    assert all(
        evaluation.status == CandidateEvaluationStatus.COMPLETED
        for evaluation in evaluations
    )
    assert batch.status == ResumeProcessingBatchStatus.COMPLETED_WITH_ERRORS


def test_evaluate_processing_batch_skips_completed_snapshot(db, monkeypatch):
    batch, jd, profiles = _processing_batch_context(db, processed_count=3)
    completed_profile = profiles[0]
    completed = CandidateEvaluation(
        job_id=batch.job_id,
        job_description_id=jd.id,
        candidate_profile_id=completed_profile.id,
        scoring_signature=current_signature_for_jd(jd),
        rubric_payload=_semantic_rubric(),
        raw_component_scores=[],
        rationale_summary="Already completed",
        status=CandidateEvaluationStatus.COMPLETED,
        scored_at=datetime.now(timezone.utc),
    )
    db.add(completed)
    db.commit()
    engine_inputs = []

    def raw_batch_engine(*, candidates, **_kwargs):
        engine_inputs.extend(candidate["id"] for candidate in candidates)
        return {
            candidate["id"]: _raw_result(candidate["id"])
            for candidate in candidates
        }

    monkeypatch.setattr(
        score_candidate,
        "evaluate_candidate_profiles_raw",
        raw_batch_engine,
    )

    result = evaluate_processing_batch(
        db=db,
        processing_batch_id=batch.id,
        worker_task_id="task-2",
    )

    db.refresh(completed)
    assert result.completed == 2
    assert result.skipped == 1
    assert str(completed_profile.id) not in engine_inputs
    assert completed.rationale_summary == "Already completed"


def test_mark_processing_batch_failed_closes_running_evaluations(db):
    batch, jd, profiles = _processing_batch_context(db, processed_count=2)
    signature = current_signature_for_jd(jd)
    evaluations = [
        CandidateEvaluation(
            job_id=batch.job_id,
            job_description_id=jd.id,
            candidate_profile_id=profile.id,
            scoring_signature=signature,
            rubric_payload={},
            raw_component_scores=[],
            rationale_summary="",
            status=CandidateEvaluationStatus.RUNNING,
        )
        for profile in profiles
    ]
    db.add_all(evaluations)
    db.commit()

    mark_processing_batch_failed(
        db,
        batch.id,
        error_message="provider unavailable",
    )

    db.refresh(batch)
    for evaluation in evaluations:
        db.refresh(evaluation)
    assert batch.status == ResumeProcessingBatchStatus.FAILED
    assert batch.completed_at is not None
    assert all(
        evaluation.status == CandidateEvaluationStatus.FAILED
        for evaluation in evaluations
    )
    assert all(
        evaluation.error_message == "provider unavailable"
        for evaluation in evaluations
    )
