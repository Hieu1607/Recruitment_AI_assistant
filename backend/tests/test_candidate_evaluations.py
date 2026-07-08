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
from src.models.enums import CandidateEvaluationStatus, ProfileStatus, UploadStatus, UserStatus  # noqa: E402
from src.models.job import Job  # noqa: E402
from src.models.job_matching import JobDescription  # noqa: E402
from src.models.resume_document import ResumeDocument  # noqa: E402
from src.models.scoring_evaluation import CandidateEvaluation, JobScoringPreference  # noqa: E402
from src.models.user_account import UserAccount  # noqa: E402
from src.services.candidate_evaluation_service import (  # noqa: E402
    evaluate_candidate_for_current_jd,
    mark_job_evaluations_outdated,
    serialize_candidate_evaluation,
)


def _create_test_tables(engine) -> None:
    tables = [
        Base.metadata.tables["user_accounts"],
        Base.metadata.tables["jobs"],
        Base.metadata.tables["resume_documents"],
        Base.metadata.tables["candidate_profiles"],
        Base.metadata.tables["job_descriptions"],
        Base.metadata.tables["match_runs"],
        Base.metadata.tables["candidate_evaluations"],
        Base.metadata.tables["job_scoring_preferences"],
    ]
    Base.metadata.create_all(engine, tables=tables)


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    _create_test_tables(engine)
    with Session(engine) as session:
        yield session


def test_candidate_evaluation_persists_raw_scores(db):
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


def test_mark_job_evaluations_outdated_only_changes_old_signature(db):
    owner = UserAccount(
        email="owner@example.com",
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
        jd_text="Need Python",
        hidden_text="Prefer RAG",
        created_by_user_id=owner.id,
        is_active=True,
    )
    db.add(jd)
    db.flush()

    resume = ResumeDocument(
        original_file_name="candidate.pdf",
        storage_uri="s3://bucket/candidate.pdf",
        upload_status=UploadStatus.PROCESSED,
        job_id=job.id,
        uploaded_by_user_id=owner.id,
        retention_expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
    )
    db.add(resume)
    db.flush()

    candidate = CandidateProfile(
        resume_document_id=resume.id,
        full_name="Candidate One",
        profile_status=ProfileStatus.REVIEWED,
    )
    db.add(candidate)
    db.flush()

    old = CandidateEvaluation(
        job_id=job.id,
        job_description_id=jd.id,
        candidate_profile_id=candidate.id,
        scoring_signature="old",
        rubric_payload={},
        raw_component_scores=[],
        rationale_summary="old",
        status=CandidateEvaluationStatus.COMPLETED.value,
    )
    current = CandidateEvaluation(
        job_id=job.id,
        job_description_id=jd.id,
        candidate_profile_id=candidate.id,
        scoring_signature="current",
        rubric_payload={},
        raw_component_scores=[],
        rationale_summary="current",
        status=CandidateEvaluationStatus.COMPLETED.value,
    )
    db.add_all([old, current])
    db.commit()

    updated = mark_job_evaluations_outdated(
        db=db,
        job_id=job.id,
        current_scoring_signature="current",
    )
    db.commit()
    db.refresh(old)
    db.refresh(current)

    assert updated == 1
    assert old.status == CandidateEvaluationStatus.OUTDATED.value
    assert current.status == CandidateEvaluationStatus.COMPLETED.value


def test_serialize_candidate_evaluation_applies_job_preferences():
    evaluation = CandidateEvaluation(
        id=uuid.uuid4(),
        job_id=uuid.uuid4(),
        job_description_id=uuid.uuid4(),
        candidate_profile_id=uuid.uuid4(),
        scoring_signature="sig-a",
        rubric_payload={},
        raw_component_scores=[
            {
                "criterionKey": "skills.python",
                "section": "skills",
                "scorePercent": 80,
                "evaluationMode": "semantic",
                "criterionType": "must_have",
                "requirementText": "Python",
                "evidenceSummary": "Python project.",
            },
            {
                "criterionKey": "experience.years",
                "section": "experience",
                "scorePercent": 50,
                "evaluationMode": "measurable",
                "criterionType": "must_have",
                "requirementText": "2+ years",
                "evidenceSummary": "One year listed.",
            },
        ],
        rationale_summary="Strong Python match.",
        status=CandidateEvaluationStatus.COMPLETED.value,
    )
    preference = JobScoringPreference(
        job_id=evaluation.job_id,
        section_weights={"skills": 75, "experience": 25},
        score_threshold=Decimal("70"),
    )

    payload = serialize_candidate_evaluation(evaluation, preference)

    assert payload["totalScore"] == 72.5
    assert payload["passedThreshold"] is True
    assert payload["componentScores"][0]["weightedScore"] == 60.0


def test_evaluate_candidate_for_current_jd_persists_completed_snapshot(db, monkeypatch):
    owner = UserAccount(
        email="owner@example.com",
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
        jd_text="Need Python",
        hidden_text="Prefer RAG",
        created_by_user_id=owner.id,
        is_active=True,
    )
    db.add(jd)
    db.flush()

    resume = ResumeDocument(
        original_file_name="candidate.pdf",
        storage_uri="s3://bucket/candidate.pdf",
        upload_status=UploadStatus.PROCESSED,
        job_id=job.id,
        uploaded_by_user_id=owner.id,
        retention_expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
    )
    db.add(resume)
    db.flush()

    candidate = CandidateProfile(
        resume_document_id=resume.id,
        full_name="Candidate One",
        experience_years=Decimal("6.0"),
        profile_status=ProfileStatus.REVIEWED,
    )
    db.add(candidate)
    db.commit()

    monkeypatch.setattr(
        "src.services.score_candidate.evaluate_candidate_profile_raw",
        lambda **kwargs: {
            "rubricPayload": {"criteria": [{"key": "skills.python"}]},
            "rawComponentScores": [
                {
                    "criterionKey": "skills.python",
                    "section": "skills",
                    "criterionType": "semantic",
                    "evaluationMode": "semantic",
                    "requirementText": "Python",
                    "scorePercent": 88,
                    "evidenceSummary": "Strong Python evidence.",
                }
            ],
            "rationaleSummary": "Strong Python match.",
        },
    )

    evaluation = evaluate_candidate_for_current_jd(db=db, candidate_profile_id=candidate.id)

    db.refresh(evaluation)
    assert evaluation.status == CandidateEvaluationStatus.COMPLETED.value
    assert evaluation.scoring_signature
    assert evaluation.raw_component_scores[0]["scorePercent"] == 88
    assert evaluation.rationale_summary == "Strong Python match."
    assert evaluation.scored_at is not None
