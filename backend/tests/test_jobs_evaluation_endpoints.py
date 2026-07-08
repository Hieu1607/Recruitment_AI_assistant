import sys
import types
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "src.services.job_description_service" not in sys.modules:
    jd_stub = types.ModuleType("src.services.job_description_service")
    jd_stub._jd_to_dict = lambda jd: {}
    sys.modules["src.services.job_description_service"] = jd_stub

from src.api.v1.endpoints.jobs import (  # noqa: E402
    JobScoringPreferenceRequest,
    get_candidate_evaluation,
    list_job_evaluations,
    patch_job_description,
    score_again_job_evaluations,
    update_job_scoring_preferences,
)
from src.models.base import Base  # noqa: E402
from src.models.candidate_profile import CandidateProfile  # noqa: E402
from src.models.enums import CandidateEvaluationStatus, ProfileStatus, UploadStatus, UserStatus  # noqa: E402
from src.models.job import Job  # noqa: E402
from src.models.job_matching import JobDescription  # noqa: E402
from src.models.resume_document import ResumeDocument  # noqa: E402
from src.models.scoring_evaluation import CandidateEvaluation  # noqa: E402
from src.models.user_account import UserAccount  # noqa: E402
from src.services.candidate_evaluation_service import current_signature_for_jd  # noqa: E402


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


@pytest.fixture()
def owner(db):
    user = UserAccount(
        email="owner@example.com",
        display_name="Owner",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture()
def job_with_completed_evaluation(db, owner):
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
    db.flush()

    evaluation = CandidateEvaluation(
        job_id=job.id,
        job_description_id=jd.id,
        candidate_profile_id=candidate.id,
        scoring_signature=current_signature_for_jd(jd),
        rubric_payload={"criteria": [{"key": "skills.python"}]},
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
                "criterionKey": "skills.git",
                "section": "skills",
                "criterionType": "semantic",
                "evaluationMode": "semantic",
                "requirementText": "Git workflow",
                "scorePercent": 70,
                "evidenceSummary": "Uses Git daily.",
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
        rationale_summary="Strong Python match.",
        status=CandidateEvaluationStatus.COMPLETED.value,
        scored_at=datetime.now(timezone.utc),
    )
    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)
    db.refresh(candidate)
    db.refresh(job)
    db.refresh(jd)
    return {
        "job": job,
        "jd": jd,
        "candidate": candidate,
        "evaluation": evaluation,
    }


def test_patch_job_description_marks_evaluations_outdated(db, owner, job_with_completed_evaluation):
    patch_job_description(
        job_id=job_with_completed_evaluation["job"].id,
        body=types.SimpleNamespace(hidden_text="new hidden criteria", jd_text=None, title=None, is_active=None),
        db=db,
        current_user=owner,
    )

    evaluation = db.get(CandidateEvaluation, job_with_completed_evaluation["evaluation"].id)
    assert evaluation.status == CandidateEvaluationStatus.OUTDATED.value


def test_get_job_evaluations_returns_weighted_scores(db, owner, job_with_completed_evaluation):
    payload = list_job_evaluations(
        job_id=job_with_completed_evaluation["job"].id,
        db=db,
        current_user=owner,
    )

    assert payload.completed_count == 1
    assert payload.items[0].totalScore == 66.66
    assert payload.section_weights == {"skills": 66.67, "experience": 33.33}
    assert payload.scoring_preferences_applied is False


def test_get_candidate_evaluation_returns_weighted_detail(db, owner, job_with_completed_evaluation):
    payload = get_candidate_evaluation(
        job_id=job_with_completed_evaluation["job"].id,
        candidate_profile_id=job_with_completed_evaluation["candidate"].id,
        db=db,
        current_user=owner,
    )

    assert payload.candidate_profile_id == str(job_with_completed_evaluation["candidate"].id)
    assert payload.componentScores[0].scorePercent == 80


def test_put_scoring_preferences_recalculates_without_llm(db, owner, job_with_completed_evaluation):
    payload = update_job_scoring_preferences(
        job_id=job_with_completed_evaluation["job"].id,
        body=JobScoringPreferenceRequest(
            section_weights={"skills": 100},
            score_threshold=60,
        ),
        db=db,
        current_user=owner,
    )

    assert payload.section_weights == {"skills": 100.0}
    assert payload.score_threshold == 60.0

    evaluations = list_job_evaluations(
        job_id=job_with_completed_evaluation["job"].id,
        db=db,
        current_user=owner,
    )
    assert evaluations.scoring_preferences_applied is True


def test_score_again_job_evaluations_enqueues_missing_current_signature(db, owner, job_with_completed_evaluation, monkeypatch):
    queued: list[str] = []

    patch_job_description(
        job_id=job_with_completed_evaluation["job"].id,
        body=types.SimpleNamespace(hidden_text="new hidden criteria", jd_text=None, title=None, is_active=None),
        db=db,
        current_user=owner,
    )
    monkeypatch.setattr(
        "worker.tasks.evaluate_candidate.delay",
        lambda candidate_profile_id: queued.append(candidate_profile_id) or types.SimpleNamespace(id="eval-task-id"),
    )

    payload = score_again_job_evaluations(
        job_id=job_with_completed_evaluation["job"].id,
        db=db,
        current_user=owner,
    )

    assert payload["queued"] == 1
    assert queued == [str(job_with_completed_evaluation["candidate"].id)]
    refreshed = get_candidate_evaluation(
        job_id=job_with_completed_evaluation["job"].id,
        candidate_profile_id=job_with_completed_evaluation["candidate"].id,
        db=db,
        current_user=owner,
    )
    assert refreshed.status == CandidateEvaluationStatus.PENDING.value
