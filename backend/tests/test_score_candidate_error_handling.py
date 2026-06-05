import sys
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.base import Base  # noqa: E402
from src.models.candidate_profile import CandidateProfile  # noqa: E402
from src.models.enums import MatchRunStatus, ProfileStatus, UploadStatus, UserStatus  # noqa: E402
from src.models.job import Job  # noqa: E402
from src.models.job_matching import JobDescription, MatchRun  # noqa: E402
from src.models.resume_document import ResumeDocument  # noqa: E402
from src.models.user_account import UserAccount  # noqa: E402
from src.services.llm_service import LLMProviderError, LLMProviderLimitError  # noqa: E402
from src.services.score_candidate import score_candidates  # noqa: E402
from src.services.scoring_errors import ScoringProviderLimitError  # noqa: E402


def _create_test_tables(engine):
    tables = [
        Base.metadata.tables["user_accounts"],
        Base.metadata.tables["jobs"],
        Base.metadata.tables["job_descriptions"],
        Base.metadata.tables["resume_documents"],
        Base.metadata.tables["candidate_profiles"],
        Base.metadata.tables["match_runs"],
        Base.metadata.tables["match_results"],
    ]
    Base.metadata.create_all(engine, tables=tables)


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
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
def scoring_context(db, owner):
    job = Job(owner_user_id=owner.id, title="Senior AI Engineer", status="active")
    db.add(job)
    db.flush()

    jd = JobDescription(
        job_id=job.id,
        title="Senior AI Engineer",
        jd_text="Build production AI systems with Python.",
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
        experience_text="Built AI systems",
        skills_text="Python, FastAPI",
    )
    db.add(candidate)
    db.commit()
    db.refresh(jd)
    db.refresh(candidate)

    return SimpleNamespace(job=job, job_description=jd, candidate=candidate)


def test_score_candidates_marks_run_failed_and_logs_error_when_provider_quota_is_exhausted(
    monkeypatch, caplog, db, owner, scoring_context
):
    monkeypatch.setattr(
        "src.services.score_candidate._scoring_llm_provider",
        lambda: object(),
    )
    monkeypatch.setattr(
        "src.services.score_candidate._extract_locked_rubric",
        lambda **kwargs: (_ for _ in ()).throw(
            LLMProviderError(
                "Groq request failed: Error code: 429 - {'error': {'message': 'Rate limit reached for model on tokens per day (TPD)', 'code': 'rate_limit_exceeded'}}"
            )
        ),
    )
    caplog.set_level("ERROR")

    with pytest.raises(ScoringProviderLimitError):
        score_candidates(
            db=db,
            job_description_id=scoring_context.job_description.id,
            initiated_by_user_id=owner.id,
            score_threshold=Decimal("50"),
        )

    runs = db.execute(select(MatchRun)).scalars().all()
    assert len(runs) == 1
    assert runs[0].run_status == MatchRunStatus.FAILED
    assert runs[0].completed_at is not None
    assert any(record.levelname in {"ERROR", "CRITICAL"} for record in caplog.records)
    assert any("quota" in record.getMessage().lower() or "rate limit" in record.getMessage().lower() for record in caplog.records)
    assert not any(record.exc_info for record in caplog.records)


def test_score_candidates_does_not_continue_when_semantic_scoring_hits_provider_limit(
    monkeypatch, caplog, db, owner, scoring_context
):
    class QuotaLimitedLLM:
        provider = "groq"
        model_name = "llama-3.3-70b-versatile"

        def generate(self, prompt):
            raise LLMProviderLimitError("Groq chat request hit quota or rate limit")

        def clone_with_model(self, **kwargs):
            return self

    monkeypatch.setattr(
        "src.services.score_candidate._scoring_llm_provider",
        lambda: QuotaLimitedLLM(),
    )
    monkeypatch.setattr(
        "src.services.score_candidate._extract_locked_rubric",
        lambda **kwargs: {
            "criteria": [
                {
                    "key": "skills.1",
                    "section": "skills",
                    "requirementText": "Python",
                    "type": "semantic",
                    "measurable": None,
                    "weight": 1.0,
                }
            ],
            "sectionWeights": {"skills": 1.0},
        },
    )
    caplog.set_level("ERROR")

    with pytest.raises(ScoringProviderLimitError):
        score_candidates(
            db=db,
            job_description_id=scoring_context.job_description.id,
            initiated_by_user_id=owner.id,
            score_threshold=Decimal("50"),
        )

    runs = db.execute(select(MatchRun)).scalars().all()
    assert len(runs) == 1
    assert runs[0].run_status == MatchRunStatus.FAILED
    assert runs[0].completed_at is not None
    assert any("quota" in record.getMessage().lower() or "rate limit" in record.getMessage().lower() for record in caplog.records)
    assert not any(record.exc_info for record in caplog.records)
