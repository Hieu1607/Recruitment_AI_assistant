import sys
import types
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pydantic_settings  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("pydantic_settings")

    class BaseSettings:
        pass

    stub.BaseSettings = BaseSettings
    sys.modules["pydantic_settings"] = stub

if "jose" not in sys.modules:
    jose_stub = types.ModuleType("jose")

    class JWTError(Exception):
        pass

    jose_stub.JWTError = JWTError
    jose_stub.jwt = types.SimpleNamespace(decode=lambda *args, **kwargs: {})
    sys.modules["jose"] = jose_stub

if "langchain_core.messages" not in sys.modules:
    langchain_core = types.ModuleType("langchain_core")
    messages = types.ModuleType("langchain_core.messages")

    class HumanMessage:
        def __init__(self, content):
            self.content = content

    class AIMessage:
        def __init__(self, content):
            self.content = content

    messages.HumanMessage = HumanMessage
    messages.AIMessage = AIMessage
    sys.modules["langchain_core"] = langchain_core
    sys.modules["langchain_core.messages"] = messages

if "multipart" not in sys.modules:
    multipart_stub = types.ModuleType("multipart")
    multipart_stub.__version__ = "0.0-test"
    multipart_multipart_stub = types.ModuleType("multipart.multipart")
    multipart_multipart_stub.parse_options_header = lambda value: ("", {})
    sys.modules["multipart"] = multipart_stub
    sys.modules["multipart.multipart"] = multipart_multipart_stub

if "src.services.ai_agent.graph" not in sys.modules:
    graph_stub = types.ModuleType("src.services.ai_agent.graph")
    graph_stub.get_graph = lambda: types.SimpleNamespace(invoke=lambda payload: payload)
    sys.modules["src.services.ai_agent.graph"] = graph_stub

if "src.services.job_description_service" not in sys.modules:
    jd_stub = types.ModuleType("src.services.job_description_service")
    jd_stub._jd_to_dict = lambda jd: {}
    sys.modules["src.services.job_description_service"] = jd_stub

if "src.services.resume_service" not in sys.modules:
    resume_stub = types.ModuleType("src.services.resume_service")
    resume_stub._normalize_location_name = lambda value: value
    resume_stub._resume_to_dict = lambda resume: {}
    resume_stub.create_resume_document = lambda **kwargs: types.SimpleNamespace(
        id=uuid.uuid4()
    )
    resume_stub.parse_pdf_to_sections = lambda **kwargs: []
    sys.modules["src.services.resume_service"] = resume_stub

if "src.services.score_candidate" not in sys.modules:
    score_stub = types.ModuleType("src.services.score_candidate")
    score_stub.score_candidates = lambda **kwargs: {}
    sys.modules["src.services.score_candidate"] = score_stub

from src.api.v1.endpoints.jobs import (  # noqa: E402
    ScoreRequest,
    get_score_run,
    score_job_candidates,
)
from src.models.base import Base  # noqa: E402
from src.models.candidate_profile import CandidateProfile  # noqa: E402
from src.models.enums import MatchRunStatus, ProfileStatus, UploadStatus, UserStatus  # noqa: E402
from src.models.job import Job  # noqa: E402
from src.models.job_matching import JobDescription, MatchResult, MatchRun  # noqa: E402
from src.models.resume_document import ResumeDocument  # noqa: E402
from src.models.user_account import UserAccount  # noqa: E402
from src.services.scoring_errors import ScoringProviderLimitError  # noqa: E402


def _create_test_tables(engine):
    tables = [
        Base.metadata.tables["user_accounts"],
        Base.metadata.tables["jobs"],
        Base.metadata.tables["resume_documents"],
        Base.metadata.tables["candidate_profiles"],
        Base.metadata.tables["job_descriptions"],
        Base.metadata.tables["match_runs"],
        Base.metadata.tables["match_results"],
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
def owned_score_run(db, owner):
    job = Job(owner_user_id=owner.id, title="Teacher", status="active")
    db.add(job)
    db.flush()

    jd = JobDescription(
        job_id=job.id,
        title="Math teacher",
        jd_text="Updated job description",
        hidden_text="",
        created_by_user_id=owner.id,
        is_active=True,
    )
    db.add(jd)
    db.flush()

    resume = ResumeDocument(
        original_file_name="candidate.pdf",
        storage_uri="s3://bucket/resumes/candidate.pdf",
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
        current_job_title="Teacher",
        experience_years=Decimal("5.0"),
        education_text="Bachelor of Science in Mathematics",
        profile_status=ProfileStatus.REVIEWED,
    )
    db.add(candidate)
    db.flush()

    run = MatchRun(
        job_description_id=jd.id,
        score_threshold=Decimal("50.0"),
        initiated_by_user_id=owner.id,
        run_status=MatchRunStatus.COMPLETED,
        completed_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.flush()

    result = MatchResult(
        match_run_id=run.id,
        candidate_profile_id=candidate.id,
        score_list_index=0,
        total_score=Decimal("88.50"),
        passed_threshold=True,
        rationale_summary="Overall score 88.5/100.",
        component_scores=[
            {
                "criterionKey": "math_degree",
                "criterionType": "semantic",
                "evaluationMode": "semantic",
                "requirementText": "Bachelor's degree in mathematics.",
                "weight": 1.0,
                "score": 88.5,
                "weightedScore": 88.5,
                "evidenceSummary": "Candidate studied mathematics.",
            }
        ],
    )
    db.add(result)
    db.commit()
    db.refresh(run)
    db.refresh(candidate)
    db.refresh(job)
    return {"job": job, "run": run, "candidate": candidate, "jd": jd}


def test_job_score_endpoint_returns_429_when_scoring_provider_limit_is_hit(
    monkeypatch, db, owner
):
    job_id = uuid.uuid4()
    jd = types.SimpleNamespace(id=uuid.uuid4())

    monkeypatch.setattr(
        "src.api.v1.endpoints.jobs.require_job_scoped_jd",
        lambda *args, **kwargs: jd,
    )
    monkeypatch.setattr(
        "src.api.v1.endpoints.jobs.score_candidates",
        lambda **kwargs: (_ for _ in ()).throw(
            ScoringProviderLimitError(
                "Scoring is temporarily unavailable because the configured LLM quota has been exhausted. Please retry later."
            )
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        score_job_candidates(
            job_id=job_id,
            body=ScoreRequest(),
            db=db,
            current_user=owner,
        )

    assert exc_info.value.status_code == 429
    assert "quota" in exc_info.value.detail.lower()


def test_job_score_endpoint_defaults_batch_size_to_three(monkeypatch, db, owner):
    job_id = uuid.uuid4()
    jd = types.SimpleNamespace(id=uuid.uuid4())
    captured = {}

    monkeypatch.setattr(
        "src.api.v1.endpoints.jobs.require_job_scoped_jd",
        lambda *args, **kwargs: jd,
    )

    def fake_score_candidates(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        "src.api.v1.endpoints.jobs.score_candidates",
        fake_score_candidates,
    )
    monkeypatch.setattr(
        "src.api.v1.endpoints.jobs.create_notification",
        lambda **kwargs: None,
    )

    score_job_candidates(
        job_id=job_id,
        body=ScoreRequest(),
        db=db,
        current_user=owner,
    )

    assert captured["batch_size"] == 3


def test_job_score_endpoint_creates_completion_notification(monkeypatch, db, owner):
    job_id = uuid.uuid4()
    jd = types.SimpleNamespace(id=uuid.uuid4())
    match_run_id = uuid.uuid4()
    captured = {}

    monkeypatch.setattr(
        "src.api.v1.endpoints.jobs.require_job_scoped_jd",
        lambda *args, **kwargs: jd,
    )
    monkeypatch.setattr(
        "src.api.v1.endpoints.jobs.score_candidates",
        lambda **kwargs: {
            "match_run_id": str(match_run_id),
            "job_description_id": str(jd.id),
            "total_candidates": 2,
            "total_passed_candidates": 1,
            "batches": 1,
            "scores": [],
        },
    )

    def fake_create_notification(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "src.api.v1.endpoints.jobs.create_notification",
        fake_create_notification,
        raising=False,
    )

    score_job_candidates(
        job_id=job_id,
        body=ScoreRequest(),
        db=db,
        current_user=owner,
    )

    assert captured["db"] is db
    assert captured["user_id"] == owner.id
    assert captured["notification_type"] == "scoring_completed"
    assert captured["target_url"] == f"/scoring/{match_run_id}"
    assert captured["metadata"]["total_candidates"] == 2
    assert captured["metadata"]["total_passed_candidates"] == 1


def test_get_score_run_returns_persisted_run_result(db, owner, owned_score_run):
    payload = get_score_run(
        job_id=owned_score_run["job"].id,
        match_run_id=owned_score_run["run"].id,
        db=db,
        current_user=owner,
    )

    assert payload.match_run_id == str(owned_score_run["run"].id)
    assert payload.job_description_id == str(owned_score_run["jd"].id)
    assert payload.total_candidates == 1
    assert payload.total_passed_candidates == 1
    assert payload.scores[0].candidateId == str(owned_score_run["candidate"].id)
    assert payload.scores[0].candidateName == "Candidate One"
    assert payload.scores[0].resumeFileName == "candidate.pdf"
    assert payload.scores[0].totalScore == 88.5
    assert payload.scores[0].componentScores[0].criterionKey == "math_degree"
