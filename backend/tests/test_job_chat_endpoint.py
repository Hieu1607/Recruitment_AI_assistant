import sys
from pathlib import Path
import uuid
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.v1.endpoints.jobs import ChatRequest, chat_about_job  # noqa: E402
from src.models.base import Base  # noqa: E402
from src.models.candidate_profile import CandidateProfile  # noqa: E402
from src.models.enums import ProfileStatus, UploadStatus, UserStatus  # noqa: E402
from src.models.job import Job  # noqa: E402
from src.models.resume_document import ResumeDocument  # noqa: E402
from src.models.user_account import UserAccount  # noqa: E402


def _create_test_tables(engine):
    tables = [
        Base.metadata.tables["user_accounts"],
        Base.metadata.tables["jobs"],
        Base.metadata.tables["resume_documents"],
        Base.metadata.tables["candidate_profiles"],
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
def outsider(db):
    user = UserAccount(
        email="outsider@example.com",
        display_name="Outsider",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture()
def owned_job_with_candidate(db, owner):
    job = Job(owner_user_id=owner.id, title="Platform Engineer", status="active")
    db.add(job)
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

    profile = CandidateProfile(
        resume_document_id=resume.id,
        full_name="Candidate One",
        email="candidate@example.com",
        current_job_title="Backend Engineer",
        skills_text="Python, FastAPI",
        profile_status=ProfileStatus.REVIEWED,
    )
    db.add(profile)
    db.commit()
    db.refresh(job)
    return job


def test_chat_about_job_returns_answer_and_candidate_scope(monkeypatch, db, owner, owned_job_with_candidate):
    class FakeGraph:
        def invoke(self, payload):
            return {
                "messages": payload["messages"],
                "answer": "Candidate One looks strong for backend work.",
                "dsl_candidates": [{"id": "candidate-1"}],
            }

    monkeypatch.setattr("src.api.v1.endpoints.jobs.get_graph", lambda: FakeGraph())

    response = chat_about_job(
        job_id=owned_job_with_candidate.id,
        body=ChatRequest(message="Who is the strongest backend candidate?"),
        db=db,
        current_user=owner,
    )

    assert response.answer == "Candidate One looks strong for backend work."
    assert response.candidates_in_scope == 1
    assert response.session_id


def test_chat_about_job_answers_total_count_without_graph(
    monkeypatch, db, owner, owned_job_with_candidate
):
    class ExplodingGraph:
        def invoke(self, payload):
            raise AssertionError("graph should not run for total candidate count questions")

    monkeypatch.setattr("src.api.v1.endpoints.jobs.get_graph", lambda: ExplodingGraph())

    response = chat_about_job(
        job_id=owned_job_with_candidate.id,
        body=ChatRequest(message="How many candidates are in this job?"),
        db=db,
        current_user=owner,
    )

    assert response.answer == "Có 1 ứng viên trong job này."
    assert response.candidates_in_scope == 1
    assert response.session_id


def test_chat_about_job_rejects_non_owner(db, outsider, owned_job_with_candidate):
    with pytest.raises(HTTPException) as exc_info:
        chat_about_job(
            job_id=owned_job_with_candidate.id,
            body=ChatRequest(message="Can I see these candidates?"),
            db=db,
            current_user=outsider,
        )

    assert exc_info.value.status_code == 404
