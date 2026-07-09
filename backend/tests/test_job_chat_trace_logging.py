import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.pop("VITE_UI_LANGUAGE", None)

from src.api.v1.endpoints.jobs import ChatRequest, chat_about_job  # noqa: E402
from src.models.base import Base  # noqa: E402
from src.models.candidate_profile import CandidateProfile  # noqa: E402
from src.models.enums import ProfileStatus, UploadStatus, UserStatus  # noqa: E402
from src.models.job import Job  # noqa: E402
from src.models.job_matching import JobDescription  # noqa: E402
from src.models.resume_document import ResumeDocument  # noqa: E402
from src.models.user_account import UserAccount  # noqa: E402


def _create_test_tables(engine):
    tables = [
        Base.metadata.tables["user_accounts"],
        Base.metadata.tables["jobs"],
        Base.metadata.tables["resume_processing_batches"],
        Base.metadata.tables["resume_documents"],
        Base.metadata.tables["candidate_profiles"],
        Base.metadata.tables["job_descriptions"],
        Base.metadata.tables["query_sessions"],
        Base.metadata.tables["query_turns"],
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


def test_chat_about_job_writes_langgraph_trace_log(monkeypatch, tmp_path, db, owner, owned_job_with_candidate):
    jd = JobDescription(
        title="AI Platform",
        jd_text="Build LLM applications and ranking workflows.",
        hidden_text="Prefer production ML deployment experience.",
        job_id=owned_job_with_candidate.id,
        created_by_user_id=owner.id,
        is_active=True,
    )
    db.add(jd)
    db.commit()

    class FakeGraph:
        def invoke(self, payload):
            return {
                "messages": payload["messages"],
                "answer": "Candidate One looks strong for backend work.",
                "dsl_candidates": [{"id": "candidate-1"}],
            }

    monkeypatch.setattr("src.api.v1.endpoints.jobs.get_graph", lambda: FakeGraph())
    monkeypatch.setenv("LANGGRAPH_TRACE_LOG_DIR", str(tmp_path))

    response = chat_about_job(
        job_id=owned_job_with_candidate.id,
        body=ChatRequest(message="Who is the strongest backend candidate?"),
        db=db,
        current_user=owner,
    )

    assert response.answer == "Candidate One looks strong for backend work."

    trace_files = list((tmp_path / "langgraph").glob("*/*.json"))
    assert len(trace_files) == 1

    payload = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["metadata"]["endpoint"] == "job_chat"
    assert payload["metadata"]["job_id"] == str(owned_job_with_candidate.id)
    assert payload["graph_input"]["current_job"] == {
        "job_id": str(owned_job_with_candidate.id),
        "job_title": "Platform Engineer",
        "job_description_id": str(jd.id),
        "job_description_title": "AI Platform",
        "job_description_text": "Build LLM applications and ranking workflows.",
        "job_hidden_text": "Prefer production ML deployment experience.",
    }
    assert payload["graph_output"]["answer"] == "Candidate One looks strong for backend work."
