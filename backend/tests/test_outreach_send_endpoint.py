import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.main import app
from src.models.base import Base
from src.models.candidate_profile import CandidateProfile
from src.models.deps import get_current_user, get_db
from src.models.enums import ContentSource, ProfileStatus, SentStatus, UploadStatus, UserStatus
from src.models.job import Job
from src.models.outreach import OutreachMessage
from src.models.resume_document import ResumeDocument
from src.models.user_account import UserAccount


def _create_test_tables(engine):
    tables = [
        Base.metadata.tables["user_accounts"],
        Base.metadata.tables["jobs"],
        Base.metadata.tables["resume_documents"],
        Base.metadata.tables["candidate_profiles"],
        Base.metadata.tables["outreach_messages"],
    ]
    Base.metadata.create_all(engine, tables=tables)


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    _create_test_tables(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    with factory() as session:
        yield session


@pytest.fixture()
def seeded_outreach_message(db_session: Session):
    user = UserAccount(
        email="owner@example.com",
        display_name="Owner",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    db_session.add(user)
    db_session.flush()

    job = Job(owner_user_id=user.id, title="Platform Engineer", status="active")
    db_session.add(job)
    db_session.flush()

    resume = ResumeDocument(
        original_file_name="candidate.pdf",
        storage_uri="s3://bucket/resumes/candidate.pdf",
        upload_status=UploadStatus.PROCESSED,
        job_id=job.id,
        uploaded_by_user_id=user.id,
        retention_expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
    )
    db_session.add(resume)
    db_session.flush()

    candidate = CandidateProfile(
        resume_document_id=resume.id,
        full_name="Candidate One",
        email="candidate@example.com",
        profile_status=ProfileStatus.REVIEWED,
    )
    db_session.add(candidate)
    db_session.flush()

    message = OutreachMessage(
        candidate_profile_id=candidate.id,
        created_by_user_id=user.id,
        content_source=ContentSource.AI_DRAFT,
        subject="Intro call",
        body="Would you be open to a short intro call?",
        sent_status=SentStatus.NOT_SENT,
    )
    db_session.add(message)
    db_session.commit()
    db_session.refresh(user)
    db_session.refresh(message)
    return {"user": user, "message": message}


@pytest.fixture()
def api_client(db_session: Session):
    def _override_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_db
    client = TestClient(app, follow_redirects=False)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture()
def authed_api_client(db_session: Session, seeded_outreach_message):
    def _override_db():
        yield db_session

    def _override_current_user():
        return seeded_outreach_message["user"]

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user] = _override_current_user
    client = TestClient(app, follow_redirects=False)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()


def test_send_outreach_requires_current_user(api_client, seeded_outreach_message):
    message = seeded_outreach_message["message"]
    response = api_client.post(f"/api/v1/outreach/{message.id}/send")

    assert response.status_code in {401, 403}


def test_send_outreach_queues_task_for_owner(authed_api_client, seeded_outreach_message, monkeypatch):
    import worker.tasks as tasks_module

    message = seeded_outreach_message["message"]
    queued = []

    class FakeTask:
        @staticmethod
        def delay(message_id):
            queued.append(message_id)

    monkeypatch.setattr(tasks_module, "send_outreach_email", FakeTask, raising=False)

    response = authed_api_client.post(f"/api/v1/outreach/{message.id}/send")

    assert response.status_code == 202
    assert response.json()["sent_status"] == SentStatus.NOT_SENT.value
    assert queued == [str(message.id)]
