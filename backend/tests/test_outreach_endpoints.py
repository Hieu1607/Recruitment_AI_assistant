import sys
from datetime import datetime, timezone
from pathlib import Path
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.v1.endpoints import outreach as outreach_module  # noqa: E402
from src.api.v1.endpoints.outreach import (  # noqa: E402
    OutreachCreateRequest,
    OutreachUpdateRequest,
    create_message,
    delete_message,
    get_message,
    list_messages,
    update_message,
)
from src.models.base import Base  # noqa: E402
from src.models.candidate_profile import CandidateProfile  # noqa: E402
from src.models.enums import ContentSource, ProfileStatus, SentStatus, UploadStatus, UserStatus  # noqa: E402
from src.models.job import Job  # noqa: E402
from src.models.resume_document import ResumeDocument  # noqa: E402
from src.models.user_account import UserAccount  # noqa: E402


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
def db_session_factory(monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    _create_test_tables(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    monkeypatch.setattr(outreach_module, "SessionLocal", factory)
    return factory


@pytest.fixture()
def seeded_data(db_session_factory):
    db: Session = db_session_factory()
    try:
        user = UserAccount(
            email="owner@example.com",
            display_name="Owner",
            password_hash=None,
            status=UserStatus.ACTIVE,
        )
        db.add(user)
        db.flush()

        job = Job(owner_user_id=user.id, title="Platform Engineer", status="active")
        db.add(job)
        db.flush()

        resume = ResumeDocument(
            original_file_name="candidate.pdf",
            storage_uri="s3://bucket/resumes/candidate.pdf",
            upload_status=UploadStatus.PROCESSED,
            job_id=job.id,
            uploaded_by_user_id=user.id,
            retention_expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
        )
        db.add(resume)
        db.flush()

        candidate = CandidateProfile(
            resume_document_id=resume.id,
            full_name="Candidate One",
            email="candidate@example.com",
            profile_status=ProfileStatus.REVIEWED,
        )
        db.add(candidate)
        db.commit()
        db.refresh(user)
        db.refresh(candidate)
        return {"user": user, "candidate": candidate}
    finally:
        db.close()


def test_outreach_create_list_update_and_delete(db_session_factory, seeded_data):
    user = seeded_data["user"]
    candidate = seeded_data["candidate"]

    created = create_message(
        OutreachCreateRequest(
            candidate_profile_id=candidate.id,
            created_by_user_id=user.id,
            content_source=ContentSource.AI_DRAFT,
            subject="Intro call",
            body="Would you be open to a short intro call?",
        )
    )

    listed = list_messages(
        created_by_user_id=user.id,
        candidate_profile_id=candidate.id,
        sent_status=SentStatus.NOT_SENT,
        offset=0,
        limit=50,
    )
    fetched = get_message(uuid.UUID(created.id))
    updated = update_message(
        uuid.UUID(created.id),
        OutreachUpdateRequest(sent_status=SentStatus.SENT),
    )

    assert created.sent_status == SentStatus.NOT_SENT.value
    assert created.candidate_full_name == "Candidate One"
    assert listed.total == 1
    assert listed.items[0].id == created.id
    assert fetched.subject == "Intro call"
    assert updated.sent_status == SentStatus.SENT.value
    assert updated.sent_at is not None

    delete_message(uuid.UUID(created.id))

    after_delete = list_messages(
        created_by_user_id=user.id,
        candidate_profile_id=None,
        sent_status=None,
        offset=0,
        limit=50,
    )
    assert after_delete.total == 0
