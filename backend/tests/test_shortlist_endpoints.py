import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.v1.endpoints import shortlist as shortlist_module  # noqa: E402
from src.api.v1.endpoints.shortlist import (  # noqa: E402
    CollectionCreateRequest,
    CollectionUpdateRequest,
    ItemAddRequest,
    SessionCreateRequest,
    SessionUpdateRequest,
    TurnCreateRequest,
    add_item,
    create_collection,
    create_session,
    create_turn,
    delete_collection,
    delete_session,
    get_collection,
    list_items,
    list_sessions,
    update_collection,
    update_session,
)
from src.models.base import Base  # noqa: E402
from src.models.candidate_profile import CandidateProfile  # noqa: E402
from src.models.enums import ProfileStatus, UploadStatus, UserStatus  # noqa: E402
from src.models.job import Job  # noqa: E402
from src.models.query_shortlist import QuerySession, QueryTurn  # noqa: E402
from src.models.resume_document import ResumeDocument  # noqa: E402
from src.models.user_account import UserAccount  # noqa: E402


def _create_test_tables(engine):
    tables = [
        Base.metadata.tables["user_accounts"],
        Base.metadata.tables["jobs"],
        Base.metadata.tables["resume_documents"],
        Base.metadata.tables["candidate_profiles"],
        Base.metadata.tables["query_sessions"],
        Base.metadata.tables["query_turns"],
        Base.metadata.tables["shortlist_collections"],
        Base.metadata.tables["shortlist_items"],
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
    monkeypatch.setattr(shortlist_module, "SessionLocal", factory)
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


def test_shortlist_session_crud(db_session_factory, seeded_data):
    user = seeded_data["user"]

    created = create_session(
        SessionCreateRequest(user_id=user.id, session_title="Top backend talent")
    )
    listed = list_sessions(user_id=user.id, offset=0, limit=50)
    updated = update_session(
        uuid.UUID(created.id),
        SessionUpdateRequest(session_title="Renamed session"),
    )

    assert created.user_id == str(user.id)
    assert listed.total == 1
    assert listed.items[0].session_title == "Top backend talent"
    assert updated.session_title == "Renamed session"

    delete_session(uuid.UUID(created.id))

    with db_session_factory() as db:
        assert db.get(QuerySession, uuid.UUID(created.id)) is None


def test_shortlist_collection_duplicate_and_items_flow(
    db_session_factory, seeded_data
):
    user = seeded_data["user"]
    candidate = seeded_data["candidate"]

    session = create_session(
        SessionCreateRequest(user_id=user.id, session_title="Session A")
    )
    turn = create_turn(
        uuid.UUID(session.id),
        TurnCreateRequest(
            user_question="Who matches backend best?",
            answer_text="Candidate One",
            matched_candidate_ids=[str(candidate.id)],
            matched_count=1,
        ),
    )

    collection = create_collection(
        CollectionCreateRequest(
            created_by_user_id=user.id,
            name="Priority shortlist",
            source_query_turn_id=uuid.UUID(turn.id),
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        create_collection(
            CollectionCreateRequest(
                created_by_user_id=user.id,
                name="Priority shortlist",
            )
        )

    assert exc_info.value.status_code == 409

    item = add_item(
        uuid.UUID(collection.id),
        ItemAddRequest(candidate_profile_id=candidate.id),
    )
    listed_items = list_items(uuid.UUID(collection.id), offset=0, limit=100)
    fetched_collection = get_collection(uuid.UUID(collection.id))
    renamed = update_collection(
        uuid.UUID(collection.id),
        CollectionUpdateRequest(name="Renamed shortlist"),
    )

    assert item.candidate_profile_id == str(candidate.id)
    assert listed_items.total == 1
    assert fetched_collection.item_count == 1
    assert renamed.name == "Renamed shortlist"

    with pytest.raises(HTTPException) as dup_item:
        add_item(
            uuid.UUID(collection.id),
            ItemAddRequest(candidate_profile_id=candidate.id),
        )

    assert dup_item.value.status_code == 409

    delete_collection(uuid.UUID(collection.id))

    with db_session_factory() as db:
        assert db.get(QueryTurn, uuid.UUID(turn.id)) is not None
