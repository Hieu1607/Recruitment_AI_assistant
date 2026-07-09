import sys
from pathlib import Path
import uuid
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.v1.endpoints.jobs import (  # noqa: E402
    ChatRequest,
    ChatSessionUpdateRequest,
    _load_job_candidates,
    chat_about_job,
    delete_job_chat_session,
    get_job_setup_status,
    list_job_chat_sessions,
    list_job_chat_turns,
    update_job_chat_session,
)
from src.models.base import Base  # noqa: E402
from src.models.candidate_profile import CandidateProfile  # noqa: E402
from src.models.enums import MatchRunStatus, ProfileStatus, UploadStatus, UserStatus  # noqa: E402
from src.models.job import Job  # noqa: E402
from src.models.job_matching import JobDescription, MatchRun  # noqa: E402
from src.models.query_shortlist import QuerySession, QueryTurn  # noqa: E402
from src.models.resume_document import ResumeDocument  # noqa: E402
from src.models.user_account import UserAccount  # noqa: E402


def _create_test_tables(engine):
    tables = [
        Base.metadata.tables["user_accounts"],
        Base.metadata.tables["jobs"],
        Base.metadata.tables["resume_documents"],
        Base.metadata.tables["candidate_profiles"],
        Base.metadata.tables["job_descriptions"],
        Base.metadata.tables["match_runs"],
        Base.metadata.tables["query_sessions"],
        Base.metadata.tables["query_turns"],
        Base.metadata.tables["shortlist_collections"],
        Base.metadata.tables["shortlist_items"],
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
            assert [message.content for message in payload["messages"]] == [
                "Who is the strongest backend candidate?"
            ]
            assert payload["current_job"] == {
                "job_id": str(owned_job_with_candidate.id),
                "job_title": "Platform Engineer",
                "job_description_id": str(jd.id),
                "job_description_title": "AI Platform",
                "job_description_text": "Build LLM applications and ranking workflows.",
                "job_hidden_text": "Prefer production ML deployment experience.",
            }
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

    stored_session = db.get(QuerySession, uuid.UUID(response.session_id))
    assert stored_session is not None
    assert stored_session.job_id == owned_job_with_candidate.id
    assert stored_session.user_id == owner.id
    assert stored_session.session_title == "Who is the strongest backend candidate?"

    turns = db.query(QueryTurn).filter(QueryTurn.query_session_id == stored_session.id).all()
    assert len(turns) == 1
    assert turns[0].user_question == "Who is the strongest backend candidate?"
    assert turns[0].answer_text == "Candidate One looks strong for backend work."
    assert turns[0].matched_count == 1


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

    assert response.answer == "There is 1 candidate in this job."
    assert response.candidates_in_scope == 1
    assert response.session_id

    turn = (
        db.query(QueryTurn)
        .filter(QueryTurn.query_session_id == uuid.UUID(response.session_id))
        .one()
    )
    assert turn.user_question == "How many candidates are in this job?"
    assert turn.answer_text == "There is 1 candidate in this job."
    assert turn.matched_count == 1


def test_chat_about_job_answers_total_count_in_vietnamese_for_vietnamese_question(
    monkeypatch, db, owner, owned_job_with_candidate
):
    class ExplodingGraph:
        def invoke(self, payload):
            raise AssertionError("graph should not run for total candidate count questions")

    monkeypatch.setattr("src.api.v1.endpoints.jobs.get_graph", lambda: ExplodingGraph())

    response = chat_about_job(
        job_id=owned_job_with_candidate.id,
        body=ChatRequest(message="Có bao nhiêu ứng viên trong job này?"),
        db=db,
        current_user=owner,
    )

    assert response.answer == "Có 1 ứng viên trong job này."


def test_load_job_candidates_normalizes_legacy_hanoi_locations(db, owner):
    job = Job(owner_user_id=owner.id, title="AI Engineer", status="active")
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
        full_name="Candidate Hanoi",
        location_normalized="Hanoi",
        profile_status=ProfileStatus.REVIEWED,
    )
    db.add(profile)
    db.commit()

    candidates = _load_job_candidates(db, job.id, 10)

    assert candidates[0]["location_normalized"] == "Hà Nội"


def test_chat_about_job_rebuilds_history_from_saved_turns(
    monkeypatch, db, owner, owned_job_with_candidate
):
    seen_histories = []

    class FakeGraph:
        def invoke(self, payload):
            seen_histories.append([message.content for message in payload["messages"]])
            return {
                "messages": payload["messages"],
                "answer": f"answer {len(seen_histories)}",
                "dsl_candidates": None,
            }

    monkeypatch.setattr("src.api.v1.endpoints.jobs.get_graph", lambda: FakeGraph())

    first = chat_about_job(
        job_id=owned_job_with_candidate.id,
        body=ChatRequest(message="First question"),
        db=db,
        current_user=owner,
    )
    second = chat_about_job(
        job_id=owned_job_with_candidate.id,
        body=ChatRequest(message="Second question", session_id=first.session_id),
        db=db,
        current_user=owner,
    )

    assert second.session_id == first.session_id
    assert seen_histories[1] == ["First question", "answer 1", "Second question"]
    turns = (
        db.query(QueryTurn)
        .filter(QueryTurn.query_session_id == uuid.UUID(first.session_id))
        .order_by(QueryTurn.created_at.asc())
        .all()
    )
    assert [turn.user_question for turn in turns] == ["First question", "Second question"]


def test_job_chat_session_crud_is_scoped_to_owner_and_job(
    monkeypatch, db, owner, outsider, owned_job_with_candidate
):
    class FakeGraph:
        def invoke(self, payload):
            return {
                "messages": payload["messages"],
                "answer": "Stored answer",
                "dsl_candidates": None,
            }

    monkeypatch.setattr("src.api.v1.endpoints.jobs.get_graph", lambda: FakeGraph())
    response = chat_about_job(
        job_id=owned_job_with_candidate.id,
        body=ChatRequest(message="Persist this conversation"),
        db=db,
        current_user=owner,
    )

    listed = list_job_chat_sessions(
        job_id=owned_job_with_candidate.id,
        offset=0,
        limit=50,
        db=db,
        current_user=owner,
    )
    assert listed.total == 1
    assert listed.items[0].id == response.session_id

    renamed = update_job_chat_session(
        job_id=owned_job_with_candidate.id,
        session_id=uuid.UUID(response.session_id),
        body=ChatSessionUpdateRequest(session_title="Renamed"),
        db=db,
        current_user=owner,
    )
    assert renamed.session_title == "Renamed"

    turns = list_job_chat_turns(
        job_id=owned_job_with_candidate.id,
        session_id=uuid.UUID(response.session_id),
        offset=0,
        limit=50,
        db=db,
        current_user=owner,
    )
    assert len(turns) == 1
    assert turns[0].answer_text == "Stored answer"

    with pytest.raises(HTTPException) as exc_info:
        list_job_chat_sessions(
            job_id=owned_job_with_candidate.id,
            offset=0,
            limit=50,
            db=db,
            current_user=outsider,
        )
    assert exc_info.value.status_code == 404

    delete_job_chat_session(
        job_id=owned_job_with_candidate.id,
        session_id=uuid.UUID(response.session_id),
        db=db,
        current_user=owner,
    )
    assert db.get(QuerySession, uuid.UUID(response.session_id)) is None
    assert db.query(QueryTurn).count() == 0


def test_setup_status_reports_database_backed_progress(
    monkeypatch, db, owner, owned_job_with_candidate
):
    jd = JobDescription(
        title="Platform Engineer",
        jd_text="Build APIs",
        hidden_text="Python",
        job_id=owned_job_with_candidate.id,
        created_by_user_id=owner.id,
        is_active=True,
    )
    db.add(jd)
    db.flush()
    db.add(
        MatchRun(
            job_description_id=jd.id,
            score_threshold=50,
            initiated_by_user_id=owner.id,
            run_status=MatchRunStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc),
        )
    )
    db.commit()

    class FakeGraph:
        def invoke(self, payload):
            return {
                "messages": payload["messages"],
                "answer": "Stored answer",
                "dsl_candidates": None,
            }

    monkeypatch.setattr("src.api.v1.endpoints.jobs.get_graph", lambda: FakeGraph())
    chat_about_job(
        job_id=owned_job_with_candidate.id,
        body=ChatRequest(message="Who should I call?"),
        db=db,
        current_user=owner,
    )

    status = get_job_setup_status(
        job_id=owned_job_with_candidate.id,
        db=db,
        current_user=owner,
    )

    assert status.job_id == str(owned_job_with_candidate.id)
    assert status.resume_count == 1
    assert status.processed_candidate_count == 1
    assert status.has_uploaded_resumes is True
    assert status.has_processed_candidates is True
    assert status.has_active_job_description is True
    assert status.has_completed_score_run is True
    assert status.has_chat_turn is True
    assert status.completed_score_run_count == 1
    assert status.chat_session_count == 1
    assert status.chat_turn_count == 1


def test_chat_about_job_rejects_non_owner(db, outsider, owned_job_with_candidate):
    with pytest.raises(HTTPException) as exc_info:
        chat_about_job(
            job_id=owned_job_with_candidate.id,
            body=ChatRequest(message="Can I see these candidates?"),
            db=db,
            current_user=outsider,
        )

    assert exc_info.value.status_code == 404
