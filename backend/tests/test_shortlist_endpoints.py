import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

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
from src.main import app  # noqa: E402
from src.models.base import Base  # noqa: E402
from src.models.candidate_profile import CandidateProfile  # noqa: E402
from src.models.deps import get_current_user, get_db  # noqa: E402
from src.models.enums import ContentSource, ProfileStatus, SentStatus, UploadStatus, UserStatus  # noqa: E402
from src.models.interview_invitation import InterviewInvitation  # noqa: E402
from src.models.interview_template import InterviewTemplate  # noqa: E402
from src.models.job import Job  # noqa: E402
from src.models.job_matching import InterviewQuestionSet, JobDescription  # noqa: E402
from src.models.oauth_identity import GMAIL_SEND_SCOPE, OAuthIdentity  # noqa: E402
from src.models.outreach import OutreachMessage  # noqa: E402
from src.models.query_shortlist import QuerySession, QueryTurn, ShortlistCollection, ShortlistItem  # noqa: E402
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
        Base.metadata.tables["interview_question_sets"],
        Base.metadata.tables["oauth_identities"],
        Base.metadata.tables["outreach_messages"],
        Base.metadata.tables["outreach_templates"],
        Base.metadata.tables["interview_templates"],
        Base.metadata.tables["interview_invitations"],
        Base.metadata.tables["query_sessions"],
        Base.metadata.tables["query_turns"],
        Base.metadata.tables["shortlist_collections"],
        Base.metadata.tables["shortlist_items"],
    ]
    Base.metadata.create_all(engine, tables=tables)


def _persist_session(db: Session, user: UserAccount, title: str) -> QuerySession:
    session = QuerySession(user_id=user.id, session_title=title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def _persist_turn(
    db: Session,
    session: QuerySession,
    *,
    user_question: str = "Who matches backend best?",
    answer_text: str = "Candidate One",
    matched_candidate_ids: list[str] | None = None,
    matched_count: int | None = None,
) -> QueryTurn:
    turn = QueryTurn(
        query_session_id=session.id,
        user_question=user_question,
        answer_text=answer_text,
        matched_candidate_ids=matched_candidate_ids,
        matched_count=matched_count,
    )
    db.add(turn)
    db.commit()
    db.refresh(turn)
    return turn


def _persist_collection(
    db: Session,
    user: UserAccount,
    name: str,
    *,
    source_query_turn_id: uuid.UUID | None = None,
) -> ShortlistCollection:
    collection = ShortlistCollection(
        name=name,
        created_by_user_id=user.id,
        source_query_turn_id=source_query_turn_id,
    )
    db.add(collection)
    db.commit()
    db.refresh(collection)
    return collection


def _persist_item(
    db: Session,
    collection: ShortlistCollection,
    candidate: CandidateProfile,
) -> ShortlistItem:
    item = ShortlistItem(
        shortlist_collection_id=collection.id,
        candidate_profile_id=candidate.id,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    _create_test_tables(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    with factory() as session:
        yield session


@pytest.fixture()
def seeded_data(db_session: Session):
    owner = UserAccount(
        email="owner@example.com",
        display_name="Owner",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    other_user = UserAccount(
        email="other@example.com",
        display_name="Other",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    db_session.add_all([owner, other_user])
    db_session.flush()

    job = Job(owner_user_id=owner.id, title="Platform Engineer", status="active")
    db_session.add(job)
    db_session.flush()

    oauth_identity = OAuthIdentity(
        user_id=owner.id,
        provider="google",
        provider_subject="google-owner",
        email=owner.email,
        refresh_token_encrypted="encrypted-refresh-token",
        scope=f"openid email profile {GMAIL_SEND_SCOPE}",
    )
    db_session.add(oauth_identity)
    db_session.flush()

    resume = ResumeDocument(
        original_file_name="candidate.pdf",
        storage_uri="s3://bucket/resumes/candidate.pdf",
        upload_status=UploadStatus.PROCESSED,
        job_id=job.id,
        uploaded_by_user_id=owner.id,
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

    jd = JobDescription(
        job_id=job.id,
        title="Platform Engineer JD",
        jd_text="Need backend systems design and communication skills.",
        created_by_user_id=owner.id,
        is_active=True,
    )
    db_session.add(jd)
    db_session.flush()

    question_set = InterviewQuestionSet(
        candidate_profile_id=candidate.id,
        job_description_id=jd.id,
        generated_by_user_id=owner.id,
        question_payload={"questions": [{"key": "q1", "prompt": "Tell us about yourself"}]},
    )
    db_session.add(question_set)
    db_session.flush()

    missing_email_resume = ResumeDocument(
        original_file_name="missing-email.pdf",
        storage_uri="s3://bucket/resumes/missing-email.pdf",
        upload_status=UploadStatus.PROCESSED,
        job_id=job.id,
        uploaded_by_user_id=owner.id,
        retention_expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
    )
    db_session.add(missing_email_resume)
    db_session.flush()

    missing_email_candidate = CandidateProfile(
        resume_document_id=missing_email_resume.id,
        full_name="Candidate Missing Email",
        email=None,
        profile_status=ProfileStatus.REVIEWED,
    )
    db_session.add(missing_email_candidate)
    db_session.commit()
    db_session.refresh(owner)
    db_session.refresh(other_user)
    db_session.refresh(job)
    db_session.refresh(candidate)
    db_session.refresh(missing_email_candidate)

    return {
        "owner": owner,
        "other_user": other_user,
        "job": job,
        "candidate": candidate,
        "missing_email_candidate": missing_email_candidate,
        "job_description": jd,
        "question_set_id": question_set.id,
    }


@pytest.fixture()
def client_factory(db_session: Session):
    @contextmanager
    def _factory(current_user: UserAccount | None = None):
        def _override_db():
            yield db_session

        app.dependency_overrides[get_db] = _override_db
        if current_user is not None:
            app.dependency_overrides[get_current_user] = lambda: current_user

        client = TestClient(app, follow_redirects=False)
        try:
            yield client
        finally:
            app.dependency_overrides.clear()

    return _factory


def test_shortlist_endpoints_require_auth(client_factory):
    with client_factory() as client:
        response = client.get("/api/v1/shortlist/collections/")

    assert response.status_code in {401, 403}


def test_shortlist_session_crud_uses_current_user(db_session: Session, seeded_data):
    owner = seeded_data["owner"]

    created = create_session(
        SessionCreateRequest(session_title="Top backend talent"),
        db=db_session,
        current_user=owner,
    )
    listed = list_sessions(offset=0, limit=50, db=db_session, current_user=owner)
    updated = update_session(
        uuid.UUID(created.id),
        SessionUpdateRequest(session_title="Renamed session"),
        db=db_session,
        current_user=owner,
    )

    assert created.user_id == str(owner.id)
    assert listed.total == 1
    assert listed.items[0].session_title == "Top backend talent"
    assert updated.session_title == "Renamed session"

    delete_session(uuid.UUID(created.id), db=db_session, current_user=owner)

    assert db_session.get(QuerySession, uuid.UUID(created.id)) is None


def test_shortlist_collection_duplicate_and_items_flow(
    db_session: Session,
    seeded_data,
):
    owner = seeded_data["owner"]
    candidate = seeded_data["candidate"]

    session = create_session(
        SessionCreateRequest(session_title="Session A"),
        db=db_session,
        current_user=owner,
    )
    turn = create_turn(
        uuid.UUID(session.id),
        TurnCreateRequest(
            user_question="Who matches backend best?",
            answer_text="Candidate One",
            matched_candidate_ids=[str(candidate.id)],
            matched_count=1,
        ),
        db=db_session,
        current_user=owner,
    )

    collection = create_collection(
        CollectionCreateRequest(
            name="Priority shortlist",
            source_query_turn_id=uuid.UUID(turn.id),
        ),
        db=db_session,
        current_user=owner,
    )

    with pytest.raises(HTTPException) as exc_info:
        create_collection(
            CollectionCreateRequest(name="Priority shortlist"),
            db=db_session,
            current_user=owner,
        )

    assert exc_info.value.status_code == 409

    item = add_item(
        uuid.UUID(collection.id),
        ItemAddRequest(candidate_profile_id=candidate.id),
        db=db_session,
        current_user=owner,
    )
    listed_items = list_items(
        uuid.UUID(collection.id),
        offset=0,
        limit=100,
        db=db_session,
        current_user=owner,
    )
    fetched_collection = get_collection(
        uuid.UUID(collection.id),
        db=db_session,
        current_user=owner,
    )
    renamed = update_collection(
        uuid.UUID(collection.id),
        CollectionUpdateRequest(name="Renamed shortlist"),
        db=db_session,
        current_user=owner,
    )

    assert item.candidate_profile_id == str(candidate.id)
    assert listed_items.total == 1
    assert fetched_collection.item_count == 1
    assert renamed.name == "Renamed shortlist"

    with pytest.raises(HTTPException) as dup_item:
        add_item(
            uuid.UUID(collection.id),
            ItemAddRequest(candidate_profile_id=candidate.id),
            db=db_session,
            current_user=owner,
        )

    assert dup_item.value.status_code == 409

    delete_collection(uuid.UUID(collection.id), db=db_session, current_user=owner)

    assert db_session.get(QueryTurn, uuid.UUID(turn.id)) is not None


def test_shortlist_create_flow_does_not_require_client_ownership_fields(
    client_factory,
    seeded_data,
):
    owner = seeded_data["owner"]

    with client_factory(owner) as client:
        session_response = client.post(
            "/api/v1/shortlist/sessions/",
            json={"session_title": "Top backend talent"},
        )
        collection_response = client.post(
            "/api/v1/shortlist/collections/",
            json={"name": "Priority shortlist"},
        )

    assert session_response.status_code == 201
    assert session_response.json()["user_id"] == str(owner.id)
    assert collection_response.status_code == 201
    assert collection_response.json()["created_by_user_id"] == str(owner.id)


def test_shortlist_list_apis_only_return_current_user_records(
    client_factory,
    seeded_data,
):
    owner = seeded_data["owner"]
    other_user = seeded_data["other_user"]

    with client_factory(owner) as owner_client:
        owner_session = owner_client.post(
            "/api/v1/shortlist/sessions/",
            json={"session_title": "Owner session"},
        )
        owner_collection = owner_client.post(
            "/api/v1/shortlist/collections/",
            json={"name": "Owner shortlist"},
        )
        assert owner_session.status_code == 201
        assert owner_collection.status_code == 201

    with client_factory(other_user) as other_client:
        other_session = other_client.post(
            "/api/v1/shortlist/sessions/",
            json={"session_title": "Other session"},
        )
        other_collection = other_client.post(
            "/api/v1/shortlist/collections/",
            json={"name": "Other shortlist"},
        )
        assert other_session.status_code == 201
        assert other_collection.status_code == 201

        other_sessions_list = other_client.get("/api/v1/shortlist/sessions/?limit=50")
        other_collections_list = other_client.get("/api/v1/shortlist/collections/?limit=50")

    with client_factory(owner) as owner_client:
        owner_sessions_list = owner_client.get("/api/v1/shortlist/sessions/?limit=50")
        owner_collections_list = owner_client.get("/api/v1/shortlist/collections/?limit=50")

    assert owner_sessions_list.status_code == 200
    assert owner_collections_list.status_code == 200
    assert owner_sessions_list.json()["total"] == 1
    assert owner_collections_list.json()["total"] == 1
    assert [item["session_title"] for item in owner_sessions_list.json()["items"]] == ["Owner session"]
    assert [item["name"] for item in owner_collections_list.json()["items"]] == ["Owner shortlist"]

    assert other_sessions_list.status_code == 200
    assert other_collections_list.status_code == 200
    assert other_sessions_list.json()["total"] == 1
    assert other_collections_list.json()["total"] == 1
    assert [item["session_title"] for item in other_sessions_list.json()["items"]] == ["Other session"]
    assert [item["name"] for item in other_collections_list.json()["items"]] == ["Other shortlist"]


def test_shortlist_session_and_turn_endpoints_hide_other_users_resources(
    db_session: Session,
    client_factory,
    seeded_data,
):
    owner = seeded_data["owner"]
    other_user = seeded_data["other_user"]

    other_session = _persist_session(db_session, other_user, "Other session")
    other_turn = _persist_turn(db_session, other_session)

    with client_factory(owner) as client:
        assert client.get(f"/api/v1/shortlist/sessions/{other_session.id}").status_code == 404
        assert client.patch(
            f"/api/v1/shortlist/sessions/{other_session.id}",
            json={"session_title": "Renamed by attacker"},
        ).status_code == 404
        assert client.delete(f"/api/v1/shortlist/sessions/{other_session.id}").status_code == 404
        assert client.post(
            f"/api/v1/shortlist/sessions/{other_session.id}/turns",
            json={
                "user_question": "Hijack turn",
                "answer_text": "Nope",
            },
        ).status_code == 404
        assert client.get(
            f"/api/v1/shortlist/sessions/{other_session.id}/turns?limit=50"
        ).status_code == 404
        assert client.get(f"/api/v1/shortlist/turns/{other_turn.id}").status_code == 404
        assert client.delete(f"/api/v1/shortlist/turns/{other_turn.id}").status_code == 404


def test_shortlist_collection_endpoints_hide_other_users_resources(
    db_session: Session,
    client_factory,
    seeded_data,
):
    owner = seeded_data["owner"]
    other_user = seeded_data["other_user"]
    candidate = seeded_data["candidate"]
    job = seeded_data["job"]
    question_set_id = seeded_data["question_set_id"]

    other_collection = _persist_collection(db_session, other_user, "Other shortlist")
    _persist_item(db_session, other_collection, candidate)

    with client_factory(owner) as client:
        assert client.get(f"/api/v1/shortlist/collections/{other_collection.id}").status_code == 404
        assert client.patch(
            f"/api/v1/shortlist/collections/{other_collection.id}",
            json={"name": "Renamed by attacker"},
        ).status_code == 404
        assert client.delete(f"/api/v1/shortlist/collections/{other_collection.id}").status_code == 404
        assert client.get(
            f"/api/v1/shortlist/collections/{other_collection.id}/dispatch-summary"
        ).status_code == 404
        assert client.post(
            f"/api/v1/shortlist/collections/{other_collection.id}/items",
            json={"candidate_profile_id": str(candidate.id)},
        ).status_code == 404
        assert client.get(
            f"/api/v1/shortlist/collections/{other_collection.id}/items?limit=100"
        ).status_code == 404
        assert client.delete(
            f"/api/v1/shortlist/collections/{other_collection.id}/items/{candidate.id}"
        ).status_code == 404
        assert client.post(
            f"/api/v1/shortlist/collections/{other_collection.id}/outreach-drafts",
            json={
                "candidate_profile_ids": [str(candidate.id)],
                "subject_template": "Invitation for {{candidate_name}}",
                "body_text_template": "Hi {{candidate_name}}, let's talk.",
                "body_html_template": "<p>Hi <strong>{{candidate_name}}</strong>, let's talk.</p>",
            },
        ).status_code == 404
        assert client.post(
            f"/api/v1/shortlist/collections/{other_collection.id}/interview-invitations",
            json={
                "candidate_profile_ids": [str(candidate.id)],
                "job_id": str(job.id),
                "interview_question_set_id": str(question_set_id),
                "send_email": False,
            },
        ).status_code == 404


def test_shortlist_collection_source_turn_must_be_owned_by_current_user(
    db_session: Session,
    client_factory,
    seeded_data,
):
    owner = seeded_data["owner"]
    other_user = seeded_data["other_user"]

    other_session = _persist_session(db_session, other_user, "Other session")
    other_turn = _persist_turn(db_session, other_session)

    with client_factory(owner) as client:
        response = client.post(
            "/api/v1/shortlist/collections/",
            json={
                "name": "Hijacked shortlist",
                "source_query_turn_id": str(other_turn.id),
            },
        )

    assert response.status_code == 404


def test_dispatch_summary_includes_candidate_status_and_blockers(
    db_session: Session,
    seeded_data,
):
    owner = seeded_data["owner"]
    job = seeded_data["job"]
    candidate = seeded_data["candidate"]
    missing_email_candidate = seeded_data["missing_email_candidate"]

    collection = create_collection(
        CollectionCreateRequest(name="Dispatch shortlist"),
        db=db_session,
        current_user=owner,
    )
    add_item(
        uuid.UUID(collection.id),
        ItemAddRequest(candidate_profile_id=candidate.id),
        db=db_session,
        current_user=owner,
    )
    add_item(
        uuid.UUID(collection.id),
        ItemAddRequest(candidate_profile_id=missing_email_candidate.id),
        db=db_session,
        current_user=owner,
    )

    template = InterviewTemplate(
        job_id=job.id,
        name="Screening",
        status="active",
        language_code="vi-VN",
        question_payload={"questions": [{"prompt": "Tell us about yourself"}]},
        report_rubric={},
    )
    db_session.add(template)
    db_session.flush()
    outreach = OutreachMessage(
        candidate_profile_id=candidate.id,
        created_by_user_id=owner.id,
        content_source=ContentSource.TEMPLATE,
        subject="Intro",
        body_text="Hello",
        body_html="<p>Hello</p>",
        sent_status=SentStatus.SENT,
        sent_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    invitation = InterviewInvitation(
        job_id=job.id,
        candidate_profile_id=candidate.id,
        interview_template_id=template.id,
        sent_by_user_id=owner.id,
    )
    db_session.add_all([outreach, invitation])
    db_session.commit()

    summary = shortlist_module.get_dispatch_summary(
        uuid.UUID(collection.id),
        db=db_session,
        current_user=owner,
    )

    assert summary.collection.id == collection.id
    assert summary.job is not None
    assert summary.job.id == str(job.id)
    assert summary.capabilities.active_interview_templates_count == 1
    assert [item.full_name for item in summary.candidates] == [
        "Candidate One",
        "Candidate Missing Email",
    ]
    first = summary.candidates[0]
    second = summary.candidates[1]
    assert first.outreach is not None
    assert first.outreach.status == "sent"
    assert first.interview is not None
    assert first.interview.status == "pending"
    assert first.blockers == []
    assert second.contact_status == "missing_email"
    assert "missing_email" in second.blockers


def test_create_outreach_drafts_skips_duplicates_and_missing_email(
    db_session: Session,
    seeded_data,
):
    owner = seeded_data["owner"]
    candidate = seeded_data["candidate"]
    missing_email_candidate = seeded_data["missing_email_candidate"]
    collection = create_collection(
        CollectionCreateRequest(name="Draft shortlist"),
        db=db_session,
        current_user=owner,
    )
    add_item(
        uuid.UUID(collection.id),
        ItemAddRequest(candidate_profile_id=candidate.id),
        db=db_session,
        current_user=owner,
    )
    add_item(
        uuid.UUID(collection.id),
        ItemAddRequest(candidate_profile_id=missing_email_candidate.id),
        db=db_session,
        current_user=owner,
    )

    result = shortlist_module.create_collection_outreach_drafts(
        uuid.UUID(collection.id),
        shortlist_module.OutreachDraftBatchRequest(
            candidate_profile_ids=[candidate.id, missing_email_candidate.id],
            subject_template="Invitation for {{candidate_name}}",
            body_text_template="Hi {{candidate_name}}, let's talk.",
            body_html_template="<p>Hi <strong>{{candidate_name}}</strong>, let's talk.</p>",
        ),
        db=db_session,
        current_user=owner,
    )

    assert result.created_count == 1
    assert result.skipped_count == 1
    assert result.results[0].status == "created"
    assert result.results[1].status == "skipped_missing_email"

    created = (
        db_session.query(OutreachMessage)
        .filter(OutreachMessage.candidate_profile_id == candidate.id)
        .first()
    )
    assert created is not None
    assert created.body_text == "Hi Candidate One, let's talk."
    assert "<p>" in created.body_html

    duplicate_result = shortlist_module.create_collection_outreach_drafts(
        uuid.UUID(collection.id),
        shortlist_module.OutreachDraftBatchRequest(
            candidate_profile_ids=[candidate.id],
            subject_template="Second {{candidate_name}}",
            body_text_template="Second body",
            body_html_template="<p>Second body</p>",
        ),
        db=db_session,
        current_user=owner,
    )

    assert duplicate_result.created_count == 0
    assert duplicate_result.results[0].status == "skipped_duplicate"


def test_create_interview_invitations_skips_duplicates_and_missing_email(
    db_session: Session,
    seeded_data,
):
    owner = seeded_data["owner"]
    job = seeded_data["job"]
    candidate = seeded_data["candidate"]
    missing_email_candidate = seeded_data["missing_email_candidate"]
    collection = create_collection(
        CollectionCreateRequest(name="Interview shortlist"),
        db=db_session,
        current_user=owner,
    )
    add_item(
        uuid.UUID(collection.id),
        ItemAddRequest(candidate_profile_id=candidate.id),
        db=db_session,
        current_user=owner,
    )
    add_item(
        uuid.UUID(collection.id),
        ItemAddRequest(candidate_profile_id=missing_email_candidate.id),
        db=db_session,
        current_user=owner,
    )

    template = InterviewTemplate(
        job_id=job.id,
        name="Screening",
        status="active",
        language_code="vi-VN",
        question_payload={"questions": [{"prompt": "Tell us about yourself"}]},
        report_rubric={},
    )
    db_session.add(template)
    db_session.commit()
    db_session.refresh(template)

    result = shortlist_module.create_collection_interview_invitations(
        uuid.UUID(collection.id),
        shortlist_module.InterviewInvitationBatchRequest(
            candidate_profile_ids=[candidate.id, missing_email_candidate.id],
            job_id=job.id,
            interview_template_id=template.id,
            expires_in_hours=72,
        ),
        db=db_session,
        current_user=owner,
    )

    assert result.created_count == 1
    assert result.skipped_count == 1
    assert result.results[0].status == "created"
    assert result.results[1].status == "skipped_missing_email"

    duplicate_result = shortlist_module.create_collection_interview_invitations(
        uuid.UUID(collection.id),
        shortlist_module.InterviewInvitationBatchRequest(
            candidate_profile_ids=[candidate.id],
            job_id=job.id,
            interview_template_id=template.id,
        ),
        db=db_session,
        current_user=owner,
    )

    assert duplicate_result.created_count == 0
    assert duplicate_result.results[0].status == "skipped_duplicate"


def test_create_interview_invitations_can_materialize_question_set_without_sending_email(
    db_session: Session,
    seeded_data,
):
    owner = seeded_data["owner"]
    job = seeded_data["job"]
    candidate = seeded_data["candidate"]
    missing_email_candidate = seeded_data["missing_email_candidate"]
    question_set_id = seeded_data["question_set_id"]
    collection = create_collection(
        CollectionCreateRequest(name="Question set shortlist"),
        db=db_session,
        current_user=owner,
    )
    add_item(
        uuid.UUID(collection.id),
        ItemAddRequest(candidate_profile_id=candidate.id),
        db=db_session,
        current_user=owner,
    )
    add_item(
        uuid.UUID(collection.id),
        ItemAddRequest(candidate_profile_id=missing_email_candidate.id),
        db=db_session,
        current_user=owner,
    )

    result = shortlist_module.create_collection_interview_invitations(
        uuid.UUID(collection.id),
        shortlist_module.InterviewInvitationBatchRequest(
            candidate_profile_ids=[candidate.id, missing_email_candidate.id],
            job_id=job.id,
            interview_question_set_id=question_set_id,
            send_email=False,
        ),
        db=db_session,
        current_user=owner,
    )

    assert result.created_count == 2
    assert result.skipped_count == 0
    assert all(item.status == "created" for item in result.results)

    invitations = (
        db_session.query(InterviewInvitation)
        .order_by(InterviewInvitation.created_at.asc())
        .all()
    )
    assert len(invitations) == 2
    assert invitations[0].sent_at is None
    assert invitations[0].interview_template_id == invitations[1].interview_template_id

    template = db_session.get(InterviewTemplate, invitations[0].interview_template_id)
    assert template is not None
    assert template.job_id == job.id
    assert template.question_payload["questions"][0]["key"] == "q1"
