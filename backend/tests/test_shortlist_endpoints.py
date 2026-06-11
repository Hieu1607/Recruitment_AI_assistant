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
from src.models.enums import ContentSource, ProfileStatus, SentStatus, UploadStatus, UserStatus  # noqa: E402
from src.models.interview_invitation import InterviewInvitation  # noqa: E402
from src.models.interview_template import InterviewTemplate  # noqa: E402
from src.models.job import Job  # noqa: E402
from src.models.job_matching import InterviewQuestionSet, JobDescription  # noqa: E402
from src.models.oauth_identity import GMAIL_SEND_SCOPE, OAuthIdentity  # noqa: E402
from src.models.outreach import OutreachMessage  # noqa: E402
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

        oauth_identity = OAuthIdentity(
            user_id=user.id,
            provider="google",
            provider_subject="google-owner",
            email=user.email,
            refresh_token_encrypted="encrypted-refresh-token",
            scope=f"openid email profile {GMAIL_SEND_SCOPE}",
        )
        db.add(oauth_identity)
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
        db.flush()

        jd = JobDescription(
            job_id=job.id,
            title="Platform Engineer JD",
            jd_text="Need backend systems design and communication skills.",
            created_by_user_id=user.id,
            is_active=True,
        )
        db.add(jd)
        db.flush()

        question_set = InterviewQuestionSet(
            candidate_profile_id=candidate.id,
            job_description_id=jd.id,
            generated_by_user_id=user.id,
            question_payload={"questions": [{"key": "q1", "prompt": "Tell us about yourself"}]},
        )
        db.add(question_set)
        db.flush()

        missing_email_resume = ResumeDocument(
            original_file_name="missing-email.pdf",
            storage_uri="s3://bucket/resumes/missing-email.pdf",
            upload_status=UploadStatus.PROCESSED,
            job_id=job.id,
            uploaded_by_user_id=user.id,
            retention_expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
        )
        db.add(missing_email_resume)
        db.flush()

        missing_email_candidate = CandidateProfile(
            resume_document_id=missing_email_resume.id,
            full_name="Candidate Missing Email",
            email=None,
            profile_status=ProfileStatus.REVIEWED,
        )
        db.add(missing_email_candidate)
        db.commit()
        db.refresh(user)
        db.refresh(job)
        db.refresh(candidate)
        db.refresh(missing_email_candidate)
        return {
            "user": user,
            "job": job,
            "candidate": candidate,
            "missing_email_candidate": missing_email_candidate,
            "job_description": jd,
            "question_set_id": question_set.id,
        }
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


def test_dispatch_summary_includes_candidate_status_and_blockers(
    db_session_factory, seeded_data
):
    user = seeded_data["user"]
    job = seeded_data["job"]
    candidate = seeded_data["candidate"]
    missing_email_candidate = seeded_data["missing_email_candidate"]

    collection = create_collection(
        CollectionCreateRequest(created_by_user_id=user.id, name="Dispatch shortlist")
    )
    add_item(uuid.UUID(collection.id), ItemAddRequest(candidate_profile_id=candidate.id))
    add_item(
        uuid.UUID(collection.id),
        ItemAddRequest(candidate_profile_id=missing_email_candidate.id),
    )

    with db_session_factory() as db:
        template = InterviewTemplate(
            job_id=job.id,
            name="Screening",
            status="active",
            language_code="vi-VN",
            question_payload={"questions": [{"prompt": "Tell us about yourself"}]},
            report_rubric={},
        )
        db.add(template)
        db.flush()
        outreach = OutreachMessage(
            candidate_profile_id=candidate.id,
            created_by_user_id=user.id,
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
            sent_by_user_id=user.id,
        )
        db.add_all([outreach, invitation])
        db.commit()

    summary = shortlist_module.get_dispatch_summary(uuid.UUID(collection.id))

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
    db_session_factory, seeded_data
):
    user = seeded_data["user"]
    candidate = seeded_data["candidate"]
    missing_email_candidate = seeded_data["missing_email_candidate"]
    collection = create_collection(
        CollectionCreateRequest(created_by_user_id=user.id, name="Draft shortlist")
    )
    add_item(uuid.UUID(collection.id), ItemAddRequest(candidate_profile_id=candidate.id))
    add_item(
        uuid.UUID(collection.id),
        ItemAddRequest(candidate_profile_id=missing_email_candidate.id),
    )

    result = shortlist_module.create_collection_outreach_drafts(
        uuid.UUID(collection.id),
        shortlist_module.OutreachDraftBatchRequest(
            candidate_profile_ids=[candidate.id, missing_email_candidate.id],
            subject_template="Invitation for {{candidate_name}}",
            body_text_template="Hi {{candidate_name}}, let's talk.",
            body_html_template="<p>Hi <strong>{{candidate_name}}</strong>, let's talk.</p>",
        ),
    )

    assert result.created_count == 1
    assert result.skipped_count == 1
    assert result.results[0].status == "created"
    assert result.results[1].status == "skipped_missing_email"

    with db_session_factory() as db:
        created = db.query(OutreachMessage).filter(OutreachMessage.candidate_profile_id == candidate.id).first()
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
    )

    assert duplicate_result.created_count == 0
    assert duplicate_result.results[0].status == "skipped_duplicate"


def test_create_interview_invitations_skips_duplicates_and_missing_email(
    db_session_factory, seeded_data
):
    user = seeded_data["user"]
    job = seeded_data["job"]
    candidate = seeded_data["candidate"]
    missing_email_candidate = seeded_data["missing_email_candidate"]
    collection = create_collection(
        CollectionCreateRequest(created_by_user_id=user.id, name="Interview shortlist")
    )
    add_item(uuid.UUID(collection.id), ItemAddRequest(candidate_profile_id=candidate.id))
    add_item(
        uuid.UUID(collection.id),
        ItemAddRequest(candidate_profile_id=missing_email_candidate.id),
    )

    with db_session_factory() as db:
        template = InterviewTemplate(
            job_id=job.id,
            name="Screening",
            status="active",
            language_code="vi-VN",
            question_payload={"questions": [{"prompt": "Tell us about yourself"}]},
            report_rubric={},
        )
        db.add(template)
        db.commit()
        db.refresh(template)
        template_id = template.id

    result = shortlist_module.create_collection_interview_invitations(
        uuid.UUID(collection.id),
        shortlist_module.InterviewInvitationBatchRequest(
            candidate_profile_ids=[candidate.id, missing_email_candidate.id],
            job_id=job.id,
            interview_template_id=template_id,
            expires_in_hours=72,
        ),
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
            interview_template_id=template_id,
        ),
    )

    assert duplicate_result.created_count == 0
    assert duplicate_result.results[0].status == "skipped_duplicate"


def test_create_interview_invitations_can_materialize_question_set_without_sending_email(
    db_session_factory,
    seeded_data,
):
    user = seeded_data["user"]
    job = seeded_data["job"]
    candidate = seeded_data["candidate"]
    missing_email_candidate = seeded_data["missing_email_candidate"]
    question_set_id = seeded_data["question_set_id"]
    collection = create_collection(
        CollectionCreateRequest(created_by_user_id=user.id, name="Question set shortlist")
    )
    add_item(uuid.UUID(collection.id), ItemAddRequest(candidate_profile_id=candidate.id))
    add_item(
        uuid.UUID(collection.id),
        ItemAddRequest(candidate_profile_id=missing_email_candidate.id),
    )

    result = shortlist_module.create_collection_interview_invitations(
        uuid.UUID(collection.id),
            shortlist_module.InterviewInvitationBatchRequest(
                candidate_profile_ids=[candidate.id, missing_email_candidate.id],
                job_id=job.id,
                interview_question_set_id=question_set_id,
                send_email=False,
            ),
        )

    assert result.created_count == 2
    assert result.skipped_count == 0
    assert all(item.status == "created" for item in result.results)

    with db_session_factory() as db:
        invitations = db.query(InterviewInvitation).order_by(InterviewInvitation.created_at.asc()).all()
        assert len(invitations) == 2
        assert invitations[0].sent_at is None
        assert invitations[0].interview_template_id == invitations[1].interview_template_id

        template = db.get(InterviewTemplate, invitations[0].interview_template_id)
        assert template is not None
        assert template.job_id == job.id
        assert template.question_payload["questions"][0]["key"] == "q1"
