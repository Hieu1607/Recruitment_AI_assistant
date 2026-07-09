import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.v1.endpoints.activities import router  # noqa: E402
from src.models.base import Base  # noqa: E402
from src.models.candidate_profile import CandidateProfile  # noqa: E402
from src.models.deps import get_current_user, get_db  # noqa: E402
from src.models.enums import ContentSource, MatchRunStatus, SentStatus, UploadStatus, UserStatus  # noqa: E402
from src.models.interview_invitation import InterviewInvitation  # noqa: E402
from src.models.interview_template import InterviewTemplate  # noqa: E402
from src.models.job import Job  # noqa: E402
from src.models.job_matching import JobDescription, MatchRun  # noqa: E402
from src.models.outreach import OutreachMessage  # noqa: E402
from src.models.query_shortlist import ShortlistCollection, ShortlistItem  # noqa: E402
from src.models.resume_document import ExtractionTrace, ResumeDocument  # noqa: E402
from src.models.user_account import UserAccount  # noqa: E402


def _create_test_tables(engine):
    Base.metadata.create_all(
        engine,
        tables=[
            Base.metadata.tables["user_accounts"],
            Base.metadata.tables["jobs"],
            Base.metadata.tables["resume_processing_batches"],
            Base.metadata.tables["resume_documents"],
            Base.metadata.tables["candidate_profiles"],
            Base.metadata.tables["outreach_messages"],
            Base.metadata.tables["interview_templates"],
            Base.metadata.tables["interview_invitations"],
            Base.metadata.tables["job_descriptions"],
            Base.metadata.tables["match_runs"],
            Base.metadata.tables["shortlist_collections"],
            Base.metadata.tables["shortlist_items"],
            Base.metadata.tables["extraction_traces"],
        ],
    )


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    _create_test_tables(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture()
def users(db):
    current = UserAccount(
        email="current@example.com",
        display_name="Current User",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    other = UserAccount(
        email="other@example.com",
        display_name="Other User",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    db.add_all([current, other])
    db.commit()
    db.refresh(current)
    db.refresh(other)
    return current, other


@pytest.fixture()
def client(db, users):
    current, _other = users
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/activities")

    def _override_db():
        yield db

    def _override_current_user():
        return current

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user] = _override_current_user

    with TestClient(app) as test_client:
        yield test_client


def _job(db, owner: UserAccount, title: str) -> Job:
    item = Job(
        owner_user_id=owner.id,
        title=title,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def _resume(
    db,
    *,
    job: Job,
    owner: UserAccount,
    name: str,
    uploaded_at: datetime,
    upload_status: UploadStatus,
    processed_at: datetime | None = None,
) -> ResumeDocument:
    item = ResumeDocument(
        original_file_name=name,
        storage_uri=f"s3://bucket/{name}",
        upload_status=upload_status,
        job_id=job.id,
        uploaded_by_user_id=owner.id,
        uploaded_at=uploaded_at,
        processed_at=processed_at,
        retention_expires_at=uploaded_at + timedelta(days=30),
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def _candidate(db, *, resume: ResumeDocument, name: str) -> CandidateProfile:
    item = CandidateProfile(
        resume_document_id=resume.id,
        full_name=name,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def _shortlist_item(
    db,
    *,
    owner: UserAccount,
    candidate: CandidateProfile,
    name: str,
    added_at: datetime,
) -> ShortlistItem:
    collection = ShortlistCollection(
        name=name,
        created_by_user_id=owner.id,
    )
    db.add(collection)
    db.commit()
    db.refresh(collection)

    item = ShortlistItem(
        shortlist_collection_id=collection.id,
        candidate_profile_id=candidate.id,
        added_at=added_at,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def _outreach(
    db,
    *,
    candidate: CandidateProfile,
    owner: UserAccount,
    subject: str,
    status: SentStatus,
    created_at: datetime,
    sent_at: datetime | None = None,
) -> OutreachMessage:
    item = OutreachMessage(
        candidate_profile_id=candidate.id,
        created_by_user_id=owner.id,
        content_source=ContentSource.TEMPLATE,
        subject=subject,
        body_text="Hello",
        body_html="<p>Hello</p>",
        sent_status=status,
        created_at=created_at,
        sent_at=sent_at,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def _interview_template(db, *, job: Job, name: str) -> InterviewTemplate:
    item = InterviewTemplate(
        job_id=job.id,
        name=name,
        question_payload={},
        report_rubric={},
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def _invitation(
    db,
    *,
    job: Job,
    candidate: CandidateProfile,
    template: InterviewTemplate,
    created_at: datetime,
    sent_by_user_id,
    sent_at: datetime | None = None,
    completed_at: datetime | None = None,
    cancelled_at: datetime | None = None,
    status: str = "pending",
) -> InterviewInvitation:
    item = InterviewInvitation(
        job_id=job.id,
        candidate_profile_id=candidate.id,
        interview_template_id=template.id,
        status=status,
        created_at=created_at,
        sent_by_user_id=sent_by_user_id,
        sent_at=sent_at,
        completed_at=completed_at,
        cancelled_at=cancelled_at,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def _match_run(
    db,
    *,
    job: Job,
    owner: UserAccount,
    created_at: datetime,
    completed_at: datetime,
) -> MatchRun:
    jd = JobDescription(
        job_id=job.id,
        created_by_user_id=owner.id,
        title=f"{job.title} JD",
        jd_text="JD body",
        is_active=True,
        created_at=created_at,
    )
    db.add(jd)
    db.commit()
    db.refresh(jd)

    run = MatchRun(
        job_description_id=jd.id,
        score_threshold=75,
        initiated_by_user_id=owner.id,
        run_status=MatchRunStatus.COMPLETED,
        created_at=created_at,
        completed_at=completed_at,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def _failed_trace(db, *, resume: ResumeDocument, created_at: datetime, message: str = "Parser failed") -> ExtractionTrace:
    item = ExtractionTrace(
        resume_document_id=resume.id,
        stage="parse",
        status="failed",
        message=message,
        created_at=created_at,
        payload={},
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def test_activity_feed_merges_meaningful_events_in_descending_order(db, users, client):
    current, other = users
    now = datetime.now(timezone.utc)
    job = _job(db, current, "Backend Engineer")
    other_job = _job(db, other, "Other Job")

    processed_resume = _resume(
        db,
        job=job,
        owner=current,
        name="alice-nguyen.pdf",
        uploaded_at=now - timedelta(minutes=20),
        upload_status=UploadStatus.PROCESSED,
        processed_at=now - timedelta(minutes=5),
    )
    processed_candidate = _candidate(db, resume=processed_resume, name="Alice Nguyen")
    _shortlist_item(
        db,
        owner=current,
        candidate=processed_candidate,
        name="Top 10",
        added_at=now - timedelta(minutes=3),
    )
    _outreach(
        db,
        candidate=processed_candidate,
        owner=current,
        subject="Interview next steps",
        status=SentStatus.SENT,
        created_at=now - timedelta(minutes=2),
        sent_at=now - timedelta(minutes=1),
    )

    invitation_resume = _resume(
        db,
        job=job,
        owner=current,
        name="bao-tran.pdf",
        uploaded_at=now - timedelta(minutes=40),
        upload_status=UploadStatus.UPLOADED,
    )
    invitation_candidate = _candidate(db, resume=invitation_resume, name="Bao Tran")
    template = _interview_template(db, job=job, name="Phone Screen")
    _invitation(
        db,
        job=job,
        candidate=invitation_candidate,
        template=template,
        created_at=now - timedelta(minutes=2),
        sent_by_user_id=current.id,
        status="pending",
    )

    _match_run(
        db,
        job=job,
        owner=current,
        created_at=now - timedelta(minutes=12),
        completed_at=now - timedelta(minutes=7),
    )

    other_resume = _resume(
        db,
        job=other_job,
        owner=other,
        name="other.pdf",
        uploaded_at=now - timedelta(minutes=1),
        upload_status=UploadStatus.UPLOADED,
    )
    other_candidate = _candidate(db, resume=other_resume, name="Other Candidate")
    _outreach(
        db,
        candidate=other_candidate,
        owner=other,
        subject="Should stay hidden",
        status=SentStatus.SENT,
        created_at=now - timedelta(minutes=1),
        sent_at=now - timedelta(seconds=30),
    )

    response = client.get(f"/api/v1/activities/?job_id={job.id}&limit=10")

    assert response.status_code == 200
    payload = response.json()
    kinds = [item["kind"] for item in payload["items"]]
    assert kinds == [
        "outreach_sent",
        "interview_link_created",
        "shortlist_added",
        "resume_processed",
        "scoring_completed",
    ]
    assert all(item["subject_name"] != "Other Candidate" for item in payload["items"])
    assert all(item["kind"] != "resume_uploaded" for item in payload["items"])


def test_activity_feed_de_noises_terminal_resume_and_sent_interview_events(db, users, client):
    current, _other = users
    now = datetime.now(timezone.utc)
    job = _job(db, current, "Frontend Engineer")

    failed_resume = _resume(
        db,
        job=job,
        owner=current,
        name="failed-cv.pdf",
        uploaded_at=now - timedelta(minutes=30),
        upload_status=UploadStatus.FAILED,
    )
    _failed_trace(
        db,
        resume=failed_resume,
        created_at=now - timedelta(minutes=4),
        message="OCR provider timeout",
    )

    uploaded_resume = _resume(
        db,
        job=job,
        owner=current,
        name="fresh-upload.pdf",
        uploaded_at=now - timedelta(minutes=6),
        upload_status=UploadStatus.UPLOADED,
    )
    sent_candidate = _candidate(db, resume=uploaded_resume, name="Fresh Upload")

    sent_template = _interview_template(db, job=job, name="Async Interview")
    _invitation(
        db,
        job=job,
        candidate=sent_candidate,
        template=sent_template,
        created_at=now - timedelta(minutes=10),
        sent_by_user_id=current.id,
        sent_at=now - timedelta(minutes=2),
        status="sent",
    )
    _resume(
        db,
        job=job,
        owner=current,
        name="new-upload-only.pdf",
        uploaded_at=now - timedelta(minutes=1),
        upload_status=UploadStatus.UPLOADED,
    )

    response = client.get(f"/api/v1/activities/?job_id={job.id}&limit=10")

    assert response.status_code == 200
    payload = response.json()
    kinds = [item["kind"] for item in payload["items"]]
    assert "resume_failed" in kinds
    assert "resume_uploaded" in kinds
    assert "interview_invitation_sent" in kinds
    assert "interview_link_created" not in kinds

    failed_item = next(item for item in payload["items"] if item["kind"] == "resume_failed")
    assert failed_item["subject_name"] == "failed-cv.pdf"
    assert failed_item["metadata"]["message"] == "OCR provider timeout"
