import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import src.models  # noqa: F401, E402
from src.main import app  # noqa: E402
from src.models.base import Base  # noqa: E402
from src.models.candidate_profile import CandidateProfile  # noqa: E402
from src.models.deps import get_db  # noqa: E402
from src.models.enums import ProfileStatus, UploadStatus, UserStatus  # noqa: E402
from src.models.interview_invitation import InterviewInvitation  # noqa: E402
from src.models.interview_session import InterviewSession, InterviewTranscriptTurn  # noqa: E402
from src.models.interview_template import InterviewTemplate  # noqa: E402
from src.models.job import Job  # noqa: E402
from src.models.resume_document import ResumeDocument  # noqa: E402
from src.models.user_account import UserAccount  # noqa: E402


def _make_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def _create_test_tables(engine):
    table_names = [
        "user_accounts",
        "jobs",
        "resume_documents",
        "candidate_profiles",
        "interview_templates",
        "interview_invitations",
        "interview_sessions",
        "interview_response_items",
        "interview_transcript_turns",
        "interview_reports",
    ]
    Base.metadata.create_all(engine, tables=[Base.metadata.tables[name] for name in table_names])


@pytest.fixture()
def db_session():
    engine = _make_engine()
    _create_test_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with factory() as session:
        yield session


@pytest.fixture()
def interview_invitation(db_session: Session):
    user = UserAccount(
        email="owner@example.com",
        display_name="Owner",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    db_session.add(user)
    db_session.flush()

    job = Job(owner_user_id=user.id, title="Backend Engineer", status="active")
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

    template = InterviewTemplate(
        job_id=job.id,
        name="Voice Screen",
        status="active",
        intro_script="Welcome to the interview.",
        closing_script="Thanks for your time.",
        question_payload={
            "questions": [
                {"key": "intro", "prompt": "Tell me about yourself."},
            ]
        },
        report_rubric={"score_bands": ["strong", "mixed", "weak"]},
    )
    db_session.add(template)
    db_session.flush()

    invitation = InterviewInvitation(
        job_id=job.id,
        candidate_profile_id=candidate.id,
        interview_template_id=template.id,
        public_token="valid-public-token",
        status="pending",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=4),
        max_attempts=1,
        attempt_count=0,
    )
    db_session.add(invitation)
    db_session.commit()
    db_session.refresh(invitation)
    return invitation


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


def test_public_interview_start_events_and_complete_flow(
    api_client: TestClient,
    db_session: Session,
    interview_invitation: InterviewInvitation,
):
    start_response = api_client.post(
        f"/api/v1/public/interview/{interview_invitation.public_token}/start",
        json={
            "provider": "FAKE",
            "provider_session_id": " provider-session-1 ",
            "device_metadata": {"kind": "laptop"},
        },
    )

    assert start_response.status_code == 200
    started = start_response.json()
    assert started["invitation"]["public_token"] == interview_invitation.public_token
    assert started["invitation"]["status"] == "in_progress"
    assert started["invitation"]["attempt_count"] == 1
    assert started["session"]["provider"] == "fake"
    assert started["session"]["provider_session_id"] == "provider-session-1"
    assert started["session"]["status"] == "in_progress"
    assert started["template"]["name"] == "Voice Screen"
    assert started["template"]["question_payload"]["questions"][0]["key"] == "intro"

    events_response = api_client.post(
        f"/api/v1/public/interview/{interview_invitation.public_token}/events",
        json={
            "provider": "fake",
            "events": [
                {
                    "speaker": "agent",
                    "text": "Welcome to the interview.",
                    "offset_ms": "12",
                    "question_key": "intro",
                },
                {
                    "speaker": "user",
                    "text": "I build APIs.",
                    "offset_ms": 44,
                },
            ],
        },
    )

    assert events_response.status_code == 202
    assert events_response.json() == {"accepted": True, "stored_turns": 2}

    transcript_turns = db_session.execute(
        select(InterviewTranscriptTurn).order_by(InterviewTranscriptTurn.turn_index.asc())
    ).scalars().all()
    assert [turn.speaker_role for turn in transcript_turns] == ["assistant", "candidate"]
    assert [turn.transcript_text for turn in transcript_turns] == [
        "Welcome to the interview.",
        "I build APIs.",
    ]
    assert transcript_turns[0].payload["question_key"] == "intro"
    assert "question_key" not in (transcript_turns[1].payload or {})

    complete_response = api_client.post(
        f"/api/v1/public/interview/{interview_invitation.public_token}/complete",
        json={"provider": "fake"},
    )

    assert complete_response.status_code == 200
    completed = complete_response.json()
    assert completed["invitation"]["status"] == "completed"
    assert completed["session"]["status"] == "completed"

    db_session.expire_all()
    invitation = db_session.get(InterviewInvitation, interview_invitation.id)
    session_record = db_session.execute(select(InterviewSession)).scalar_one()
    assert invitation is not None
    assert invitation.status == "completed"
    assert invitation.completed_at is not None
    assert session_record.status == "completed"
    assert session_record.completed_at is not None


def test_public_interview_status_returns_ready_invitation_snapshot(
    api_client: TestClient,
    interview_invitation: InterviewInvitation,
):
    response = api_client.get(f"/api/v1/public/interview/{interview_invitation.public_token}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["invitation"]["public_token"] == interview_invitation.public_token
    assert payload["invitation"]["status"] == "pending"
    assert payload["template"]["name"] == "Voice Screen"
    assert payload["availability"] == {
        "can_start": True,
        "reason": "ready",
        "detail": None,
    }


@pytest.mark.parametrize(
    ("status", "expires_at", "attempt_count", "completed_at", "expected_reason", "expected_detail"),
    [
        ("cancelled", datetime.now(timezone.utc) + timedelta(hours=1), 0, None, "inactive", "Interview invitation is not active"),
        ("pending", datetime.now(timezone.utc) - timedelta(minutes=1), 0, None, "expired", "Interview invitation has expired"),
        ("pending", datetime.now(timezone.utc) + timedelta(hours=1), 1, None, "attempt_limit_reached", "Interview attempt limit has been reached"),
        ("completed", datetime.now(timezone.utc) + timedelta(hours=1), 0, datetime.now(timezone.utc), "completed", "Interview invitation is already completed"),
    ],
)
def test_public_interview_status_reports_why_start_is_blocked(
    api_client: TestClient,
    db_session: Session,
    interview_invitation: InterviewInvitation,
    status: str,
    expires_at: datetime,
    attempt_count: int,
    completed_at: datetime | None,
    expected_reason: str,
    expected_detail: str,
):
    interview_invitation.status = status
    interview_invitation.expires_at = expires_at
    interview_invitation.attempt_count = attempt_count
    interview_invitation.completed_at = completed_at
    db_session.add(interview_invitation)
    db_session.commit()

    response = api_client.get(f"/api/v1/public/interview/{interview_invitation.public_token}")

    assert response.status_code == 200
    assert response.json()["availability"] == {
        "can_start": False,
        "reason": expected_reason,
        "detail": expected_detail,
    }


def test_public_interview_events_recovers_when_turn_index_conflict_happens_during_ingest(
    api_client: TestClient,
    db_session: Session,
    interview_invitation: InterviewInvitation,
    monkeypatch: pytest.MonkeyPatch,
):
    start_response = api_client.post(
        f"/api/v1/public/interview/{interview_invitation.public_token}/start",
        json={"provider": "fake", "provider_session_id": "provider-session-1"},
    )
    assert start_response.status_code == 200

    session_record = db_session.execute(select(InterviewSession)).scalar_one()
    db_session.add(
        InterviewTranscriptTurn(
            interview_session_id=session_record.id,
            speaker_role="assistant",
            turn_index=0,
            transcript_text="Tell me about yourself.",
            payload={"question_key": "intro"},
        )
    )
    db_session.commit()

    original_execute = db_session.execute
    execute_state = {"returned_stale_max": False}

    class _FakeScalarResult:
        def scalar_one(self):
            return None

    def execute_with_stale_turn_index(statement, *args, **kwargs):
        statement_text = str(statement)
        if (
            not execute_state["returned_stale_max"]
            and "max(interview_transcript_turns.turn_index)" in statement_text
        ):
            execute_state["returned_stale_max"] = True
            return _FakeScalarResult()
        return original_execute(statement, *args, **kwargs)

    monkeypatch.setattr(db_session, "execute", execute_with_stale_turn_index)

    events_response = api_client.post(
        f"/api/v1/public/interview/{interview_invitation.public_token}/events",
        json={
            "provider": "fake",
            "events": [
                {
                    "speaker": "user",
                    "text": "I build APIs.",
                    "question_key": "intro",
                }
            ],
        },
    )

    assert events_response.status_code == 202
    assert events_response.json() == {"accepted": True, "stored_turns": 1}

    transcript_turns = db_session.execute(
        select(InterviewTranscriptTurn).order_by(InterviewTranscriptTurn.turn_index.asc())
    ).scalars().all()
    assert [turn.turn_index for turn in transcript_turns] == [0, 1]
    assert [turn.speaker_role for turn in transcript_turns] == ["assistant", "candidate"]
    assert transcript_turns[1].transcript_text == "I build APIs."
    assert transcript_turns[1].payload["question_key"] == "intro"


@pytest.mark.parametrize(
    ("status", "expires_at", "attempt_count", "completed_at", "expected_status", "expected_detail"),
    [
        ("cancelled", datetime.now(timezone.utc) + timedelta(hours=1), 0, None, 410, "Interview invitation is not active"),
        ("pending", datetime.now(timezone.utc) - timedelta(minutes=1), 0, None, 410, "Interview invitation has expired"),
        ("pending", datetime.now(timezone.utc) + timedelta(hours=1), 1, None, 409, "Interview attempt limit has been reached"),
        ("completed", datetime.now(timezone.utc) + timedelta(hours=1), 0, datetime.now(timezone.utc), 409, "Interview invitation is already completed"),
    ],
)
def test_public_interview_start_validates_invitation_state(
    api_client: TestClient,
    db_session: Session,
    interview_invitation: InterviewInvitation,
    status: str,
    expires_at: datetime,
    attempt_count: int,
    completed_at: datetime | None,
    expected_status: int,
    expected_detail: str,
):
    interview_invitation.status = status
    interview_invitation.expires_at = expires_at
    interview_invitation.attempt_count = attempt_count
    interview_invitation.completed_at = completed_at
    db_session.add(interview_invitation)
    db_session.commit()

    response = api_client.post(
        f"/api/v1/public/interview/{interview_invitation.public_token}/start",
        json={"provider": "fake"},
    )

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}


def test_public_interview_start_rejects_duplicate_active_session_without_incrementing_attempts(
    api_client: TestClient,
    db_session: Session,
    interview_invitation: InterviewInvitation,
):
    first_response = api_client.post(
        f"/api/v1/public/interview/{interview_invitation.public_token}/start",
        json={"provider": "fake"},
    )
    assert first_response.status_code == 200

    duplicate_response = api_client.post(
        f"/api/v1/public/interview/{interview_invitation.public_token}/start",
        json={"provider": "fake"},
    )

    assert duplicate_response.status_code == 409
    assert duplicate_response.json() == {"detail": "Interview session is already in progress"}

    db_session.expire_all()
    invitation = db_session.get(InterviewInvitation, interview_invitation.id)
    sessions = db_session.execute(select(InterviewSession)).scalars().all()
    assert invitation is not None
    assert invitation.attempt_count == 1
    assert len(sessions) == 1
    assert sessions[0].status == "in_progress"


def test_public_interview_start_returns_stable_4xx_for_unsupported_provider(
    api_client: TestClient,
    interview_invitation: InterviewInvitation,
):
    response = api_client.post(
        f"/api/v1/public/interview/{interview_invitation.public_token}/start",
        json={"provider": "unsupported-provider"},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Unsupported voice provider: unsupported-provider"}


def test_public_interview_events_reject_provider_switch_attempt(
    api_client: TestClient,
    db_session: Session,
    interview_invitation: InterviewInvitation,
):
    interview_invitation.status = "in_progress"
    interview_invitation.attempt_count = 1
    db_session.add(interview_invitation)
    db_session.flush()

    session_record = InterviewSession(
        interview_invitation_id=interview_invitation.id,
        provider="other-provider",
        status="in_progress",
        started_at=datetime.now(timezone.utc),
    )
    db_session.add(session_record)
    db_session.commit()

    response = api_client.post(
        f"/api/v1/public/interview/{interview_invitation.public_token}/events",
        json={
            "provider": "fake",
            "events": [{"speaker": "user", "text": "hello"}],
        },
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Interview session provider does not match"}


def test_public_interview_complete_rejects_provider_switch_attempt(
    api_client: TestClient,
    db_session: Session,
    interview_invitation: InterviewInvitation,
):
    interview_invitation.status = "in_progress"
    interview_invitation.attempt_count = 1
    db_session.add(interview_invitation)
    db_session.flush()

    session_record = InterviewSession(
        interview_invitation_id=interview_invitation.id,
        provider="other-provider",
        status="in_progress",
        started_at=datetime.now(timezone.utc),
    )
    db_session.add(session_record)
    db_session.commit()

    response = api_client.post(
        f"/api/v1/public/interview/{interview_invitation.public_token}/complete",
        json={"provider": "fake"},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Interview session provider does not match"}
