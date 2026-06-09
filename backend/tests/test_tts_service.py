from __future__ import annotations

import types
from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.dialects.sqlite.base import SQLiteTypeCompiler
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import src.models  # noqa: F401
from src.main import app
from src.models.base import Base
from src.models.candidate_profile import CandidateProfile
from src.models.deps import get_db
from src.models.enums import ProfileStatus, UploadStatus, UserStatus
from src.models.interview_invitation import InterviewInvitation
from src.models.interview_template import InterviewTemplate
from src.models.job import Job
from src.models.resume_document import ResumeDocument
from src.models.user_account import UserAccount


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


if not hasattr(SQLiteTypeCompiler, "visit_JSONB"):
    SQLiteTypeCompiler.visit_JSONB = SQLiteTypeCompiler.visit_JSON


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
    with Session(bind=engine) as session:
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
        language_code="en",
        intro_script="Welcome to the interview.",
        closing_script="Thanks for your time.",
        question_payload={"questions": [{"key": "system_design", "prompt": "Tell me about a backend system you built."}]},
    )
    db_session.add(template)
    db_session.flush()

    invitation = InterviewInvitation(
        job_id=job.id,
        candidate_profile_id=candidate.id,
        interview_template_id=template.id,
        public_token="valid-public-token",
        status="pending",
        max_attempts=1,
        attempt_count=0,
    )
    db_session.add(invitation)
    db_session.commit()
    db_session.refresh(invitation)
    return invitation


def test_synthesize_speech_uses_edge_tts_when_available(monkeypatch):
    from src.services import tts_service

    class _FakeCommunicate:
        def __init__(self, text: str, voice: str, rate: str, volume: str):
            self.text = text
            self.voice = voice
            self.rate = rate
            self.volume = volume

        async def stream(self):
            yield {"type": "audio", "data": b"edge-audio"}

    monkeypatch.setattr(
        tts_service,
        "edge_tts",
        types.SimpleNamespace(Communicate=_FakeCommunicate),
    )
    monkeypatch.setattr(tts_service.settings, "EDGE_TTS_VOICE_VI", "vi-VN-HoaiMyNeural")

    audio = tts_service.synthesize_speech("Xin chao", language_code="vi-VN")

    assert audio == b"edge-audio"


def test_synthesize_speech_falls_back_to_shopaikey_openai_tts(monkeypatch):
    from src.services import tts_service

    class _BrokenCommunicate:
        def __init__(self, text: str, voice: str, rate: str, volume: str):
            pass

        async def stream(self):
            raise RuntimeError("edge down")
            yield

    fallback_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        tts_service,
        "edge_tts",
        types.SimpleNamespace(Communicate=_BrokenCommunicate),
    )
    monkeypatch.setattr(
        tts_service,
        "_synthesize_with_shopaikey_openai_tts",
        lambda text, *, language_code: fallback_calls.append((text, language_code)) or b"fallback-audio",
    )

    audio = tts_service.synthesize_speech("Hello there", language_code="en-US")

    assert audio == b"fallback-audio"
    assert fallback_calls == [("Hello there", "en-US")]


def test_public_interview_tts_endpoint_returns_audio_blob(db_session, interview_invitation, monkeypatch):
    from src.services import interview_session_service

    def _override_db():
        yield db_session

    monkeypatch.setattr(
        interview_session_service,
        "synthesize_speech",
        lambda text, *, language_code: f"{language_code}:{text}".encode("utf-8"),
    )
    app.dependency_overrides[get_db] = _override_db
    try:
        client = TestClient(app)
        response = client.post(
            f"/api/v1/public/interview/{interview_invitation.public_token}/tts",
            json={"text": "Welcome to the interview."},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content == b"en:Welcome to the interview."


def test_public_interview_tts_endpoint_rejects_blank_text(db_session, interview_invitation):
    def _override_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_db
    try:
        client = TestClient(app)
        response = client.post(
            f"/api/v1/public/interview/{interview_invitation.public_token}/tts",
            json={"text": "   "},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
