import importlib.util
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import src.models  # noqa: F401, E402
from src.models.base import Base  # noqa: E402
from src.models.candidate_profile import CandidateProfile  # noqa: E402
from src.models.enums import ProfileStatus, UploadStatus, UserStatus  # noqa: E402
from src.models.interview_invitation import InterviewInvitation  # noqa: E402
from src.models.interview_session import (  # noqa: E402
    InterviewReport,
    InterviewResponseItem,
    InterviewSession,
    InterviewTranscriptTurn,
)
from src.models.interview_template import InterviewTemplate  # noqa: E402
from src.models.job import Job  # noqa: E402
from src.models.resume_document import ResumeDocument  # noqa: E402
from src.models.user_account import UserAccount  # noqa: E402


MIGRATION_PATH = BACKEND_ROOT / "migrations" / "versions" / "20260522_0006_add_voice_interview_domain.py"


def _make_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
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
def seeded_interview_domain(db_session: Session):
    user = UserAccount(
        email="owner@example.com",
        display_name="Owner",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    db_session.add(user)
    db_session.flush()

    primary_job = Job(owner_user_id=user.id, title="Primary Job", status="active")
    secondary_job = Job(owner_user_id=user.id, title="Secondary Job", status="active")
    db_session.add_all([primary_job, secondary_job])
    db_session.flush()

    resume = ResumeDocument(
        original_file_name="candidate.pdf",
        storage_uri="s3://bucket/resumes/candidate.pdf",
        upload_status=UploadStatus.PROCESSED,
        job_id=primary_job.id,
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

    secondary_resume = ResumeDocument(
        original_file_name="candidate-two.pdf",
        storage_uri="s3://bucket/resumes/candidate-two.pdf",
        upload_status=UploadStatus.PROCESSED,
        job_id=secondary_job.id,
        uploaded_by_user_id=user.id,
        retention_expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
    )
    db_session.add(secondary_resume)
    db_session.flush()

    secondary_candidate = CandidateProfile(
        resume_document_id=secondary_resume.id,
        full_name="Candidate Two",
        email="candidate-two@example.com",
        profile_status=ProfileStatus.REVIEWED,
    )
    db_session.add(secondary_candidate)
    db_session.flush()

    template = InterviewTemplate(
        job_id=primary_job.id,
        name="Voice Screen",
        question_payload={"questions": []},
        report_rubric={"rubric": []},
    )
    secondary_template = InterviewTemplate(
        job_id=secondary_job.id,
        name="Secondary Voice Screen",
        question_payload={"questions": []},
        report_rubric={"rubric": []},
    )
    db_session.add_all([template, secondary_template])
    db_session.commit()

    return {
        "user_id": user.id,
        "primary_job_id": primary_job.id,
        "secondary_job_id": secondary_job.id,
        "candidate_id": candidate.id,
        "secondary_candidate_id": secondary_candidate.id,
        "template_id": template.id,
        "secondary_template_id": secondary_template.id,
    }


def test_voice_interview_tables_are_registered_in_metadata():
    engine = _make_engine()
    _create_test_tables(engine)

    assert set(inspect(engine).get_table_names()) >= {
        "interview_templates",
        "interview_invitations",
        "interview_sessions",
        "interview_response_items",
        "interview_transcript_turns",
        "interview_reports",
    }


def test_invitation_constraints_and_defaults_are_enforced(db_session: Session, seeded_interview_domain):
    template_id = seeded_interview_domain["template_id"]
    primary_job_id = seeded_interview_domain["primary_job_id"]
    secondary_job_id = seeded_interview_domain["secondary_job_id"]
    candidate_id = seeded_interview_domain["candidate_id"]

    invitation = InterviewInvitation(
        job_id=primary_job_id,
        candidate_profile_id=candidate_id,
        interview_template_id=template_id,
    )
    db_session.add(invitation)
    db_session.commit()
    db_session.refresh(invitation)

    assert invitation.public_token
    assert invitation.max_attempts == 1
    assert invitation.attempt_count == 0

    db_session.add(
        InterviewInvitation(
            job_id=secondary_job_id,
            candidate_profile_id=seeded_interview_domain["secondary_candidate_id"],
            interview_template_id=template_id,
        )
    )
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    for kwargs in (
        {"max_attempts": 0, "attempt_count": 0},
        {"max_attempts": 1, "attempt_count": -1},
        {"max_attempts": 1, "attempt_count": 2},
    ):
        db_session.add(
            InterviewInvitation(
                job_id=primary_job_id,
                candidate_profile_id=candidate_id,
                interview_template_id=template_id,
                public_token=f"manual-{uuid.uuid4().hex}",
                **kwargs,
            )
        )
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()


def test_invitation_rejects_candidate_job_mismatch_before_flush(db_session: Session, seeded_interview_domain):
    db_session.add(
        InterviewInvitation(
            job_id=seeded_interview_domain["secondary_job_id"],
            candidate_profile_id=seeded_interview_domain["candidate_id"],
            interview_template_id=seeded_interview_domain["secondary_template_id"],
        )
    )

    with pytest.raises(ValueError, match="candidate profile resume_document.job_id"):
        db_session.flush()
    db_session.rollback()


def test_response_item_delete_preserves_transcript_turns(db_session: Session, seeded_interview_domain):
    invitation = InterviewInvitation(
        job_id=seeded_interview_domain["primary_job_id"],
        candidate_profile_id=seeded_interview_domain["candidate_id"],
        interview_template_id=seeded_interview_domain["template_id"],
    )
    db_session.add(invitation)
    db_session.flush()

    session_record = InterviewSession(interview_invitation_id=invitation.id)
    db_session.add(session_record)
    db_session.flush()

    response_item = InterviewResponseItem(
        interview_session_id=session_record.id,
        question_key="q1",
        response_text="Answer",
    )
    db_session.add(response_item)
    db_session.flush()

    transcript_turn = InterviewTranscriptTurn(
        interview_session_id=session_record.id,
        response_item_id=response_item.id,
        speaker_role="candidate",
        turn_index=0,
        transcript_text="Answer",
    )
    db_session.add(transcript_turn)
    db_session.commit()

    transcript_turn_id = transcript_turn.id
    db_session.delete(response_item)
    db_session.commit()

    preserved_turn = db_session.get(InterviewTranscriptTurn, transcript_turn_id)
    assert preserved_turn is not None
    assert preserved_turn.response_item_id is None


def test_json_server_defaults_fill_required_payloads(db_session: Session, seeded_interview_domain):
    template_table = Base.metadata.tables["interview_templates"]
    report_table = Base.metadata.tables["interview_reports"]

    template_id = uuid.uuid4()
    db_session.execute(
        template_table.insert().values(
            id=template_id,
            job_id=seeded_interview_domain["primary_job_id"],
            name="Core Insert Template",
        )
    )

    invitation = InterviewInvitation(
        job_id=seeded_interview_domain["primary_job_id"],
        candidate_profile_id=seeded_interview_domain["candidate_id"],
        interview_template_id=seeded_interview_domain["template_id"],
    )
    db_session.add(invitation)
    db_session.flush()

    session_record = InterviewSession(interview_invitation_id=invitation.id)
    db_session.add(session_record)
    db_session.flush()

    report_id = uuid.uuid4()
    db_session.execute(
        report_table.insert().values(
            id=report_id,
            interview_session_id=session_record.id,
            interview_template_id=seeded_interview_domain["template_id"],
        )
    )
    db_session.commit()

    inserted_template = db_session.get(InterviewTemplate, template_id)
    inserted_report = db_session.get(InterviewReport, report_id)

    assert inserted_template.question_payload == {}
    assert inserted_template.report_rubric == {}
    assert inserted_report.report_payload == {}


def test_voice_interview_migration_declares_expected_tables_and_indexes(monkeypatch):
    spec = importlib.util.spec_from_file_location("voice_interview_migration", MIGRATION_PATH)
    migration_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(migration_module)

    created_tables: list[str] = []
    created_indexes: list[tuple[str, str, bool]] = []

    monkeypatch.setattr(
        migration_module.op,
        "create_table",
        lambda table_name, *args, **kwargs: created_tables.append(table_name),
    )
    monkeypatch.setattr(
        migration_module.op,
        "create_index",
        lambda index_name, table_name, columns, unique=False: created_indexes.append(
            (index_name, table_name, unique)
        ),
    )

    migration_module.upgrade()

    assert created_tables == [
        "interview_templates",
        "interview_invitations",
        "interview_sessions",
        "interview_response_items",
        "interview_transcript_turns",
        "interview_reports",
    ]
    assert {
        ("ix_interview_templates_job_id", "interview_templates", False),
        ("ix_interview_invitations_public_token", "interview_invitations", True),
        ("ix_interview_sessions_interview_invitation_id", "interview_sessions", False),
        ("ix_interview_response_items_interview_session_id", "interview_response_items", False),
        ("ix_interview_transcript_turns_response_item_id", "interview_transcript_turns", False),
        ("ix_interview_reports_interview_session_id", "interview_reports", True),
    }.issubset(set(created_indexes))
