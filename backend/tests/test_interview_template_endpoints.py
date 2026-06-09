import importlib.util
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import src.models  # noqa: F401, E402
from src.models.base import Base  # noqa: E402
from src.models.candidate_profile import CandidateProfile  # noqa: E402
from src.models.deps import get_current_user, get_db  # noqa: E402
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
from src.models.job_matching import InterviewQuestionSet, JobDescription  # noqa: E402
from src.models.resume_document import ResumeDocument  # noqa: E402
from src.models.user_account import UserAccount  # noqa: E402
from src.main import app  # noqa: E402


MIGRATION_PATH = BACKEND_ROOT / "migrations" / "versions" / "20260522_0006_add_voice_interview_domain.py"


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
        "job_descriptions",
        "interview_question_sets",
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

    primary_jd = JobDescription(
        job_id=primary_job.id,
        title="Primary JD",
        jd_text="Need strong communication and backend fundamentals.",
        created_by_user_id=user.id,
        is_active=True,
    )
    secondary_jd = JobDescription(
        job_id=secondary_job.id,
        title="Secondary JD",
        jd_text="Need data and automation experience.",
        created_by_user_id=user.id,
        is_active=True,
    )
    db_session.add_all([primary_jd, secondary_jd])
    db_session.flush()

    primary_question_set = InterviewQuestionSet(
        candidate_profile_id=candidate.id,
        job_description_id=primary_jd.id,
        generated_by_user_id=user.id,
        question_payload={"questions": [{"key": "q1", "prompt": "Tell me about yourself."}]},
    )
    db_session.add(primary_question_set)
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
        "user_email": user.email,
        "primary_job_id": primary_job.id,
        "secondary_job_id": secondary_job.id,
        "candidate_id": candidate.id,
        "secondary_candidate_id": secondary_candidate.id,
        "primary_job_description_id": primary_jd.id,
        "secondary_job_description_id": secondary_jd.id,
        "question_set_id": primary_question_set.id,
        "template_id": template.id,
        "secondary_template_id": secondary_template.id,
    }


@pytest.fixture()
def api_client(db_session: Session, seeded_interview_domain):
    def _override_db():
        yield db_session

    def _override_current_user():
        user = db_session.get(UserAccount, seeded_interview_domain["user_id"])
        assert user is not None
        return user

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user] = _override_current_user
    client = TestClient(app, follow_redirects=False)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()


def test_recruiter_can_create_list_get_and_update_interview_templates(
    api_client: TestClient,
    seeded_interview_domain,
):
    job_id = seeded_interview_domain["primary_job_id"]

    create_response = api_client.post(
        f"/api/v1/jobs/{job_id}/interview-templates",
        json={
            "name": "Structured Screen",
            "language_code": "en-US",
            "status": "active",
            "intro_script": "Welcome to the interview.",
            "closing_script": "Thanks for your time.",
            "question_payload": {
                "questions": [
                    {"key": "q1", "prompt": "Tell me about yourself."},
                ]
            },
            "report_rubric": {"score_bands": ["strong", "mixed", "weak"]},
        },
    )

    assert create_response.status_code == 201
    created = create_response.json()
    assert created["job_id"] == str(job_id)
    assert created["name"] == "Structured Screen"
    assert created["version"] == 1
    assert created["question_payload"]["questions"][0]["key"] == "q1"

    list_response = api_client.get(f"/api/v1/jobs/{job_id}/interview-templates")
    assert list_response.status_code == 200
    listed = list_response.json()
    assert listed["total"] == 2
    assert {item["id"] for item in listed["items"]} >= {created["id"], str(seeded_interview_domain["template_id"])}

    get_response = api_client.get(f"/api/v1/interview-templates/{created['id']}")
    assert get_response.status_code == 200
    fetched = get_response.json()
    assert fetched["id"] == created["id"]
    assert fetched["intro_script"] == "Welcome to the interview."

    patch_response = api_client.patch(
        f"/api/v1/interview-templates/{created['id']}",
        json={
            "name": "Structured Screen v2",
            "question_payload": {
                "questions": [
                    {"key": "q1", "prompt": "Walk me through your most relevant project."},
                ]
            },
        },
    )
    assert patch_response.status_code == 200
    updated = patch_response.json()
    assert updated["name"] == "Structured Screen v2"
    assert updated["version"] == 2
    assert updated["question_payload"]["questions"][0]["prompt"].startswith("Walk me through")

    whitespace_patch_response = api_client.patch(
        f"/api/v1/interview-templates/{created['id']}",
        json={
            "intro_script": "  Welcome to the interview.  ",
            "closing_script": "   Thanks for your time.   ",
        },
    )
    assert whitespace_patch_response.status_code == 200
    whitespace_updated = whitespace_patch_response.json()
    assert whitespace_updated["version"] == 2
    assert whitespace_updated["intro_script"] == "Welcome to the interview."
    assert whitespace_updated["closing_script"] == "Thanks for your time."


def test_interview_template_endpoints_enforce_recruiter_job_ownership(
    db_session: Session,
    seeded_interview_domain,
):
    outsider = UserAccount(
        email="outsider@example.com",
        display_name="Outsider",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    db_session.add(outsider)
    db_session.commit()

    def _override_db():
        yield db_session

    def _override_current_user():
        return db_session.get(UserAccount, outsider.id)

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user] = _override_current_user
    client = TestClient(app, follow_redirects=False)
    try:
        response = client.get(
            f"/api/v1/jobs/{seeded_interview_domain['primary_job_id']}/interview-templates"
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404


def test_recruiter_can_create_and_list_interview_invitations(
    api_client: TestClient,
    seeded_interview_domain,
    monkeypatch,
):
    import worker.tasks as tasks_module

    queued = []

    class FakeTask:
        @staticmethod
        def delay(invitation_id):
            queued.append(invitation_id)

    monkeypatch.setattr(tasks_module, "send_interview_invitation_email", FakeTask, raising=False)

    job_id = seeded_interview_domain["primary_job_id"]
    template_id = seeded_interview_domain["template_id"]
    candidate_id = seeded_interview_domain["candidate_id"]

    create_response = api_client.post(
        "/api/v1/interview-invitations",
        json={
            "job_id": str(job_id),
            "candidate_profile_id": str(candidate_id),
            "interview_template_id": str(template_id),
            "expires_in_hours": 48,
        },
    )

    assert create_response.status_code == 201
    created = create_response.json()
    assert created["job_id"] == str(job_id)
    assert created["candidate_profile_id"] == str(candidate_id)
    assert created["interview_template_id"] == str(template_id)
    assert created["status"] == "pending"
    assert created["attempt_count"] == 0
    assert created["max_attempts"] == 1
    assert created["sent_by_user_id"] == str(seeded_interview_domain["user_id"])
    assert created["public_url"].endswith(f"/interviews/{created['public_token']}")
    assert created["expires_at"] is not None

    list_response = api_client.get(f"/api/v1/jobs/{job_id}/interview-invitations")
    assert list_response.status_code == 200
    listed = list_response.json()
    assert listed["total"] == 1
    assert listed["items"][0]["id"] == created["id"]
    assert listed["items"][0]["public_url"] == created["public_url"]
    assert queued == [created["id"]]


def test_recruiter_can_create_interview_invitation_from_question_set_without_sending_email(
    api_client: TestClient,
    seeded_interview_domain,
    monkeypatch,
):
    import worker.tasks as tasks_module

    queued = []

    class FakeTask:
        @staticmethod
        def delay(invitation_id):
            queued.append(invitation_id)

    monkeypatch.setattr(tasks_module, "send_interview_invitation_email", FakeTask, raising=False)

    create_response = api_client.post(
        "/api/v1/interview-invitations",
        json={
            "job_id": str(seeded_interview_domain["primary_job_id"]),
            "candidate_profile_id": str(seeded_interview_domain["candidate_id"]),
            "interview_question_set_id": str(seeded_interview_domain["question_set_id"]),
            "send_email": False,
        },
    )

    assert create_response.status_code == 201
    created = create_response.json()
    assert created["candidate_profile_id"] == str(seeded_interview_domain["candidate_id"])
    assert created["sent_at"] is None
    assert created["status"] == "pending"
    assert queued == []

    get_template_response = api_client.get(
        f"/api/v1/interview-templates/{created['interview_template_id']}"
    )
    assert get_template_response.status_code == 200
    materialized = get_template_response.json()
    assert materialized["job_id"] == str(seeded_interview_domain["primary_job_id"])
    assert materialized["question_payload"]["questions"][0]["key"] == "q1"


def test_recruiter_can_delete_unused_interview_template(
    api_client: TestClient,
    seeded_interview_domain,
):
    create_response = api_client.post(
        f"/api/v1/jobs/{seeded_interview_domain['primary_job_id']}/interview-templates",
        json={
            "name": "Delete Me",
            "status": "draft",
        },
    )
    assert create_response.status_code == 201
    template_id = create_response.json()["id"]

    delete_response = api_client.delete(f"/api/v1/interview-templates/{template_id}")

    assert delete_response.status_code == 200
    assert delete_response.json() == {"deleted": True, "template_id": template_id}

    list_response = api_client.get(
        f"/api/v1/jobs/{seeded_interview_domain['primary_job_id']}/interview-templates"
    )
    assert list_response.status_code == 200
    assert {item["id"] for item in list_response.json()["items"]} == {
        str(seeded_interview_domain["template_id"])
    }


def test_recruiter_can_revoke_pending_interview_invitation(
    api_client: TestClient,
    seeded_interview_domain,
    monkeypatch,
):
    import worker.tasks as tasks_module

    monkeypatch.setattr(
        tasks_module,
        "send_interview_invitation_email",
        type("FakeTask", (), {"delay": staticmethod(lambda invitation_id: None)}),
        raising=False,
    )

    create_response = api_client.post(
        "/api/v1/interview-invitations",
        json={
            "job_id": str(seeded_interview_domain["primary_job_id"]),
            "candidate_profile_id": str(seeded_interview_domain["candidate_id"]),
            "interview_template_id": str(seeded_interview_domain["template_id"]),
        },
    )
    assert create_response.status_code == 201
    invitation_id = create_response.json()["id"]

    revoke_response = api_client.post(f"/api/v1/interview-invitations/{invitation_id}/revoke")

    assert revoke_response.status_code == 200
    revoked = revoke_response.json()
    assert revoked["id"] == invitation_id
    assert revoked["status"] == "cancelled"
    assert revoked["cancelled_at"] is not None


def test_recruiter_cannot_delete_template_with_existing_invitation(
    api_client: TestClient,
    seeded_interview_domain,
    monkeypatch,
):
    import worker.tasks as tasks_module

    monkeypatch.setattr(
        tasks_module,
        "send_interview_invitation_email",
        type("FakeTask", (), {"delay": staticmethod(lambda invitation_id: None)}),
        raising=False,
    )

    create_response = api_client.post(
        "/api/v1/interview-invitations",
        json={
            "job_id": str(seeded_interview_domain["primary_job_id"]),
            "candidate_profile_id": str(seeded_interview_domain["candidate_id"]),
            "interview_template_id": str(seeded_interview_domain["template_id"]),
        },
    )
    assert create_response.status_code == 201

    delete_response = api_client.delete(
        f"/api/v1/interview-templates/{seeded_interview_domain['template_id']}"
    )

    assert delete_response.status_code == 409


def test_create_interview_invitation_does_not_set_sent_at_until_email_success(
    db_session,
    seeded_interview_domain,
):
    from src.schemas.interview_invitation import InterviewInvitationCreateRequest
    from src.services.interview_invitation_service import create_interview_invitation

    invitation = create_interview_invitation(
        db_session,
        user_id=seeded_interview_domain["user_id"],
        body=InterviewInvitationCreateRequest(
            job_id=seeded_interview_domain["primary_job_id"],
            candidate_profile_id=seeded_interview_domain["candidate_id"],
            interview_template_id=seeded_interview_domain["template_id"],
            expires_in_hours=72,
        ),
    )

    assert invitation.sent_at is None
    assert invitation.status == "pending"


def test_interview_template_rejects_whitespace_only_trimmed_fields(
    api_client: TestClient,
    seeded_interview_domain,
):
    job_id = seeded_interview_domain["primary_job_id"]

    create_response = api_client.post(
        f"/api/v1/jobs/{job_id}/interview-templates",
        json={
            "name": "   ",
            "language_code": "en-US",
            "status": "active",
        },
    )
    assert create_response.status_code == 422

    patch_response = api_client.patch(
        f"/api/v1/interview-templates/{seeded_interview_domain['template_id']}",
        json={
            "language_code": "   ",
            "status": "   ",
        },
    )
    assert patch_response.status_code == 422

    create_script_response = api_client.post(
        f"/api/v1/jobs/{job_id}/interview-templates",
        json={
            "name": "Valid Name",
            "language_code": "en-US",
            "status": "active",
            "intro_script": "   ",
        },
    )
    assert create_script_response.status_code == 422

    patch_script_response = api_client.patch(
        f"/api/v1/interview-templates/{seeded_interview_domain['template_id']}",
        json={
            "closing_script": "   ",
        },
    )
    assert patch_script_response.status_code == 422


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
    captured_table_args: dict[str, tuple] = {}

    monkeypatch.setattr(
        migration_module.op,
        "create_table",
        lambda table_name, *args, **kwargs: (
            created_tables.append(table_name),
            captured_table_args.setdefault(table_name, args),
        )[-1],
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

    template_args = captured_table_args["interview_templates"]
    template_columns = {arg.name: arg for arg in template_args if hasattr(arg, "name") and hasattr(arg, "server_default")}
    template_constraints = {
        getattr(arg, "name", None): arg
        for arg in template_args
        if getattr(arg, "name", None)
    }
    assert str(template_columns["language_code"].server_default.arg) == "vi-VN"
    assert str(template_columns["status"].server_default.arg) == "draft"
    assert str(template_columns["version"].server_default.arg) == "1"
    assert "uq_interview_templates_id_job_id" in template_constraints

    invitation_args = captured_table_args["interview_invitations"]
    invitation_columns = {arg.name: arg for arg in invitation_args if hasattr(arg, "name") and hasattr(arg, "server_default")}
    invitation_constraints = {
        getattr(arg, "name", None): arg
        for arg in invitation_args
        if getattr(arg, "name", None)
    }
    assert str(invitation_columns["status"].server_default.arg) == "pending"
    assert str(invitation_columns["max_attempts"].server_default.arg) == "1"
    assert str(invitation_columns["attempt_count"].server_default.arg) == "0"
    assert {
        "ck_interview_invitations_max_attempts_positive",
        "ck_interview_invitations_attempt_count_non_negative",
        "ck_interview_invitations_attempt_count_within_max",
    }.issubset(set(invitation_constraints))
