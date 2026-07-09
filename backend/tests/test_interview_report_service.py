import importlib
import logging
import sys
import types
import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.dialects.sqlite.base import SQLiteTypeCompiler
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import src.models  # noqa: F401
from src.main import app
from src.models.base import Base
from src.models.candidate_profile import CandidateProfile
from src.models.deps import get_current_user, get_db
from src.models.enums import ProfileStatus, UploadStatus, UserStatus
from src.models.interview_invitation import InterviewInvitation
from src.models.interview_session import InterviewReport, InterviewSession, InterviewTranscriptTurn
from src.models.interview_template import InterviewTemplate
from src.models.job import Job
from src.models.resume_document import ResumeDocument
from src.models.user_account import UserAccount
from src.schemas.interview_public import PublicInterviewCompleteRequest


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
        "user_notifications",
        "jobs",
        "resume_processing_batches",
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
        intro_script="Welcome to the interview.",
        closing_script="Thanks for your time.",
        question_payload={
            "questions": [
                {"key": "system_design", "prompt": "Tell me about a backend system you built."},
                {"key": "failures", "prompt": "How did you handle failures?"},
            ]
        },
        report_rubric={"focus": ["system design", "operational resilience"]},
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


def _create_completed_session(db_session: Session, interview_invitation: InterviewInvitation) -> InterviewSession:
    interview_invitation.status = "in_progress"
    interview_invitation.attempt_count = 1
    db_session.add(interview_invitation)
    db_session.flush()

    session_record = InterviewSession(
        interview_invitation_id=interview_invitation.id,
        provider="fake",
        provider_session_id="provider-session-1",
        status="completed",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )
    db_session.add(session_record)
    db_session.flush()

    db_session.add_all(
        [
            InterviewTranscriptTurn(
                interview_session_id=session_record.id,
                speaker_role="assistant",
                turn_index=0,
                transcript_text="Tell me about a backend system you built.",
                payload={"question_key": "system_design"},
            ),
            InterviewTranscriptTurn(
                interview_session_id=session_record.id,
                speaker_role="candidate",
                turn_index=1,
                transcript_text=(
                    "I built a queue-based ingestion service with retries, metrics, "
                    "and idempotent processing for candidate resumes."
                ),
                payload={"question_key": "system_design"},
            ),
            InterviewTranscriptTurn(
                interview_session_id=session_record.id,
                speaker_role="assistant",
                turn_index=2,
                transcript_text="How did you handle failures?",
                payload={"question_key": "failures"},
            ),
            InterviewTranscriptTurn(
                interview_session_id=session_record.id,
                speaker_role="candidate",
                turn_index=3,
                transcript_text=(
                    "I added dead-letter queues, alerting, and replay tooling to "
                    "recover safely from downstream outages."
                ),
                payload={"question_key": "failures"},
            ),
        ]
    )
    db_session.commit()
    db_session.refresh(session_record)
    return session_record


@pytest.fixture()
def api_client(db_session: Session, interview_invitation: InterviewInvitation):
    owner = db_session.get(UserAccount, interview_invitation.job.owner_user_id)

    def _override_db():
        yield db_session

    def _override_current_user():
        return owner

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user] = _override_current_user
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_generate_interview_report_from_completed_session_persists_structured_payload_and_markdown(
    db_session: Session,
    interview_invitation: InterviewInvitation,
    monkeypatch,
    caplog,
):
    from src.services import interview_report_service
    from src.services.llm_service import LLMResponse

    session_record = _create_completed_session(db_session, interview_invitation)
    generated_json = """
    {
      "candidate_overview": "Candidate described ownership of backend ingestion and recovery workflows.",
      "questions": [
        {
          "question_key": "system_design",
          "question_text": "Tell me about a backend system you built.",
          "question_transcript_turn_id": "__TURN_0__",
          "question_turn_index": 0,
          "answer_text": "I built a queue-based ingestion service with retries, metrics, and idempotent processing for candidate resumes.",
          "answer_transcript_turn_id": "__TURN_1__",
          "answer_turn_index": 1,
          "evaluation": "Explained queue-based ingestion architecture with retries and idempotency."
        },
        {
          "question_key": "failures",
          "question_text": "How did you handle failures?",
          "question_transcript_turn_id": "__TURN_2__",
          "question_turn_index": 2,
          "answer_text": "I added dead-letter queues, alerting, and replay tooling to recover safely from downstream outages.",
          "answer_transcript_turn_id": "__TURN_3__",
          "answer_turn_index": 3,
          "evaluation": "Covered failure handling with dead-letter queues, alerting, and replay tooling."
        }
      ],
      "overall_summary": "Interview shows backend experience with reliability-focused examples."
    }
    """

    monkeypatch.setattr(
        interview_report_service.LLMProvider,
        "generate",
        lambda self, prompt, system_prompt=None: LLMResponse(
            text=generated_json.replace("__TURN_0__", str(session_record.transcript_turns[0].id))
            .replace("__TURN_1__", str(session_record.transcript_turns[1].id))
            .replace("__TURN_2__", str(session_record.transcript_turns[2].id))
            .replace("__TURN_3__", str(session_record.transcript_turns[3].id)),
            provider="test",
            model="test-model",
        ),
    )
    caplog.set_level(logging.INFO, logger="src.services.interview_report_service")

    report = interview_report_service.generate_interview_report(
        db_session,
        interview_session_id=session_record.id,
    )

    persisted_report = db_session.execute(select(InterviewReport)).scalar_one()
    assert report.interview_session_id == session_record.id
    assert persisted_report.interview_template_id == interview_invitation.interview_template_id
    assert persisted_report.report_payload["status"] == "completed"
    assert persisted_report.report_payload["summary"]["candidate_overview"].startswith("Candidate described ownership")
    assert persisted_report.report_payload["summary"]["questions"][0] == {
        "question_key": "system_design",
        "question_text": "Tell me about a backend system you built.",
        "question_transcript_turn_id": str(session_record.transcript_turns[0].id),
        "question_turn_index": 0,
        "answer_text": "I built a queue-based ingestion service with retries, metrics, and idempotent processing for candidate resumes.",
        "answer_transcript_turn_id": str(session_record.transcript_turns[1].id),
        "answer_turn_index": 1,
        "evaluation": "Explained queue-based ingestion architecture with retries and idempotency.",
    }
    assert "## Candidate Overview" in persisted_report.summary_text
    assert "## Recommendation" not in persisted_report.summary_text
    assert "accept" not in persisted_report.summary_text.lower()
    assert "reject" not in persisted_report.summary_text.lower()
    timing_log = next(
        record.getMessage()
        for record in caplog.records
        if "interview_report_llm_completed" in record.getMessage()
    )
    assert f"session_id={session_record.id}" in timing_log
    assert "provider=test" in timing_log
    assert "model=test-model" in timing_log
    assert "prompt_chars=" in timing_log
    assert "response_chars=" in timing_log
    assert "request_ms=" in timing_log
    assert generated_json.strip() not in timing_log


def test_recruiter_can_fetch_interview_report_by_session_id(
    api_client: TestClient,
    db_session: Session,
    interview_invitation: InterviewInvitation,
    monkeypatch,
):
    from src.services import interview_report_service
    from src.services.llm_service import LLMResponse

    session_record = _create_completed_session(db_session, interview_invitation)
    generated_json = """
    {
      "candidate_overview": "Candidate described ownership of backend ingestion and recovery workflows.",
      "questions": [
        {
          "question_key": "system_design",
          "question_text": "Tell me about a backend system you built.",
          "question_transcript_turn_id": "__TURN_0__",
          "question_turn_index": 0,
          "answer_text": "I built a queue-based ingestion service with retries, metrics, and idempotent processing for candidate resumes.",
          "answer_transcript_turn_id": "__TURN_1__",
          "answer_turn_index": 1,
          "evaluation": "Explained queue-based ingestion architecture with retries and idempotency."
        }
      ],
      "overall_summary": "Interview shows backend experience with reliability-focused examples."
    }
    """

    monkeypatch.setattr(
        interview_report_service.LLMProvider,
        "generate",
        lambda self, prompt, system_prompt=None: LLMResponse(
            text=generated_json.replace("__TURN_0__", str(session_record.transcript_turns[0].id)).replace(
                "__TURN_1__", str(session_record.transcript_turns[1].id)
            ),
            provider="test",
            model="test-model",
        ),
    )

    interview_report_service.generate_interview_report(
        db_session,
        interview_session_id=session_record.id,
    )

    response = api_client.get(f"/api/v1/interview-reports/{session_record.id}")
    assert response.status_code == 200
    body = response.json()
    assert body["interview_session_id"] == str(session_record.id)
    assert body["report_payload"]["status"] == "completed"
    assert body["summary_text"].startswith("# Interview Report")


def test_complete_public_interview_session_dispatches_report_generation(
    db_session: Session,
    interview_invitation: InterviewInvitation,
    monkeypatch,
):
    from src.services import interview_session_service

    interview_invitation.status = "in_progress"
    interview_invitation.attempt_count = 1
    db_session.add(interview_invitation)
    db_session.flush()

    session_record = InterviewSession(
        interview_invitation_id=interview_invitation.id,
        provider="fake",
        provider_session_id="provider-session-1",
        status="in_progress",
        started_at=datetime.now(timezone.utc),
    )
    db_session.add(session_record)
    db_session.commit()

    dispatched_session_ids: list[uuid.UUID] = []
    monkeypatch.setattr(
        interview_session_service,
        "enqueue_interview_report_generation",
        lambda db, interview_session_id: dispatched_session_ids.append(interview_session_id),
    )

    response = interview_session_service.complete_public_interview_session(
        db_session,
        token=interview_invitation.public_token,
        body=PublicInterviewCompleteRequest(provider="fake"),
    )

    assert response.session.status == "completed"
    assert dispatched_session_ids == [session_record.id]


def test_complete_public_interview_session_persists_failed_enqueue_state(
    db_session: Session,
    interview_invitation: InterviewInvitation,
    monkeypatch,
):
    from src.services import interview_session_service

    interview_invitation.status = "in_progress"
    interview_invitation.attempt_count = 1
    db_session.add(interview_invitation)
    db_session.flush()

    session_record = InterviewSession(
        interview_invitation_id=interview_invitation.id,
        provider="fake",
        provider_session_id="provider-session-1",
        status="in_progress",
        started_at=datetime.now(timezone.utc),
    )
    db_session.add(session_record)
    db_session.commit()

    monkeypatch.setattr(
        interview_session_service,
        "enqueue_interview_report_generation",
        lambda db, interview_session_id: (_ for _ in ()).throw(RuntimeError("broker unavailable")),
    )

    response = interview_session_service.complete_public_interview_session(
        db_session,
        token=interview_invitation.public_token,
        body=PublicInterviewCompleteRequest(provider="fake"),
    )

    persisted_report = db_session.execute(select(InterviewReport)).scalar_one()
    assert response.session.status == "completed"
    assert persisted_report.report_payload["status"] == "failed"
    assert persisted_report.report_payload["failure"]["stage"] == "enqueue"
    assert persisted_report.report_payload["failure"]["retryable"] is True
    assert "broker unavailable" in persisted_report.report_payload["failure"]["message"]


def test_enqueue_interview_report_commits_pending_before_dispatch(
    db_session: Session,
    interview_invitation: InterviewInvitation,
    monkeypatch,
):
    from src.services import interview_session_service
    import worker.tasks as tasks_module

    session_record = _create_completed_session(db_session, interview_invitation)
    dispatches: list[tuple[list[str], str]] = []

    def record_dispatch(*, args: list[str], task_id: str):
        db_session.expire_all()
        persisted_report = db_session.execute(
            select(InterviewReport).where(
                InterviewReport.interview_session_id == session_record.id
            )
        ).scalar_one_or_none()

        assert persisted_report is not None
        assert persisted_report.report_payload["status"] == "pending"
        assert persisted_report.report_payload["task"]["task_id"] == task_id
        dispatches.append((args, task_id))
        return types.SimpleNamespace(id=task_id)

    monkeypatch.setattr(
        tasks_module,
        "generate_interview_report",
        types.SimpleNamespace(
            delay=lambda interview_session_id: record_dispatch(
                args=[interview_session_id],
                task_id="delay-does-not-provide-task-id",
            ),
            apply_async=record_dispatch,
        ),
        raising=False,
    )

    interview_session_service.enqueue_interview_report_generation(
        db_session,
        session_record.id,
    )

    assert len(dispatches) == 1
    assert dispatches[0][0] == [str(session_record.id)]
    uuid.UUID(dispatches[0][1])


def test_generate_interview_report_rejects_incomplete_summary_payload(
    db_session: Session,
    interview_invitation: InterviewInvitation,
    monkeypatch,
):
    from pydantic import ValidationError

    from src.services import interview_report_service
    from src.services.llm_service import LLMResponse

    session_record = _create_completed_session(db_session, interview_invitation)
    generated_json = """
    {
      "candidate_overview": "Candidate overview only.",
      "questions": [],
      "overall_summary": "Overall summary."
    }
    """
    monkeypatch.setattr(
        interview_report_service.LLMProvider,
        "generate",
        lambda self, prompt, system_prompt=None: LLMResponse(
            text=generated_json,
            provider="test",
            model="test-model",
        ),
    )

    with pytest.raises(ValidationError):
        interview_report_service.generate_interview_report(
            db_session,
            interview_session_id=session_record.id,
        )


def test_generate_interview_report_rejects_mismatched_evidence_links(
    db_session: Session,
    interview_invitation: InterviewInvitation,
    monkeypatch,
):
    from src.services import interview_report_service
    from src.services.llm_service import LLMResponse

    session_record = _create_completed_session(db_session, interview_invitation)
    generated_json = f"""
    {{
      "candidate_overview": "Candidate discussed backend ingestion work.",
      "questions": [
        {{
          "question_key": "hallucinated_question",
          "question_text": "Tell me about a backend system you built.",
          "question_transcript_turn_id": "{session_record.transcript_turns[0].id}",
          "question_turn_index": 0,
          "answer_text": "I built a queue-based ingestion service with retries, metrics, and idempotent processing for candidate resumes.",
          "answer_transcript_turn_id": "{session_record.transcript_turns[1].id}",
          "answer_turn_index": 1,
          "evaluation": "Discussed queue-based processing."
        }}
      ],
      "overall_summary": "Candidate showed relevant backend experience."
    }}
    """
    monkeypatch.setattr(
        interview_report_service.LLMProvider,
        "generate",
        lambda self, prompt, system_prompt=None: LLMResponse(
            text=generated_json,
            provider="test",
            model="test-model",
        ),
    )

    with pytest.raises(ValueError, match="Evidence question_key mismatch"):
        interview_report_service.generate_interview_report(
            db_session,
            interview_session_id=session_record.id,
        )


def test_generate_interview_report_task_marks_permanent_failure_without_retry(monkeypatch):
    from pydantic import ValidationError

    if "celery" not in sys.modules:
        celery_stub = types.ModuleType("celery")

        class _FakeCelery:
            def __init__(self, *args, **kwargs):
                self.conf = types.SimpleNamespace(update=lambda **kwargs: None)

            def autodiscover_tasks(self, *args, **kwargs):
                return None

            def task(self, *args, **kwargs):
                def _decorator(func):
                    func.delay = lambda *f_args, **f_kwargs: func(*f_args, **f_kwargs)
                    return func

                return _decorator

        celery_stub.Celery = _FakeCelery
        sys.modules["celery"] = celery_stub

    sys.modules.pop("worker.tasks", None)
    sys.modules.pop("worker", None)
    tasks = importlib.import_module("worker.tasks")

    error = ValidationError.from_exception_data(
        "InterviewReportSummary",
        [{"type": "missing", "loc": ("questions",), "msg": "Field required", "input": {}}],
    )
    recorded_failures: list[tuple[uuid.UUID, str, bool]] = []
    retry_calls: list[str] = []

    monkeypatch.setattr(
        "src.services.interview_report_service.generate_interview_report_for_session",
        lambda interview_session_id: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        "src.services.interview_report_service.mark_interview_report_pending",
        lambda interview_session_id, *, task_id, retry_count=0, state="queued": {"status": "pending"},
    )
    monkeypatch.setattr(
        "src.services.interview_report_service.mark_interview_report_failure",
        lambda interview_session_id, *, stage, message, retryable, retry_count=0: (
            recorded_failures.append((interview_session_id, stage, retryable)) or {"status": "failed"}
        ),
    )

    result = tasks.generate_interview_report(
        types.SimpleNamespace(
            retry=lambda exc: retry_calls.append(str(exc)),
            request=types.SimpleNamespace(retries=0),
            max_retries=2,
        ),
        str(uuid.uuid4()),
    )

    assert result["status"] == "failed"
    assert recorded_failures[0][1:] == ("generation", False)
    assert retry_calls == []


def test_generate_interview_report_task_calls_service_without_re_marking_pending(monkeypatch):
    if "celery" not in sys.modules:
        celery_stub = types.ModuleType("celery")

        class _FakeCelery:
            def __init__(self, *args, **kwargs):
                self.conf = types.SimpleNamespace(update=lambda **kwargs: None)

            def autodiscover_tasks(self, *args, **kwargs):
                return None

            def task(self, *args, **kwargs):
                def _decorator(func):
                    func.delay = lambda *f_args, **f_kwargs: func(*f_args, **f_kwargs)
                    return func

                return _decorator

        celery_stub.Celery = _FakeCelery
        sys.modules["celery"] = celery_stub

    sys.modules.pop("worker.tasks", None)
    sys.modules.pop("worker", None)
    tasks = importlib.import_module("worker.tasks")

    called_with: list[uuid.UUID] = []
    pending_calls: list[uuid.UUID] = []
    monkeypatch.setattr(
        "src.services.interview_report_service.mark_interview_report_pending",
        lambda interview_session_id, *, task_id, retry_count=0, state="queued": (
            pending_calls.append(interview_session_id) or {"status": "pending"}
        ),
    )
    monkeypatch.setattr(
        "src.services.interview_report_service.generate_interview_report_for_session",
        lambda interview_session_id: called_with.append(interview_session_id) or {"status": "completed"},
    )

    result = tasks.generate_interview_report(types.SimpleNamespace(retry=lambda exc: (_ for _ in ()).throw(exc)), str(uuid.uuid4()))

    assert result == {"status": "completed"}
    assert len(called_with) == 1
    assert pending_calls == []
