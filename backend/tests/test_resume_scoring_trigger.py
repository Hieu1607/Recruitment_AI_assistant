import importlib
import sys
import types
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.modules.pop("worker.tasks", None)
sys.modules.pop("worker", None)

worker_tasks = importlib.import_module("worker.tasks")  # noqa: E402

from src.models.base import Base  # noqa: E402
from src.models.candidate_profile import CandidateProfile  # noqa: E402
from src.models.enums import ProfileStatus, UploadStatus, UserStatus  # noqa: E402
from src.models.job import Job  # noqa: E402
from src.models.job_matching import JobDescription  # noqa: E402
from src.models.resume_document import ResumeDocument  # noqa: E402
from src.models.scoring_evaluation import CandidateEvaluation  # noqa: E402
from src.models.user_account import UserAccount  # noqa: E402


def _create_test_tables(engine) -> None:
    tables = [
        Base.metadata.tables["user_accounts"],
        Base.metadata.tables["jobs"],
        Base.metadata.tables["resume_documents"],
        Base.metadata.tables["candidate_profiles"],
        Base.metadata.tables["job_descriptions"],
        Base.metadata.tables["candidate_evaluations"],
    ]
    Base.metadata.create_all(engine, tables=tables)


def test_process_resume_enqueues_candidate_evaluation_after_success(monkeypatch):
    resume_document_id = str(uuid.uuid4())
    candidate_profile_id = str(uuid.uuid4())
    queued: list[str] = []

    monkeypatch.setattr(
        "src.services.resume_service.process_single_resume",
        lambda *_args, **_kwargs: {
            "status": "completed",
            "candidate_profile_id": candidate_profile_id,
            "extraction_mode": "ocr",
        },
    )
    class _DummySession:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("src.models.session.SessionLocal", lambda: _DummySession())
    monkeypatch.setattr(
        "src.services.candidate_evaluation_service.queue_candidate_evaluation_for_current_jd",
        lambda db, candidate_profile_id: queued.append(str(candidate_profile_id)) or True,
    )

    result = worker_tasks.process_resume.run(resume_document_id)

    assert result["status"] == "completed"
    assert queued == [candidate_profile_id]


def test_process_resume_does_not_enqueue_candidate_evaluation_when_parsing_failed(monkeypatch):
    resume_document_id = str(uuid.uuid4())
    queued: list[str] = []

    monkeypatch.setattr(
        "src.services.resume_service.process_single_resume",
        lambda *_args, **_kwargs: {
            "status": "failed",
            "error": "parse failed",
            "candidate_profile_id": str(uuid.uuid4()),
        },
    )
    monkeypatch.setattr(
        worker_tasks,
        "evaluate_candidate",
        types.SimpleNamespace(
            delay=lambda profile_id: queued.append(profile_id) or types.SimpleNamespace(id="eval-task-id")
        ),
        raising=False,
    )

    result = worker_tasks.process_resume.run(resume_document_id)

    assert result["status"] == "failed"
    assert queued == []


def test_process_resume_creates_pending_evaluation_before_worker_runs(monkeypatch):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    _create_test_tables(engine)
    with Session(engine) as session:
        owner = UserAccount(
            email="owner@example.com",
            display_name="Owner",
            password_hash=None,
            status=UserStatus.ACTIVE,
        )
        session.add(owner)
        session.flush()

        job = Job(owner_user_id=owner.id, title="AI Engineer", status="active")
        session.add(job)
        session.flush()

        resume = ResumeDocument(
            original_file_name="candidate.pdf",
            storage_uri="s3://bucket/candidate.pdf",
            upload_status=UploadStatus.PROCESSED,
            job_id=job.id,
            uploaded_by_user_id=owner.id,
            retention_expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
        )
        session.add(resume)
        session.flush()

        candidate = CandidateProfile(
            resume_document_id=resume.id,
            full_name="Candidate One",
            experience_years=Decimal("6.0"),
            profile_status=ProfileStatus.REVIEWED,
        )
        session.add(candidate)

        jd = JobDescription(
            job_id=job.id,
            title="AI Engineer JD",
            jd_text="Need Python",
            hidden_text="Prefer RAG",
            created_by_user_id=owner.id,
            is_active=True,
        )
        session.add(jd)
        session.commit()

        candidate_profile_id = str(candidate.id)
        resume_document_id = str(resume.id)
        queued: list[str] = []

        monkeypatch.setattr(
            "src.services.resume_service.process_single_resume",
            lambda *_args, **_kwargs: {
                "status": "completed",
                "candidate_profile_id": candidate_profile_id,
                "extraction_mode": "ocr",
            },
        )
        monkeypatch.setattr(
            worker_tasks,
            "evaluate_candidate",
            types.SimpleNamespace(
                delay=lambda profile_id: queued.append(profile_id) or types.SimpleNamespace(id="eval-task-id")
            ),
            raising=False,
        )
        monkeypatch.setattr("src.models.session.SessionLocal", lambda: Session(engine))

        result = worker_tasks.process_resume.run(resume_document_id)

        session.expire_all()
        evaluation = (
            session.query(CandidateEvaluation)
            .filter(CandidateEvaluation.candidate_profile_id == candidate.id)
            .first()
        )

        assert result["status"] == "completed"
        assert queued == [candidate_profile_id]
        assert evaluation is not None
        assert evaluation.status == "pending"
