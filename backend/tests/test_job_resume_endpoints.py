import os
import sys
import types
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "minio" not in sys.modules:
    minio_stub = types.ModuleType("minio")
    minio_error_stub = types.ModuleType("minio.error")

    class Minio:
        def __init__(self, *args, **kwargs):
            pass

    class S3Error(Exception):
        code = "S3Error"

    minio_stub.Minio = Minio
    minio_error_stub.S3Error = S3Error
    sys.modules["minio"] = minio_stub
    sys.modules["minio.error"] = minio_error_stub

if "src.services.resume_service" not in sys.modules:
    resume_service_stub = types.ModuleType("src.services.resume_service")

    def _resume_to_dict(resume):
        return {
            "id": str(resume.id),
            "job_id": str(resume.job_id),
            "original_file_name": resume.original_file_name,
            "storage_uri": resume.storage_uri,
            "upload_status": (
                resume.upload_status.value
                if hasattr(resume.upload_status, "value")
                else resume.upload_status
            ),
            "duplicate_group_key": resume.duplicate_group_key,
            "uploaded_by_user_id": str(resume.uploaded_by_user_id),
            "uploaded_at": resume.uploaded_at.isoformat() if resume.uploaded_at else None,
            "processed_at": resume.processed_at.isoformat() if resume.processed_at else None,
            "retention_expires_at": (
                resume.retention_expires_at.isoformat()
                if resume.retention_expires_at
                else None
            ),
        }

    resume_service_stub._resume_to_dict = _resume_to_dict
    resume_service_stub._normalize_location_name = lambda value: value
    resume_service_stub.create_resume_document = lambda *args, **kwargs: None
    resume_service_stub.parse_pdf_to_sections = lambda *args, **kwargs: {}
    sys.modules["src.services.resume_service"] = resume_service_stub

from src.api.v1.endpoints import jobs as jobs_module  # noqa: E402
from src.api.v1.endpoints.jobs import delete_job_resume, list_job_resumes  # noqa: E402
from src.models.base import Base  # noqa: E402
from src.models.candidate_profile import CandidateProfile  # noqa: E402
from src.models.deps import get_current_user, get_db  # noqa: E402
from src.models.enums import ProfileStatus, UploadStatus, UserStatus  # noqa: E402
from src.models.job import Job  # noqa: E402
from src.models.resume_document import ResumeDocument  # noqa: E402
from src.models.resume_processing_batch import ResumeProcessingBatch  # noqa: E402
from src.models.user_account import UserAccount  # noqa: E402


def _create_test_tables(engine):
    tables = [
        Base.metadata.tables["user_accounts"],
        Base.metadata.tables["jobs"],
        Base.metadata.tables["resume_processing_batches"],
        Base.metadata.tables["resume_documents"],
        Base.metadata.tables["candidate_profiles"],
    ]
    Base.metadata.create_all(engine, tables=tables)


@contextmanager
def _client_for_user(db: Session, current_user: UserAccount):
    test_app = FastAPI()
    test_app.include_router(jobs_module.router, prefix="/api/v1/jobs")

    def _override_db():
        yield db

    test_app.dependency_overrides[get_db] = _override_db
    test_app.dependency_overrides[get_current_user] = lambda: current_user

    client = TestClient(test_app, follow_redirects=False)
    try:
        yield client
    finally:
        test_app.dependency_overrides.clear()


def test_list_job_resumes_includes_candidate_and_uploader_display_names():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    _create_test_tables(engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    db: Session = session_factory()
    try:
        owner = UserAccount(
            email="owner@example.com",
            display_name="Hiring Manager",
            password_hash=None,
            status=UserStatus.ACTIVE,
        )
        uploader = UserAccount(
            email="recruiter@example.com",
            display_name="Recruiter One",
            password_hash=None,
            status=UserStatus.ACTIVE,
        )
        db.add_all([owner, uploader])
        db.flush()

        job = Job(owner_user_id=owner.id, title="Platform Engineer", status="active")
        db.add(job)
        db.flush()

        resume = ResumeDocument(
            original_file_name="nguyen_van_a_cv.pdf",
            storage_uri="s3://bucket/resumes/nguyen_van_a_cv.pdf",
            upload_status=UploadStatus.PROCESSED,
            job_id=job.id,
            uploaded_by_user_id=uploader.id,
            retention_expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
        )
        db.add(resume)
        db.flush()

        candidate = CandidateProfile(
            resume_document_id=resume.id,
            full_name="Nguyen Van A",
            submitted_full_name="Nguyen Van A",
            email="candidate@example.com",
            profile_status=ProfileStatus.REVIEWED,
        )
        db.add(candidate)
        db.commit()

        response = list_job_resumes(job.id, None, 50, 0, db, owner)

        assert response.total == 1
        assert response.items[0].candidate_profile_id == str(candidate.id)
        assert response.items[0].candidate_display_name == "Nguyen Van A"
        assert response.items[0].uploader_display_name == "Recruiter One"

        db.add(
            ResumeDocument(
                original_file_name="second_cv.pdf",
                storage_uri="s3://bucket/resumes/second_cv.pdf",
                upload_status=UploadStatus.PROCESSED,
                job_id=job.id,
                uploaded_by_user_id=uploader.id,
                retention_expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
            )
        )
        db.commit()

        paginated_response = list_job_resumes(job.id, None, 1, 0, db, owner)
        assert len(paginated_response.items) == 1
        assert paginated_response.total == 2
    finally:
        db.close()


def test_delete_job_resume_is_idempotent_when_resume_is_already_missing():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    _create_test_tables(engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    db: Session = session_factory()
    try:
        owner = UserAccount(
            email="owner@example.com",
            display_name="Hiring Manager",
            password_hash=None,
            status=UserStatus.ACTIVE,
        )
        db.add(owner)
        db.flush()

        job = Job(owner_user_id=owner.id, title="Platform Engineer", status="active")
        db.add(job)
        db.commit()

        missing_resume_id = uuid.uuid4()

        response = delete_job_resume(job.id, missing_resume_id, db, owner)

        assert response == {"deleted": True, "resume_id": str(missing_resume_id)}
    finally:
        db.close()


def test_upload_job_resumes_accepts_multiple_files_in_one_request():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    _create_test_tables(engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    db: Session = session_factory()
    try:
        owner = UserAccount(
            email="owner@example.com",
            display_name="Hiring Manager",
            password_hash=None,
            status=UserStatus.ACTIVE,
        )
        db.add(owner)
        db.flush()

        job = Job(owner_user_id=owner.id, title="Platform Engineer", status="active")
        db.add(job)
        db.commit()
        db.refresh(owner)
        db.refresh(job)

        storage = MagicMock()
        storage.upload_bytes.side_effect = [
            "s3://bucket/resumes/alice.pdf",
            "s3://bucket/resumes/bob.pdf",
        ]

        def _create_resume_document(
            *,
            db,
            storage_uri,
            original_file_name,
            job_id,
            uploaded_by_user_id,
            processing_batch_id,
        ):
            resume = ResumeDocument(
                original_file_name=original_file_name,
                storage_uri=storage_uri,
                upload_status=UploadStatus.UPLOADED.value,
                job_id=job_id,
                uploaded_by_user_id=uploaded_by_user_id,
                retention_expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
                processing_batch_id=processing_batch_id,
            )
            db.add(resume)
            db.commit()
            db.refresh(resume)
            return resume

        queued_tasks = [
            types.SimpleNamespace(id="task-1"),
            types.SimpleNamespace(id="task-2"),
        ]

        with _client_for_user(db, owner) as client, patch(
            "src.api.v1.endpoints.jobs.get_object_storage",
            return_value=storage,
        ), patch(
            "src.api.v1.endpoints.jobs.create_resume_document",
            side_effect=_create_resume_document,
        ), patch(
            "src.api.v1.endpoints.jobs.process_resume.delay",
            side_effect=queued_tasks,
        ) as enqueue_resume:
            response = client.post(
                f"/api/v1/jobs/{job.id}/resumes",
                files=[
                    ("files", ("alice.pdf", b"%PDF-1.4 alice", "application/pdf")),
                    ("files", ("bob.pdf", b"%PDF-1.4 bob", "application/pdf")),
                ],
            )

        assert response.status_code == 202
        payload = response.json()

        assert payload["total_files"] == 2
        assert payload["queued_files"] == 2
        processing_batch_id = uuid.UUID(payload["processing_batch_id"])
        assert [item["file_name"] for item in payload["items"]] == ["alice.pdf", "bob.pdf"]
        assert [item["status"] for item in payload["items"]] == ["queued", "queued"]
        assert [item["task_id"] for item in payload["items"]] == ["task-1", "task-2"]

        resumes = (
            db.query(ResumeDocument)
            .filter(ResumeDocument.job_id == job.id)
            .order_by(ResumeDocument.original_file_name.asc())
            .all()
        )
        assert len(resumes) == 2
        assert {resume.processing_batch_id for resume in resumes} == {processing_batch_id}
        assert [resume.original_file_name for resume in resumes] == ["alice.pdf", "bob.pdf"]
        assert [resume.upload_status for resume in resumes] == [
            UploadStatus.UPLOADED.value,
            UploadStatus.UPLOADED.value,
        ]

        assert enqueue_resume.call_count == 2
        assert enqueue_resume.call_args_list == [
            call(str(resumes[0].id), processing_batch_id=str(processing_batch_id)),
            call(str(resumes[1].id), processing_batch_id=str(processing_batch_id)),
        ]
    finally:
        db.close()
