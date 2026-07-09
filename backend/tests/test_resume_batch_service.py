import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.base import Base  # noqa: E402
from src.models.enums import ResumeProcessingBatchStatus, UploadStatus, UserStatus  # noqa: E402
from src.models.job import Job  # noqa: E402
from src.models.resume_document import ResumeDocument  # noqa: E402
from src.models.resume_processing_batch import ResumeProcessingBatch  # noqa: E402
from src.models.user_account import UserAccount  # noqa: E402


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    tables = [
        Base.metadata.tables["user_accounts"],
        Base.metadata.tables["jobs"],
        Base.metadata.tables["resume_processing_batches"],
        Base.metadata.tables["resume_documents"],
    ]
    Base.metadata.create_all(engine, tables=tables)
    with Session(engine) as session:
        yield session


def test_resume_processing_batch_groups_resume_documents(db):
    owner = UserAccount(
        email="batch-owner@example.com",
        display_name="Batch Owner",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    db.add(owner)
    db.flush()
    job = Job(owner_user_id=owner.id, title="AI Engineer", status="active")
    db.add(job)
    db.flush()

    batch = ResumeProcessingBatch(
        job_id=job.id,
        total_count=2,
        status=ResumeProcessingBatchStatus.PARSING,
    )
    db.add(batch)
    db.flush()

    resumes = [
        ResumeDocument(
            original_file_name=f"candidate-{index}.pdf",
            storage_uri=f"s3://bucket/candidate-{index}.pdf",
            upload_status=UploadStatus.UPLOADED,
            job_id=job.id,
            uploaded_by_user_id=owner.id,
            retention_expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
            processing_batch_id=batch.id,
        )
        for index in range(2)
    ]
    db.add_all(resumes)
    db.commit()
    db.refresh(batch)

    assert batch.status == ResumeProcessingBatchStatus.PARSING
    assert batch.total_count == 2
    assert {resume.id for resume in batch.resume_documents} == {
        resume.id for resume in resumes
    }
