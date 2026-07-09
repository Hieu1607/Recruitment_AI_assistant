import sys
from datetime import datetime, timedelta, timezone
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
from src.services.resume_batch_service import (  # noqa: E402
    claim_evaluation_dispatch,
    reconcile_batch_after_parse,
)


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


def _create_batch(db, statuses):
    owner = UserAccount(
        email=f"owner-{len(statuses)}-{id(statuses)}@example.com",
        display_name="Batch Owner",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    db.add(owner)
    db.flush()
    job = Job(owner_user_id=owner.id, title="AI Engineer", status="active")
    db.add(job)
    db.flush()
    batch = ResumeProcessingBatch(job_id=job.id, total_count=len(statuses))
    db.add(batch)
    db.flush()
    for index, status in enumerate(statuses):
        db.add(
            ResumeDocument(
                original_file_name=f"candidate-{index}.pdf",
                storage_uri=f"s3://bucket/candidate-{index}.pdf",
                upload_status=status,
                job_id=job.id,
                uploaded_by_user_id=owner.id,
                retention_expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
                processing_batch_id=batch.id,
            )
        )
    db.commit()
    return batch


def test_reconcile_moves_terminal_batch_to_evaluation_pending(db):
    batch = _create_batch(
        db,
        [UploadStatus.PROCESSED, UploadStatus.PROCESSED, UploadStatus.FAILED],
    )

    transition = reconcile_batch_after_parse(db, batch.id)

    db.refresh(batch)
    assert transition.should_dispatch is True
    assert transition.processed_candidate_count == 2
    assert batch.status == ResumeProcessingBatchStatus.EVALUATION_PENDING
    assert batch.terminal_count == 3
    assert batch.processed_count == 2
    assert batch.failed_count == 1


def test_reconcile_duplicate_completion_does_not_reopen_batch(db):
    batch = _create_batch(db, [UploadStatus.PROCESSED, UploadStatus.PROCESSED])

    first = reconcile_batch_after_parse(db, batch.id)
    second = reconcile_batch_after_parse(db, batch.id)

    assert first.should_dispatch is True
    assert second.should_dispatch is False


def test_reconcile_all_failed_batch_finishes_without_evaluation(db):
    batch = _create_batch(db, [UploadStatus.FAILED, UploadStatus.FAILED])

    transition = reconcile_batch_after_parse(db, batch.id)

    db.refresh(batch)
    assert transition.should_dispatch is False
    assert batch.status == ResumeProcessingBatchStatus.FAILED
    assert batch.completed_at is not None


def test_claim_dispatch_allows_stale_recovery(db):
    batch = _create_batch(db, [UploadStatus.PROCESSED])
    reconcile_batch_after_parse(db, batch.id)
    now = datetime(2026, 7, 9, 12, 0, tzinfo=timezone.utc)

    assert claim_evaluation_dispatch(
        db,
        batch.id,
        stale_after_seconds=15,
        now=now,
    )
    assert not claim_evaluation_dispatch(
        db,
        batch.id,
        stale_after_seconds=15,
        now=now + timedelta(seconds=14),
    )
    assert claim_evaluation_dispatch(
        db,
        batch.id,
        stale_after_seconds=15,
        now=now + timedelta(seconds=16),
    )
