from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from src.models.candidate_profile import CandidateProfile
from src.models.enums import (
    CandidateEvaluationStatus,
    ResumeProcessingBatchStatus,
    UploadStatus,
)
from src.models.resume_document import ResumeDocument
from src.models.resume_processing_batch import ResumeProcessingBatch
from src.models.scoring_evaluation import CandidateEvaluation


@dataclass(frozen=True)
class BatchParseTransition:
    batch_id: uuid.UUID
    should_dispatch: bool
    processed_candidate_count: int


def _status_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def create_processing_batch(
    *,
    db: Session,
    job_id: uuid.UUID,
    total_count: int,
) -> ResumeProcessingBatch:
    if total_count < 1:
        raise ValueError("Resume processing batch must contain at least one file")
    batch = ResumeProcessingBatch(job_id=job_id, total_count=total_count)
    db.add(batch)
    db.flush()
    return batch


def reconcile_batch_after_parse(
    db: Session,
    processing_batch_id: uuid.UUID,
) -> BatchParseTransition:
    batch = (
        db.query(ResumeProcessingBatch)
        .filter(ResumeProcessingBatch.id == processing_batch_id)
        .with_for_update()
        .one()
    )
    statuses = [
        _status_value(status)
        for (status,) in (
            db.query(ResumeDocument.upload_status)
            .filter(ResumeDocument.processing_batch_id == batch.id)
            .all()
        )
    ]
    batch.processed_count = statuses.count(UploadStatus.PROCESSED.value)
    batch.failed_count = statuses.count(UploadStatus.FAILED.value)
    batch.terminal_count = batch.processed_count + batch.failed_count

    should_dispatch = (
        _status_value(batch.status) == ResumeProcessingBatchStatus.PARSING.value
        and batch.terminal_count == batch.total_count
        and batch.processed_count > 0
    )
    if should_dispatch:
        batch.status = ResumeProcessingBatchStatus.EVALUATION_PENDING
    elif (
        _status_value(batch.status) == ResumeProcessingBatchStatus.PARSING.value
        and batch.terminal_count == batch.total_count
        and batch.processed_count == 0
    ):
        batch.status = ResumeProcessingBatchStatus.FAILED
        batch.completed_at = datetime.now(timezone.utc)

    db.commit()
    return BatchParseTransition(
        batch_id=batch.id,
        should_dispatch=should_dispatch,
        processed_candidate_count=batch.processed_count,
    )


def claim_evaluation_dispatch(
    db: Session,
    processing_batch_id: uuid.UUID,
    *,
    stale_after_seconds: int,
    evaluation_stale_after_seconds: int,
    now: datetime | None = None,
) -> bool:
    current_time = _as_utc(now or datetime.now(timezone.utc))
    batch = (
        db.query(ResumeProcessingBatch)
        .filter(ResumeProcessingBatch.id == processing_batch_id)
        .with_for_update()
        .one()
    )
    status = _status_value(batch.status)
    if status == ResumeProcessingBatchStatus.EVALUATING.value:
        evaluation_stale_before = current_time - timedelta(
            seconds=max(1, evaluation_stale_after_seconds)
        )
        if _as_utc(batch.updated_at) > evaluation_stale_before:
            db.rollback()
            return False
        # The worker may have been restarted after it set the batch to
        # EVALUATING.  Make the task dispatchable again rather than leaving
        # every candidate in RUNNING indefinitely.
        batch.status = ResumeProcessingBatchStatus.EVALUATION_PENDING
        batch.evaluation_task_id = None
        batch.evaluation_dispatch_attempted_at = None
    elif status != ResumeProcessingBatchStatus.EVALUATION_PENDING.value:
        db.rollback()
        return False

    attempted_at = batch.evaluation_dispatch_attempted_at
    if attempted_at is not None:
        stale_before = current_time - timedelta(seconds=max(1, stale_after_seconds))
        if _as_utc(attempted_at) > stale_before:
            db.rollback()
            return False

    batch.evaluation_dispatch_attempted_at = current_time
    db.commit()
    return True


def record_evaluation_task_id(
    db: Session,
    processing_batch_id: uuid.UUID,
    task_id: str,
) -> None:
    batch = (
        db.query(ResumeProcessingBatch)
        .filter(ResumeProcessingBatch.id == processing_batch_id)
        .with_for_update()
        .one()
    )
    if _status_value(batch.status) == ResumeProcessingBatchStatus.EVALUATION_PENDING.value:
        batch.evaluation_task_id = task_id
    db.commit()


def list_recoverable_evaluation_batches(
    db: Session,
    *,
    stale_after_seconds: int,
    evaluation_stale_after_seconds: int,
    now: datetime | None = None,
) -> list[uuid.UUID]:
    current_time = _as_utc(now or datetime.now(timezone.utc))
    stale_before = current_time - timedelta(seconds=max(1, stale_after_seconds))
    evaluation_stale_before = current_time - timedelta(
        seconds=max(1, evaluation_stale_after_seconds)
    )
    batches = db.query(ResumeProcessingBatch).filter(
        ResumeProcessingBatch.status.in_(
            [
                ResumeProcessingBatchStatus.EVALUATION_PENDING,
                ResumeProcessingBatchStatus.EVALUATING,
            ]
        )
    )
    return [
        batch.id
        for batch in batches
        if (
            _status_value(batch.status)
            == ResumeProcessingBatchStatus.EVALUATION_PENDING.value
            and (
                batch.evaluation_dispatch_attempted_at is None
                or _as_utc(batch.evaluation_dispatch_attempted_at) <= stale_before
            )
        )
        or (
            _status_value(batch.status) == ResumeProcessingBatchStatus.EVALUATING.value
            and _as_utc(batch.updated_at) <= evaluation_stale_before
        )
    ]


def mark_processing_batch_failed(
    db: Session,
    processing_batch_id: uuid.UUID,
    *,
    error_message: str,
) -> None:
    batch = (
        db.query(ResumeProcessingBatch)
        .filter(ResumeProcessingBatch.id == processing_batch_id)
        .with_for_update()
        .one()
    )
    profile_ids = [
        profile_id
        for (profile_id,) in (
            db.query(CandidateProfile.id)
            .join(ResumeDocument, ResumeDocument.id == CandidateProfile.resume_document_id)
            .filter(ResumeDocument.processing_batch_id == batch.id)
            .all()
        )
    ]
    if profile_ids:
        evaluations = (
            db.query(CandidateEvaluation)
            .filter(
                CandidateEvaluation.candidate_profile_id.in_(profile_ids),
                CandidateEvaluation.status.in_(
                    [
                        CandidateEvaluationStatus.PENDING,
                        CandidateEvaluationStatus.RUNNING,
                    ]
                ),
            )
            .all()
        )
        for evaluation in evaluations:
            evaluation.status = CandidateEvaluationStatus.FAILED
            evaluation.error_message = error_message
    batch.status = ResumeProcessingBatchStatus.FAILED
    batch.completed_at = datetime.now(timezone.utc)
    db.commit()
