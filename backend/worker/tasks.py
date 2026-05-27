import logging
import sys
import uuid
from pathlib import Path

from worker.celery_app import celery_app

logger = logging.getLogger(__name__)
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


@celery_app.task(
    name="worker.tasks.process_resume",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
)
def process_resume(
    self,
    resume_document_id: str,
    submitted_full_name: str | None = None,
    submitted_email: str | None = None,
):
    """Celery task: parse a single uploaded resume in the background.

    Accepts the ResumeDocument UUID as a string.  Opens its own DB session,
    downloads the PDF from MinIO, runs OCR / LLM extraction, and persists
    CandidateProfile + ExtractionTrace.
    """
    logger.info("process_resume started for %s", resume_document_id)
    try:
        from src.services.resume_service import process_single_resume

        result = process_single_resume(
            uuid.UUID(resume_document_id),
            submitted_full_name=submitted_full_name,
            submitted_email=submitted_email,
        )
        if result.get("status") == "failed":
            logger.warning(
                "process_resume failed for %s: %s",
                resume_document_id,
                result.get("error"),
            )
        else:
            logger.info(
                "process_resume succeeded for %s via %s",
                resume_document_id,
                result.get("extraction_mode", "unknown"),
            )
        return result
    except Exception as exc:
        logger.exception("process_resume crashed for %s", resume_document_id)
        raise self.retry(exc=exc)


@celery_app.task(
    name="worker.tasks.generate_interview_report",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
)
def generate_interview_report(self, interview_session_id: str):
    logger.info("generate_interview_report started for %s", interview_session_id)
    interview_session_uuid = uuid.UUID(interview_session_id)
    try:
        from src.services.interview_report_service import (
            generate_interview_report_for_session,
            is_permanent_report_error,
            mark_interview_report_failure,
            mark_interview_report_pending,
        )

        mark_interview_report_pending(
            interview_session_id=interview_session_uuid,
            task_id=getattr(getattr(self, "request", None), "id", None),
            retry_count=getattr(getattr(self, "request", None), "retries", 0),
            state="processing",
        )
        result = generate_interview_report_for_session(interview_session_uuid)
        logger.info("generate_interview_report succeeded for %s", interview_session_id)
        return result
    except Exception as exc:
        logger.exception("generate_interview_report crashed for %s", interview_session_id)
        retry_count = getattr(getattr(self, "request", None), "retries", 0)
        if "is_permanent_report_error" in locals() and is_permanent_report_error(exc):
            return mark_interview_report_failure(
                interview_session_uuid,
                stage="generation",
                message=str(exc),
                retryable=False,
                retry_count=retry_count,
            )
        if retry_count >= getattr(self, "max_retries", 0):
            return mark_interview_report_failure(
                interview_session_uuid,
                stage="generation",
                message=str(exc),
                retryable=False,
                retry_count=retry_count,
            )
        if "mark_interview_report_failure" in locals():
            mark_interview_report_failure(
                interview_session_uuid,
                stage="generation",
                message=str(exc),
                retryable=True,
                retry_count=retry_count,
            )
        raise self.retry(exc=exc)


# Backward-compatible stubs ------------------------------------------------


@celery_app.task(name="worker.tasks.process_document")
def process_document(document_id: int):
    logger.info("Legacy process_document called for %s", document_id)
    return True


@celery_app.task(name="worker.tasks.cleanup_logs")
def cleanup_logs():
    logger.info("cleanup_logs executed")
    return True
