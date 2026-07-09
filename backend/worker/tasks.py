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
            candidate_profile_id = result.get("candidate_profile_id")
            if candidate_profile_id:
                from src.models.session import SessionLocal
                from src.services.candidate_evaluation_service import queue_candidate_evaluation_for_current_jd

                with SessionLocal() as db:
                    queued = queue_candidate_evaluation_for_current_jd(
                        db=db,
                        candidate_profile_id=uuid.UUID(str(candidate_profile_id)),
                    )
                if not queued:
                    logger.info(
                        "candidate evaluation not queued for %s because no active JD or current evaluation already exists",
                        candidate_profile_id,
                    )
        return result
    except Exception as exc:
        logger.exception("process_resume crashed for %s", resume_document_id)
        raise self.retry(exc=exc)


@celery_app.task(
    name="worker.tasks.evaluate_candidate",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
)
def evaluate_candidate(self, candidate_profile_id: str):
    logger.info("evaluate_candidate started for %s", candidate_profile_id)
    try:
        from src.models.session import SessionLocal
        from src.services.candidate_evaluation_service import evaluate_candidate_for_current_jd

        with SessionLocal() as db:
            evaluation = evaluate_candidate_for_current_jd(
                db=db,
                candidate_profile_id=uuid.UUID(candidate_profile_id),
            )
            logger.info(
                "evaluate_candidate finished for %s with status %s",
                candidate_profile_id,
                evaluation.status,
            )
            return {
                "candidate_profile_id": candidate_profile_id,
                "evaluation_id": str(evaluation.id),
                "status": str(evaluation.status),
            }
    except Exception as exc:
        logger.exception("evaluate_candidate crashed for %s", candidate_profile_id)
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


@celery_app.task(
    name="worker.tasks.send_interview_invitation_email",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
)
def send_interview_invitation_email(self, invitation_id: str):
    from datetime import datetime, timezone

    from sqlalchemy import select

    from src.models.interview_invitation import InterviewInvitation
    from src.models.oauth_identity import OAuthIdentity
    from src.models.session import SessionLocal
    from src.models.user_account import UserAccount
    from src.services.email_templates import build_interview_invitation_email
    from src.services.interview_invitation_service import build_interview_public_url
    from src.services.mail_service import send_email

    db = SessionLocal()
    try:
        try:
            invitation_lookup_key = uuid.UUID(invitation_id)
        except (TypeError, ValueError, AttributeError):
            invitation_lookup_key = invitation_id

        invitation = db.get(InterviewInvitation, invitation_lookup_key)
        if invitation is None:
            return {"sent": False, "reason": "invitation_not_found"}
        if invitation.sent_at is not None:
            return {"sent": True, "reason": "already_sent"}

        candidate_email = invitation.candidate_profile.email if invitation.candidate_profile else None
        if not candidate_email:
            invitation.status = "email_failed"
            db.commit()
            return {"sent": False, "reason": "candidate_email_missing"}

        user = db.get(UserAccount, invitation.sent_by_user_id)
        if user is None:
            invitation.status = "email_failed"
            db.commit()
            return {"sent": False, "reason": "sender_not_found"}

        identity = (
            db.execute(
                select(OAuthIdentity).where(
                    OAuthIdentity.user_id == user.id,
                    OAuthIdentity.provider == "google",
                )
            )
            .scalar_one_or_none()
        )
        if identity is None:
            invitation.status = "email_failed"
            db.commit()
            return {"sent": False, "reason": "google_identity_missing"}

        expires_at_text = invitation.expires_at.isoformat() if invitation.expires_at else None
        subject, body = build_interview_invitation_email(
            candidate_name=invitation.candidate_profile.full_name,
            job_title=invitation.job.title,
            public_url=build_interview_public_url(invitation.public_token),
            expires_at_text=expires_at_text,
        )
        result = send_email(
            sender=user.email,
            to_email=candidate_email,
            subject=subject,
            body_text=body,
            identity=identity,
        )
        invitation.sent_at = datetime.now(timezone.utc)
        invitation.status = "sent"
        db.commit()
        return {"sent": True, "gmail_message_id": result.get("id")}
    except Exception as exc:
        db.rollback()
        logger.exception("send_interview_invitation_email crashed for %s", invitation_id)
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(
    name="worker.tasks.send_outreach_email",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
)
def send_outreach_email(self, message_id: str):
    from datetime import datetime, timezone

    from sqlalchemy import select

    from src.models.enums import SentStatus
    from src.models.oauth_identity import GMAIL_SEND_SCOPE, OAuthIdentity
    from src.models.outreach import OutreachMessage
    from src.models.session import SessionLocal
    from src.models.user_account import UserAccount
    from src.services.email_templates import build_outreach_email
    from src.services.mail_service import send_email

    db = SessionLocal()
    try:
        try:
            message_lookup_key = uuid.UUID(message_id)
        except (TypeError, ValueError, AttributeError):
            message_lookup_key = message_id

        message = db.get(OutreachMessage, message_lookup_key)
        if message is None:
            return {"sent": False, "reason": "message_not_found"}
        if message.sent_status == SentStatus.SENT:
            return {"sent": True, "reason": "already_sent"}

        candidate_email = message.candidate_profile.email if message.candidate_profile else None
        if not candidate_email:
            message.sent_status = SentStatus.FAILED
            db.commit()
            return {"sent": False, "reason": "candidate_email_missing"}

        user = db.get(UserAccount, message.created_by_user_id)
        identity = (
            db.execute(
                select(OAuthIdentity).where(
                    OAuthIdentity.user_id == message.created_by_user_id,
                    OAuthIdentity.provider == "google",
                )
            )
            .scalar_one_or_none()
        )
        if user is None:
            message.sent_status = SentStatus.FAILED
            db.commit()
            return {"sent": False, "reason": "sender_not_found"}
        if (
            identity is None
            or not identity.refresh_token_encrypted
            or not identity.has_scope(GMAIL_SEND_SCOPE)
        ):
            return {"sent": False, "reason": "gmail_not_connected"}

        subject, body_text, body_html = build_outreach_email(
            subject=message.subject,
            body_text=message.body_text,
            body_html=message.body_html,
        )
        send_email(
            sender=user.email,
            to_email=candidate_email,
            subject=subject,
            body_text=body_text,
            body_html=body_html,
            identity=identity,
        )
        message.sent_status = SentStatus.SENT
        message.sent_at = datetime.now(timezone.utc)
        db.commit()
        return {"sent": True}
    except Exception as exc:
        db.rollback()
        logger.exception("send_outreach_email crashed for %s", message_id)
        raise self.retry(exc=exc)
    finally:
        db.close()


# Backward-compatible stubs ------------------------------------------------


@celery_app.task(name="worker.tasks.process_document")
def process_document(document_id: int):
    logger.info("Legacy process_document called for %s", document_id)
    return True


@celery_app.task(name="worker.tasks.cleanup_logs")
def cleanup_logs():
    logger.info("cleanup_logs executed")
    return True
