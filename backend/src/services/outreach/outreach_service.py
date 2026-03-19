from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from src.api.errors import AppError
from src.models.candidate import CandidateProfile
from src.models.engagement import ApprovalStatus, ContentSource, OutreachMessage, SentStatus
from src.services.outreach.email_sender import EmailSendRequest, EmailSendError, send_email


@dataclass
class OutreachDraftInput:
    candidate_id: uuid.UUID
    created_by_user_id: uuid.UUID
    source_type: str
    template_id: str | None
    intent: str | None


class OutreachService:
    def create_draft(self, session: Session, payload: OutreachDraftInput) -> OutreachMessage:
        candidate = session.get(CandidateProfile, payload.candidate_id)
        if not candidate:
            raise AppError(code="candidate_not_found", message="Candidate not found", status_code=404)

        source = ContentSource.AI_DRAFT if payload.source_type == "ai_draft" else ContentSource.TEMPLATE
        subject = self._build_subject(candidate.full_name)
        body = self._build_body(
            candidate_name=candidate.full_name,
            role_hint=candidate.current_job_title,
            source_type=payload.source_type,
            intent=payload.intent,
        )

        message = OutreachMessage(
            candidate_profile_id=payload.candidate_id,
            created_by_user_id=payload.created_by_user_id,
            content_source=source,
            subject=subject,
            body=body,
            approval_status=ApprovalStatus.DRAFT,
            sent_status=SentStatus.NOT_SENT,
        )
        session.add(message)
        session.flush()
        return message

    def approve_and_send(
        self,
        session: Session,
        message_id: uuid.UUID,
        approver_user_id: uuid.UUID,
    ) -> OutreachMessage:
        message = session.get(OutreachMessage, message_id)
        if not message:
            raise AppError(code="outreach_not_found", message="Outreach message not found", status_code=404)

        if message.approval_status == ApprovalStatus.REJECTED:
            raise AppError(
                code="approval_rejected",
                message="Rejected outreach messages cannot be sent",
                status_code=409,
            )

        message.approval_status = ApprovalStatus.APPROVED
        message.approved_by_user_id = approver_user_id
        session.add(message)
        session.flush()

        candidate = session.get(CandidateProfile, message.candidate_profile_id)
        to_email = candidate.email if candidate and candidate.email else "no-reply@example.invalid"

        try:
            send_email(EmailSendRequest(to_email=to_email, subject=message.subject, body=message.body))
            message.sent_status = SentStatus.SENT
            message.sent_at = datetime.utcnow()
        except EmailSendError:
            message.sent_status = SentStatus.FAILED

        session.add(message)
        session.flush()
        return message

    @staticmethod
    def _build_subject(candidate_name: str) -> str:
        return f"Opportunity discussion for {candidate_name}"

    @staticmethod
    def _build_body(candidate_name: str, role_hint: str | None, source_type: str, intent: str | None) -> str:
        role_fragment = role_hint or "the open role"
        intro = "AI-assisted draft" if source_type == "ai_draft" else "Template draft"
        intent_fragment = intent or "exploring fit and next steps"
        return (
            f"{intro}: Hello {candidate_name},\n\n"
            f"We are reaching out about {role_fragment}. "
            f"This message is focused on {intent_fragment}.\n\n"
            "If you are interested, please reply and we can schedule a conversation."
        )


outreach_service = OutreachService()
