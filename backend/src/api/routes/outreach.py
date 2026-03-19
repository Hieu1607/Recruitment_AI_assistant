from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.dependencies.auth import CurrentUser, require_roles
from src.repositories.db import get_session
from src.services.outreach.outreach_service import OutreachDraftInput, outreach_service

router = APIRouter(prefix="/v1/outreach", tags=["outreach"])


class OutreachDraftRequest(BaseModel):
    candidateId: str
    sourceType: str
    templateId: str | None = None
    intent: str | None = None


class OutreachMessageResponse(BaseModel):
    id: str
    subject: str
    body: str
    approvalStatus: str
    sentStatus: str


@router.post("/drafts", response_model=OutreachMessageResponse, status_code=201)
def create_outreach_draft(
    payload: OutreachDraftRequest,
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[CurrentUser, Depends(require_roles("admin", "recruiter"))],
) -> OutreachMessageResponse:
    user_id = uuid.UUID(current_user.user_id)
    created = outreach_service.create_draft(
        session,
        OutreachDraftInput(
            candidate_id=uuid.UUID(payload.candidateId),
            created_by_user_id=user_id,
            source_type=payload.sourceType,
            template_id=payload.templateId,
            intent=payload.intent,
        ),
    )
    session.commit()
    return OutreachMessageResponse(
        id=str(created.id),
        subject=created.subject,
        body=created.body,
        approvalStatus=created.approval_status.value,
        sentStatus=created.sent_status.value,
    )


@router.post("/{message_id}/approve-and-send", response_model=OutreachMessageResponse)
def approve_and_send_outreach(
    message_id: str,
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[CurrentUser, Depends(require_roles("admin", "recruiter"))],
) -> OutreachMessageResponse:
    user_id = uuid.UUID(current_user.user_id)
    message = outreach_service.approve_and_send(
        session,
        message_id=uuid.UUID(message_id),
        approver_user_id=user_id,
    )
    session.commit()
    return OutreachMessageResponse(
        id=str(message.id),
        subject=message.subject,
        body=message.body,
        approvalStatus=message.approval_status.value,
        sentStatus=message.sent_status.value,
    )
