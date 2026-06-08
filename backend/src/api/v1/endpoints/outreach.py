"""CRUD endpoints for OutreachMessage.

Route map
---------
  POST   /outreach/              create a message
  GET    /outreach/              list (filter by user / candidate / sent_status)
  GET    /outreach/{id}         get single message
  PATCH  /outreach/{id}         update subject, body, or sent_status
  DELETE /outreach/{id}         delete
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from src.models.candidate_profile import CandidateProfile
from src.models.deps import get_current_user, get_db
from src.models.enums import ContentSource, SentStatus
from src.models.oauth_identity import GMAIL_SEND_SCOPE, OAuthIdentity
from src.models.outreach import OutreachMessage
from src.models.user_account import UserAccount
from src.models.session import SessionLocal

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_or_404(db, model, record_id: uuid.UUID, label: str):
    obj = db.get(model, record_id)
    if obj is None:
        raise HTTPException(status_code=404, detail=f"{label} '{record_id}' not found")
    return obj


def _get_gmail_capable_identity(db, user_id: uuid.UUID) -> OAuthIdentity | None:
    identity = (
        db.execute(
            select(OAuthIdentity).where(
                OAuthIdentity.user_id == user_id,
                OAuthIdentity.provider == "google",
            )
        )
        .scalar_one_or_none()
    )
    if identity is None:
        return None
    if not identity.refresh_token_encrypted:
        return None
    if not identity.has_scope(GMAIL_SEND_SCOPE):
        return None
    return identity


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class OutreachCreateRequest(BaseModel):
    candidate_profile_id: uuid.UUID
    created_by_user_id: uuid.UUID
    content_source: ContentSource
    subject: str = Field(..., min_length=1, max_length=255)
    body: str = Field(..., min_length=1)


class OutreachUpdateRequest(BaseModel):
    subject: Optional[str] = Field(None, min_length=1, max_length=255)
    body: Optional[str] = Field(None, min_length=1)
    sent_status: Optional[SentStatus] = None


class OutreachResponse(BaseModel):
    id: str
    candidate_profile_id: str
    candidate_full_name: Optional[str]
    created_by_user_id: str
    content_source: str
    subject: str
    body: str
    sent_status: str
    sent_at: Optional[datetime]
    created_at: datetime


class OutreachListResponse(BaseModel):
    total: int
    items: List[OutreachResponse]


# ---------------------------------------------------------------------------
# Serialiser
# ---------------------------------------------------------------------------

def _ser(m: OutreachMessage) -> OutreachResponse:
    return OutreachResponse(
        id=str(m.id),
        candidate_profile_id=str(m.candidate_profile_id),
        candidate_full_name=m.candidate_profile.full_name if m.candidate_profile else None,
        created_by_user_id=str(m.created_by_user_id),
        content_source=m.content_source.value,
        subject=m.subject,
        body=m.body,
        sent_status=m.sent_status.value,
        sent_at=m.sent_at,
        created_at=m.created_at,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/", response_model=OutreachResponse, status_code=201)
def create_message(body: OutreachCreateRequest):
    db = SessionLocal()
    try:
        _get_or_404(db, CandidateProfile, body.candidate_profile_id, "CandidateProfile")

        msg = OutreachMessage(
            candidate_profile_id=body.candidate_profile_id,
            created_by_user_id=body.created_by_user_id,
            content_source=body.content_source,
            subject=body.subject,
            body=body.body,
            sent_status=SentStatus.NOT_SENT,
        )
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return _ser(msg)
    finally:
        db.close()


@router.get("/", response_model=OutreachListResponse)
def list_messages(
    created_by_user_id: Optional[uuid.UUID] = Query(None, description="Filter by creator user"),
    candidate_profile_id: Optional[uuid.UUID] = Query(None, description="Filter by candidate"),
    sent_status: Optional[SentStatus] = Query(None, description="Filter by sent status"),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    db = SessionLocal()
    try:
        query = db.query(OutreachMessage)

        if created_by_user_id is not None:
            query = query.filter(OutreachMessage.created_by_user_id == created_by_user_id)
        if candidate_profile_id is not None:
            query = query.filter(OutreachMessage.candidate_profile_id == candidate_profile_id)
        if sent_status is not None:
            query = query.filter(OutreachMessage.sent_status == sent_status)

        total = query.count()
        rows = query.order_by(OutreachMessage.created_at.desc()).offset(offset).limit(limit).all()

        return OutreachListResponse(total=total, items=[_ser(r) for r in rows])
    finally:
        db.close()


@router.get("/{message_id}", response_model=OutreachResponse)
def get_message(message_id: uuid.UUID):
    db = SessionLocal()
    try:
        msg = _get_or_404(db, OutreachMessage, message_id, "OutreachMessage")
        return _ser(msg)
    finally:
        db.close()


@router.patch("/{message_id}", response_model=OutreachResponse)
def update_message(message_id: uuid.UUID, body: OutreachUpdateRequest):
    db = SessionLocal()
    try:
        msg = _get_or_404(db, OutreachMessage, message_id, "OutreachMessage")

        if body.subject is not None:
            msg.subject = body.subject
        if body.body is not None:
            msg.body = body.body
        if body.sent_status is not None:
            msg.sent_status = body.sent_status
            if body.sent_status == SentStatus.SENT and msg.sent_at is None:
                msg.sent_at = datetime.now(timezone.utc)

        db.commit()
        db.refresh(msg)
        return _ser(msg)
    finally:
        db.close()


@router.post("/{message_id}/send", response_model=OutreachResponse, status_code=status.HTTP_202_ACCEPTED)
def send_message(
    message_id: uuid.UUID,
    db=Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    msg = _get_or_404(db, OutreachMessage, message_id, "OutreachMessage")
    if msg.created_by_user_id != current_user.id:
        raise HTTPException(status_code=404, detail=f"OutreachMessage '{message_id}' not found")
    if msg.sent_status == SentStatus.SENT:
        return _ser(msg)
    if _get_gmail_capable_identity(db, current_user.id) is None:
        raise HTTPException(status_code=409, detail="gmail_not_connected")
    if not msg.candidate_profile or not msg.candidate_profile.email:
        msg.sent_status = SentStatus.FAILED
        db.commit()
        db.refresh(msg)
        return _ser(msg)

    from worker.tasks import send_outreach_email

    send_outreach_email.delay(str(msg.id))
    return _ser(msg)


@router.delete("/{message_id}", status_code=204)
def delete_message(message_id: uuid.UUID):
    db = SessionLocal()
    try:
        msg = _get_or_404(db, OutreachMessage, message_id, "OutreachMessage")
        db.delete(msg)
        db.commit()
    finally:
        db.close()
