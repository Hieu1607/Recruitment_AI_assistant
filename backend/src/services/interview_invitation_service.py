from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from src.core.config import settings
from src.models.interview_invitation import InterviewInvitation
from src.schemas.interview_invitation import (
    InterviewInvitationCreateRequest,
    InterviewInvitationResponse,
)
from src.services.job_scope import (
    get_current_user_owned_job,
    get_job_scoped_candidate,
    get_job_scoped_interview_template,
    get_user_owned_interview_invitation,
)


def build_interview_public_url(public_token: str) -> str:
    return f"{settings.FRONTEND_BASE_URL.rstrip('/')}/interviews/{public_token}"


def serialize_interview_invitation(invitation: InterviewInvitation) -> InterviewInvitationResponse:
    candidate_name = invitation.candidate_profile.full_name if invitation.candidate_profile is not None else None
    template_name = invitation.interview_template.name if invitation.interview_template is not None else None
    return InterviewInvitationResponse(
        id=str(invitation.id),
        job_id=str(invitation.job_id),
        candidate_profile_id=str(invitation.candidate_profile_id),
        candidate_full_name=candidate_name,
        interview_template_id=str(invitation.interview_template_id),
        interview_template_name=template_name,
        public_token=invitation.public_token,
        public_url=build_interview_public_url(invitation.public_token),
        status=invitation.status,
        expires_at=invitation.expires_at,
        max_attempts=invitation.max_attempts,
        attempt_count=invitation.attempt_count,
        sent_by_user_id=str(invitation.sent_by_user_id) if invitation.sent_by_user_id else None,
        sent_at=invitation.sent_at,
        opened_at=invitation.opened_at,
        completed_at=invitation.completed_at,
        cancelled_at=invitation.cancelled_at,
        created_at=invitation.created_at,
        updated_at=invitation.updated_at,
    )


def create_interview_invitation(
    db: Session,
    *,
    user_id: uuid.UUID,
    body: InterviewInvitationCreateRequest,
) -> InterviewInvitation:
    job_id = body.job_id
    get_current_user_owned_job(db, user_id, job_id)
    candidate = get_job_scoped_candidate(db, user_id, job_id, body.candidate_profile_id)
    template = get_job_scoped_interview_template(db, user_id, job_id, body.interview_template_id)

    expires_at = None
    if body.expires_in_hours is not None:
        expires_at = datetime.now(timezone.utc) + timedelta(hours=body.expires_in_hours)

    invitation = InterviewInvitation(
        job_id=job_id,
        candidate_profile_id=candidate.id,
        interview_template_id=template.id,
        expires_at=expires_at,
        sent_by_user_id=user_id,
    )
    db.add(invitation)
    db.commit()
    db.refresh(invitation)
    return get_interview_invitation(db, user_id=user_id, invitation_id=invitation.id)


def list_interview_invitations(db: Session, *, user_id: uuid.UUID, job_id: uuid.UUID) -> list[InterviewInvitation]:
    get_current_user_owned_job(db, user_id, job_id)
    return (
        db.execute(
            select(InterviewInvitation)
            .options(
                joinedload(InterviewInvitation.candidate_profile),
                joinedload(InterviewInvitation.interview_template),
            )
            .where(InterviewInvitation.job_id == job_id)
            .order_by(InterviewInvitation.created_at.desc())
        )
        .scalars()
        .all()
    )


def get_interview_invitation(
    db: Session,
    *,
    user_id: uuid.UUID,
    invitation_id: uuid.UUID,
) -> InterviewInvitation:
    get_user_owned_interview_invitation(db, user_id, invitation_id)
    return (
        db.execute(
            select(InterviewInvitation)
            .options(
                joinedload(InterviewInvitation.candidate_profile),
                joinedload(InterviewInvitation.interview_template),
            )
            .where(InterviewInvitation.id == invitation_id)
        )
        .scalars()
        .one()
    )
