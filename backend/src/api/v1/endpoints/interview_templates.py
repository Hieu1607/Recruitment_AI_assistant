from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.models.deps import get_current_user, get_db
from src.models.user_account import UserAccount
from src.schemas.interview_invitation import (
    InterviewInvitationCreateRequest,
    InterviewInvitationListResponse,
    InterviewInvitationResponse,
)
from src.schemas.interview_template import (
    InterviewTemplateCreateRequest,
    InterviewTemplateListResponse,
    InterviewTemplateResponse,
    InterviewTemplateUpdateRequest,
)
from src.services.interview_invitation_service import (
    create_interview_invitation,
    list_interview_invitations,
    revoke_interview_invitation,
    serialize_interview_invitation,
)
from src.services.interview_template_service import (
    create_interview_template,
    delete_interview_template,
    get_interview_template,
    list_interview_templates,
    serialize_interview_template,
    update_interview_template,
)

router = APIRouter()


@router.post("/jobs/{job_id}/interview-templates", response_model=InterviewTemplateResponse, status_code=201)
def create_job_interview_template(
    job_id: uuid.UUID,
    body: InterviewTemplateCreateRequest,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    template = create_interview_template(db, user_id=current_user.id, job_id=job_id, body=body)
    return serialize_interview_template(template)


@router.get("/jobs/{job_id}/interview-templates", response_model=InterviewTemplateListResponse)
def list_job_interview_templates(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    templates = list_interview_templates(db, user_id=current_user.id, job_id=job_id)
    return InterviewTemplateListResponse(
        items=[serialize_interview_template(template) for template in templates],
        total=len(templates),
    )


@router.get("/interview-templates/{template_id}", response_model=InterviewTemplateResponse)
def get_single_interview_template(
    template_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    template = get_interview_template(db, user_id=current_user.id, template_id=template_id)
    return serialize_interview_template(template)


@router.patch("/interview-templates/{template_id}", response_model=InterviewTemplateResponse)
def update_single_interview_template(
    template_id: uuid.UUID,
    body: InterviewTemplateUpdateRequest,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    template = update_interview_template(db, user_id=current_user.id, template_id=template_id, body=body)
    return serialize_interview_template(template)


@router.delete("/interview-templates/{template_id}")
def delete_single_interview_template(
    template_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    delete_interview_template(db, user_id=current_user.id, template_id=template_id)
    return {"deleted": True, "template_id": str(template_id)}


@router.post("/interview-invitations", response_model=InterviewInvitationResponse, status_code=201)
def create_job_interview_invitation(
    body: InterviewInvitationCreateRequest,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    invitation = create_interview_invitation(db, user_id=current_user.id, body=body)
    if body.send_email:
        from worker.tasks import send_interview_invitation_email

        send_interview_invitation_email.delay(str(invitation.id))
    return serialize_interview_invitation(invitation)


@router.get("/jobs/{job_id}/interview-invitations", response_model=InterviewInvitationListResponse)
def list_job_interview_invitations(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    invitations = list_interview_invitations(db, user_id=current_user.id, job_id=job_id)
    return InterviewInvitationListResponse(
        items=[serialize_interview_invitation(invitation) for invitation in invitations],
        total=len(invitations),
    )


@router.post("/interview-invitations/{invitation_id}/revoke", response_model=InterviewInvitationResponse)
def revoke_single_interview_invitation(
    invitation_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    invitation = revoke_interview_invitation(db, user_id=current_user.id, invitation_id=invitation_id)
    return serialize_interview_invitation(invitation)
