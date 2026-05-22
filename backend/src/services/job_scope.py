from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from src.models.candidate_profile import CandidateProfile
from src.models.interview_template import InterviewTemplate
from src.models.interview_invitation import InterviewInvitation
from src.models.job import Job
from src.models.job_matching import JobDescription
from src.models.resume_document import ResumeDocument
from src.services.public_job_service import build_public_apply_url


def serialize_job(job: Job) -> dict:
    return {
        "id": str(job.id),
        "owner_user_id": str(job.owner_user_id),
        "title": job.title,
        "status": job.status,
        "candidate_message": job.candidate_message,
        "public_apply_enabled": job.public_apply_enabled,
        "public_apply_url": build_public_apply_url(job.public_apply_token),
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "archived_at": job.archived_at.isoformat() if job.archived_at else None,
    }


def serialize_job_application_settings(job: Job) -> dict:
    return {
        "public_apply_enabled": job.public_apply_enabled,
        "public_apply_url": build_public_apply_url(job.public_apply_token),
        "candidate_message": job.candidate_message,
    }


def apply_public_job_settings(
    job: Job,
    *,
    candidate_message: str | None | object = ...,
    public_apply_enabled: bool | None | object = ...,
) -> None:
    if candidate_message is not ...:
        job.candidate_message = candidate_message.strip() if isinstance(candidate_message, str) else candidate_message

    if public_apply_enabled is not ... and public_apply_enabled is not None:
        job.public_apply_enabled = public_apply_enabled
        job.public_apply_disabled_at = None if public_apply_enabled else datetime.now(timezone.utc)


def get_current_user_owned_job(db: Session, user_id: uuid.UUID, job_id: uuid.UUID) -> Job:
    job = db.execute(
        select(Job).where(Job.id == job_id, Job.owner_user_id == user_id)
    ).scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


def get_job_scoped_jd(db: Session, user_id: uuid.UUID, job_id: uuid.UUID) -> Optional[JobDescription]:
    return db.execute(
        select(JobDescription)
        .join(Job, Job.id == JobDescription.job_id)
        .where(JobDescription.job_id == job_id, Job.owner_user_id == user_id)
        .order_by(JobDescription.created_at.desc())
    ).scalars().first()


def require_job_scoped_jd(db: Session, user_id: uuid.UUID, job_id: uuid.UUID) -> JobDescription:
    jd = get_job_scoped_jd(db, user_id, job_id)
    if jd is None:
        raise HTTPException(status_code=404, detail=f"Job description for job '{job_id}' not found")
    return jd


def get_job_scoped_resume(db: Session, user_id: uuid.UUID, job_id: uuid.UUID, resume_id: uuid.UUID) -> ResumeDocument:
    resume = db.execute(
        select(ResumeDocument)
        .join(Job, Job.id == ResumeDocument.job_id)
        .where(
            ResumeDocument.id == resume_id,
            ResumeDocument.job_id == job_id,
            Job.owner_user_id == user_id,
        )
    ).scalar_one_or_none()
    if resume is None:
        raise HTTPException(status_code=404, detail=f"Resume '{resume_id}' not found")
    return resume


def get_job_scoped_candidate(
    db: Session, user_id: uuid.UUID, job_id: uuid.UUID, candidate_id: uuid.UUID
) -> CandidateProfile:
    candidate = db.execute(
        select(CandidateProfile)
        .options(joinedload(CandidateProfile.resume_document))
        .join(ResumeDocument, ResumeDocument.id == CandidateProfile.resume_document_id)
        .join(Job, Job.id == ResumeDocument.job_id)
        .where(
            CandidateProfile.id == candidate_id,
            ResumeDocument.job_id == job_id,
            Job.owner_user_id == user_id,
        )
    ).scalar_one_or_none()
    if candidate is None:
        raise HTTPException(status_code=404, detail=f"Candidate '{candidate_id}' not found")
    return candidate


def get_job_scoped_interview_template(
    db: Session,
    user_id: uuid.UUID,
    job_id: uuid.UUID,
    template_id: uuid.UUID,
) -> InterviewTemplate:
    template = db.execute(
        select(InterviewTemplate)
        .join(Job, Job.id == InterviewTemplate.job_id)
        .where(
            InterviewTemplate.id == template_id,
            InterviewTemplate.job_id == job_id,
            Job.owner_user_id == user_id,
        )
    ).scalar_one_or_none()
    if template is None:
        raise HTTPException(status_code=404, detail=f"Interview template '{template_id}' not found")
    return template


def get_user_owned_interview_template(
    db: Session,
    user_id: uuid.UUID,
    template_id: uuid.UUID,
) -> InterviewTemplate:
    template = db.execute(
        select(InterviewTemplate)
        .join(Job, Job.id == InterviewTemplate.job_id)
        .where(
            InterviewTemplate.id == template_id,
            Job.owner_user_id == user_id,
        )
    ).scalar_one_or_none()
    if template is None:
        raise HTTPException(status_code=404, detail=f"Interview template '{template_id}' not found")
    return template


def get_user_owned_interview_invitation(
    db: Session,
    user_id: uuid.UUID,
    invitation_id: uuid.UUID,
) -> InterviewInvitation:
    invitation = db.execute(
        select(InterviewInvitation)
        .join(Job, Job.id == InterviewInvitation.job_id)
        .where(
            InterviewInvitation.id == invitation_id,
            Job.owner_user_id == user_id,
        )
    ).scalar_one_or_none()
    if invitation is None:
        raise HTTPException(status_code=404, detail=f"Interview invitation '{invitation_id}' not found")
    return invitation
