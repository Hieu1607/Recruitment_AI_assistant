from __future__ import annotations

import uuid
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from src.models.candidate_profile import CandidateProfile
from src.models.job import Job
from src.models.job_matching import JobDescription
from src.models.resume_document import ResumeDocument


def serialize_job(job: Job) -> dict:
    return {
        "id": str(job.id),
        "owner_user_id": str(job.owner_user_id),
        "title": job.title,
        "status": job.status,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "archived_at": job.archived_at.isoformat() if job.archived_at else None,
    }


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
