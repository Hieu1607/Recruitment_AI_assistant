"""
Deprecated compatibility endpoints for top-level JobDescription CRUD.

Primary UI flows should use the job-scoped endpoints under
/api/v1/jobs/{job_id}/job-description so the selected job remains the
single source of truth in the product experience.
"""
from __future__ import annotations

import uuid
from typing import Annotated, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select

from src.models.deps import get_current_user, get_db
from src.models.job import Job
from src.models.job_matching import JobDescription
from src.models.user_account import UserAccount
from src.services.job_description_service import (
    create_job_description,
    delete_job_description,
    get_job_description,
    list_job_descriptions,
    update_job_description,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class JobDescriptionCreateRequest(BaseModel):
    title: Optional[str] = Field(None, max_length=255, description="Optional job title")
    jd_text: str = Field("", description="Full job description text")
    hidden_text: str = Field("", description="Recruiter-only hidden scoring criteria")


class JobDescriptionUpdateRequest(BaseModel):
    title: Optional[str] = Field(None, max_length=255, description="New title (omit to leave unchanged)")
    jd_text: Optional[str] = Field(None, min_length=1, description="Updated JD text (omit to leave unchanged)")
    hidden_text: Optional[str] = Field(None, description="Updated recruiter-only hidden scoring criteria")
    is_active: Optional[bool] = Field(None, description="Set active/inactive status (omit to leave unchanged)")


class JobDescriptionResponse(BaseModel):
    id: str
    title: Optional[str]
    jd_text: str
    hidden_text: str
    created_by_user_id: str
    created_at: str
    is_active: bool


class JobDescriptionListResponse(BaseModel):
    total: int
    items: List[JobDescriptionResponse]


class DeleteResponse(BaseModel):
    deleted: bool
    job_description_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/",
    response_model=JobDescriptionResponse,
    status_code=201,
    summary="Create a new Job Description",
)
def create_jd(
    body: JobDescriptionCreateRequest,
    db=Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    try:
        first_job = db.execute(
            select(Job).where(Job.owner_user_id == current_user.id).order_by(Job.created_at.asc())
        ).scalars().first()
        if first_job is None:
            raise HTTPException(status_code=400, detail="Create a job before creating a job description")
        result = create_job_description(
            db=db,
            job_id=first_job.id,
            jd_text=body.jd_text,
            hidden_text=body.hidden_text,
            created_by_user_id=current_user.id,
            title=body.title,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result


@router.get(
    "/",
    response_model=JobDescriptionListResponse,
    summary="List Job Descriptions",
)
def list_jds(
    is_active: Annotated[Optional[bool], Query(description="Filter by active status")] = None,
    limit: Annotated[int, Query(ge=1, le=200, description="Max records to return")] = 50,
    offset: Annotated[int, Query(ge=0, description="Records to skip")] = 0,
    db=Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    first_job = db.execute(
        select(Job).where(Job.owner_user_id == current_user.id).order_by(Job.created_at.asc())
    ).scalars().first()
    if first_job is None:
        return JobDescriptionListResponse(total=0, items=[])
    items = list_job_descriptions(db=db, job_id=first_job.id, is_active=is_active, limit=limit, offset=offset)
    return JobDescriptionListResponse(total=len(items), items=items)


@router.get(
    "/{jd_id}",
    response_model=JobDescriptionResponse,
    summary="Get a specific Job Description",
)
def get_jd(
    jd_id: uuid.UUID,
    db=Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    jd = db.execute(
        select(Job).join(JobDescription, Job.id == JobDescription.job_id).where(
            JobDescription.id == jd_id, Job.owner_user_id == current_user.id
        )
    ).scalars().first()
    if jd is None:
        result = None
    else:
        result = get_job_description(db=db, jd_id=jd_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Job description {jd_id} not found")
    return result


@router.patch(
    "/{jd_id}",
    response_model=JobDescriptionResponse,
    summary="Update a Job Description",
    description=(
        "Partially update a job description. "
        "Only fields included in the request body are changed. "
        "Use `is_active: false` to deactivate without deleting."
    ),
)
def update_jd(
    jd_id: uuid.UUID,
    body: Annotated[
        JobDescriptionUpdateRequest,
        Body(description="Fields to update (all optional)"),
    ],
    db=Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    owner_match = db.execute(
        select(Job).join(JobDescription, Job.id == JobDescription.job_id).where(
            JobDescription.id == jd_id, Job.owner_user_id == current_user.id
        )
    ).scalars().first()
    if owner_match is None:
        raise HTTPException(status_code=404, detail=f"Job description {jd_id} not found")
    try:
        result = update_job_description(
            db=db,
            jd_id=jd_id,
            title=body.title,
            jd_text=body.jd_text,
            hidden_text=body.hidden_text,
            is_active=body.is_active,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail=f"Job description {jd_id} not found")
    return result


@router.delete(
    "/{jd_id}",
    response_model=DeleteResponse,
    summary="Delete a Job Description",
)
def delete_jd(
    jd_id: uuid.UUID,
    db=Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    owner_match = db.execute(
        select(Job).join(JobDescription, Job.id == JobDescription.job_id).where(
            JobDescription.id == jd_id, Job.owner_user_id == current_user.id
        )
    ).scalars().first()
    if owner_match is None:
        deleted = False
    else:
        deleted = delete_job_description(db=db, jd_id=jd_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Job description {jd_id} not found")
    return DeleteResponse(deleted=True, job_description_id=str(jd_id))
