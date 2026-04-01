"""
upload_jobdescriptions.py
REST endpoints for JobDescription CRUD.
"""
from __future__ import annotations

import uuid
from typing import Annotated, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field

from src.models.session import SessionLocal
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
    jd_text: str = Field(..., min_length=1, description="Full job description text")
    created_by_user_id: uuid.UUID = Field(..., description="UUID of the user creating the JD")


class JobDescriptionUpdateRequest(BaseModel):
    title: Optional[str] = Field(None, max_length=255, description="New title (omit to leave unchanged)")
    jd_text: Optional[str] = Field(None, min_length=1, description="Updated JD text (omit to leave unchanged)")
    is_active: Optional[bool] = Field(None, description="Set active/inactive status (omit to leave unchanged)")


class JobDescriptionResponse(BaseModel):
    id: str
    title: Optional[str]
    jd_text: str
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
def create_jd(body: JobDescriptionCreateRequest):
    db = SessionLocal()
    try:
        result = create_job_description(
            db=db,
            jd_text=body.jd_text,
            created_by_user_id=body.created_by_user_id,
            title=body.title,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        db.close()
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
):
    db = SessionLocal()
    try:
        items = list_job_descriptions(db=db, is_active=is_active, limit=limit, offset=offset)
    finally:
        db.close()
    return JobDescriptionListResponse(total=len(items), items=items)


@router.get(
    "/{jd_id}",
    response_model=JobDescriptionResponse,
    summary="Get a specific Job Description",
)
def get_jd(jd_id: uuid.UUID):
    db = SessionLocal()
    try:
        result = get_job_description(db=db, jd_id=jd_id)
    finally:
        db.close()
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
):
    db = SessionLocal()
    try:
        result = update_job_description(
            db=db,
            jd_id=jd_id,
            title=body.title,
            jd_text=body.jd_text,
            is_active=body.is_active,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        db.close()
    if result is None:
        raise HTTPException(status_code=404, detail=f"Job description {jd_id} not found")
    return result


@router.delete(
    "/{jd_id}",
    response_model=DeleteResponse,
    summary="Delete a Job Description",
)
def delete_jd(jd_id: uuid.UUID):
    db = SessionLocal()
    try:
        deleted = delete_job_description(db=db, jd_id=jd_id)
    finally:
        db.close()
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Job description {jd_id} not found")
    return DeleteResponse(deleted=True, job_description_id=str(jd_id))
