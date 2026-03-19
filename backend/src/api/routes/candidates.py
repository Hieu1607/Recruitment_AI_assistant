from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from src.api.dependencies.auth import CurrentUser, require_roles
from src.api.errors import NotFoundError
from src.repositories.candidate_repository import candidate_repository
from src.repositories.db import get_session

router = APIRouter(prefix="/v1/candidates", tags=["candidates"])


class CandidateUpdateRequest(BaseModel):
    full_name: str | None = Field(default=None, alias="fullName")
    phone: str | None = None
    email: str | None = None
    location_normalized: str | None = Field(default=None, alias="locationNormalized")
    educated: bool | None = None
    ever_studied_abroad: bool | None = Field(default=None, alias="everStudiedAbroad")
    current_job_title: str | None = Field(default=None, alias="currentJobTitle")

    model_config = ConfigDict(populate_by_name=True)


class CandidateProfileResponse(BaseModel):
    id: str
    fullName: str
    phone: str | None
    email: str | None
    locationNormalized: str | None
    educated: bool
    everStudiedAbroad: bool
    profileStatus: str


class ExtractionTraceResponse(BaseModel):
    id: str
    sourcePage: int
    sourceBBox: dict
    sourceTextSnippet: str
    mappedFieldName: str
    confidenceScore: float | None


def _to_candidate_response(candidate) -> CandidateProfileResponse:
    return CandidateProfileResponse(
        id=str(candidate.id),
        fullName=candidate.full_name,
        phone=candidate.phone,
        email=candidate.email,
        locationNormalized=candidate.location_normalized,
        educated=bool(candidate.educated),
        everStudiedAbroad=bool(candidate.ever_studied_abroad),
        profileStatus=candidate.profile_status.value,
    )


@router.get("", response_model=list[CandidateProfileResponse])
def list_candidates(
    session: Annotated[Session, Depends(get_session)],
    _current_user: Annotated[CurrentUser, Depends(require_roles("admin", "recruiter", "viewer"))],
    q: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[CandidateProfileResponse]:
    candidates = candidate_repository.list_candidates(session, query=q, limit=limit)
    return [_to_candidate_response(item) for item in candidates]


@router.patch("/{candidate_id}", response_model=CandidateProfileResponse)
def update_candidate(
    candidate_id: str,
    payload: CandidateUpdateRequest,
    session: Annotated[Session, Depends(get_session)],
    _current_user: Annotated[CurrentUser, Depends(require_roles("admin", "recruiter"))],
) -> CandidateProfileResponse:
    candidate_uuid = uuid.UUID(candidate_id)
    candidate = candidate_repository.update_candidate(
        session,
        candidate_id=candidate_uuid,
        patch=payload.model_dump(by_alias=False, exclude_none=True),
    )
    if not candidate:
        raise NotFoundError("Candidate not found")
    session.commit()
    return _to_candidate_response(candidate)


@router.get("/{candidate_id}/traces", response_model=list[ExtractionTraceResponse])
def get_candidate_traces(
    candidate_id: str,
    session: Annotated[Session, Depends(get_session)],
    _current_user: Annotated[CurrentUser, Depends(require_roles("admin", "recruiter", "viewer"))],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[ExtractionTraceResponse]:
    traces = candidate_repository.get_traces_for_candidate(session, uuid.UUID(candidate_id), limit=limit)
    return [
        ExtractionTraceResponse(
            id=str(item.id),
            sourcePage=item.source_page,
            sourceBBox=item.source_bbox,
            sourceTextSnippet=item.source_text_snippet,
            mappedFieldName=item.mapped_field_name,
            confidenceScore=float(item.confidence_score) if item.confidence_score is not None else None,
        )
        for item in traces
    ]
