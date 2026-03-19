from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.api.dependencies.auth import CurrentUser, require_roles
from src.repositories.db import get_session
from src.services.shortlist.shortlist_service import shortlist_service

router = APIRouter(prefix="/v1/shortlists", tags=["shortlists"])


class ShortlistCreateRequest(BaseModel):
    name: str = Field(min_length=1)
    candidateIds: list[str] = Field(min_length=1)
    sourceQueryTurnId: str | None = None


class ShortlistCollectionResponse(BaseModel):
    id: str
    name: str
    candidateIds: list[str]


@router.post("", response_model=ShortlistCollectionResponse, status_code=201)
def create_shortlist(
    payload: ShortlistCreateRequest,
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[CurrentUser, Depends(require_roles("admin", "recruiter"))],
) -> ShortlistCollectionResponse:
    user_id = uuid.UUID(current_user.user_id)
    candidate_ids = [uuid.UUID(item) for item in payload.candidateIds]
    source_query_turn_id = uuid.UUID(payload.sourceQueryTurnId) if payload.sourceQueryTurnId else None

    created = shortlist_service.create_shortlist(
        session,
        name=payload.name,
        created_by_user_id=user_id,
        candidate_ids=candidate_ids,
        source_query_turn_id=source_query_turn_id,
    )
    session.commit()

    return ShortlistCollectionResponse(
        id=str(created.collection.id),
        name=created.collection.name,
        candidateIds=created.candidate_ids,
    )


@router.get("", response_model=list[ShortlistCollectionResponse])
def list_shortlists(
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[CurrentUser, Depends(require_roles("admin", "recruiter", "viewer"))],
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
) -> list[ShortlistCollectionResponse]:
    user_id = uuid.UUID(current_user.user_id)
    rows = shortlist_service.list_shortlists_for_user(session, created_by_user_id=user_id, limit=limit)
    return [
        ShortlistCollectionResponse(
            id=str(row.collection.id),
            name=row.collection.name,
            candidateIds=row.candidate_ids,
        )
        for row in rows
    ]
