from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.models.deps import get_current_user, get_db
from src.models.user_account import UserAccount
from src.services.activity_service import list_recent_activities

router = APIRouter()


class ActivityResponse(BaseModel):
    id: str
    kind: str
    timestamp: datetime
    subject_name: str | None
    context_name: str | None
    status: str | None
    target_url: str | None
    metadata: dict[str, Any]


class ActivityListResponse(BaseModel):
    items: list[ActivityResponse]


@router.get("/", response_model=ActivityListResponse)
def list_activities(
    limit: int = Query(12, ge=1, le=50),
    job_id: uuid.UUID | None = Query(None),
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    return ActivityListResponse(
        items=list_recent_activities(
            db=db,
            user_id=current_user.id,
            job_id=job_id,
            limit=limit,
        )
    )
