from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.models.deps import get_current_user, get_db
from src.models.user_account import UserAccount
from src.services.notification_service import (
    list_user_notifications,
    mark_all_notifications_read,
    mark_notification_read,
    serialize_notification,
)

router = APIRouter()


class NotificationResponse(BaseModel):
    id: str
    user_id: str
    notification_type: str
    title: str
    body: str
    target_url: str | None
    payload: dict[str, Any]
    created_at: datetime
    read_at: datetime | None


class NotificationListResponse(BaseModel):
    items: list[NotificationResponse]
    unread_count: int


class MarkAllReadResponse(BaseModel):
    updated_count: int


@router.get("/", response_model=NotificationListResponse)
def list_notifications(
    limit: int = Query(default=20, ge=1, le=100),
    unread_only: bool = False,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
) -> NotificationListResponse:
    unread = list_user_notifications(
        db=db,
        user_id=current_user.id,
        limit=100,
        unread_only=True,
    )
    items = list_user_notifications(
        db=db,
        user_id=current_user.id,
        limit=limit,
        unread_only=unread_only,
    )
    return NotificationListResponse(
        items=[NotificationResponse(**serialize_notification(item)) for item in items],
        unread_count=len(unread),
    )


@router.post("/{notification_id}/read", response_model=NotificationResponse)
def mark_read(
    notification_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
) -> NotificationResponse:
    notification = mark_notification_read(
        db=db,
        user_id=current_user.id,
        notification_id=notification_id,
    )
    if notification is None:
        raise HTTPException(status_code=404, detail="Notification not found")
    return NotificationResponse(**serialize_notification(notification))


@router.post("/read-all", response_model=MarkAllReadResponse)
def mark_all_read(
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
) -> MarkAllReadResponse:
    return MarkAllReadResponse(
        updated_count=mark_all_notifications_read(db=db, user_id=current_user.id),
    )
