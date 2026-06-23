from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models.notification import UserNotification


def serialize_notification(notification: UserNotification) -> dict[str, Any]:
    return {
        "id": str(notification.id),
        "user_id": str(notification.user_id),
        "notification_type": notification.notification_type,
        "title": notification.title,
        "body": notification.body,
        "target_url": notification.target_url,
        "payload": notification.payload or {},
        "created_at": notification.created_at,
        "read_at": notification.read_at,
    }


def create_notification(
    *,
    db: Session,
    user_id: uuid.UUID,
    notification_type: str,
    title: str,
    body: str = "",
    target_url: str | None = None,
    metadata: dict[str, Any] | None = None,
    commit: bool = True,
) -> UserNotification:
    notification = UserNotification(
        user_id=user_id,
        notification_type=notification_type,
        title=title,
        body=body,
        target_url=target_url,
        payload=metadata or {},
    )
    db.add(notification)
    if commit:
        db.commit()
        db.refresh(notification)
    else:
        db.flush()
    return notification


def list_user_notifications(
    *,
    db: Session,
    user_id: uuid.UUID,
    limit: int = 20,
    unread_only: bool = False,
) -> list[UserNotification]:
    statement = select(UserNotification).where(UserNotification.user_id == user_id)
    if unread_only:
        statement = statement.where(UserNotification.read_at.is_(None))
    return (
        db.execute(statement.order_by(UserNotification.created_at.desc()).limit(limit))
        .scalars()
        .all()
    )


def mark_notification_read(
    *,
    db: Session,
    user_id: uuid.UUID,
    notification_id: uuid.UUID,
) -> UserNotification | None:
    notification = db.get(UserNotification, notification_id)
    if notification is None or notification.user_id != user_id:
        return None
    if notification.read_at is None:
        notification.read_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(notification)
    return notification


def mark_all_notifications_read(*, db: Session, user_id: uuid.UUID) -> int:
    notifications = list_user_notifications(db=db, user_id=user_id, limit=100, unread_only=True)
    now = datetime.now(timezone.utc)
    for notification in notifications:
        notification.read_at = now
    if notifications:
        db.commit()
    return len(notifications)
