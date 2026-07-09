import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.v1.endpoints.notifications import router  # noqa: E402
from src.models.base import Base  # noqa: E402
from src.models.deps import get_current_user, get_db  # noqa: E402
from src.models.enums import UserStatus  # noqa: E402
from src.models.notification import UserNotification  # noqa: E402
from src.models.user_account import UserAccount  # noqa: E402


def _create_test_tables(engine):
    Base.metadata.create_all(
        engine,
        tables=[
            Base.metadata.tables["user_accounts"],
            Base.metadata.tables["user_notifications"],
        ],
    )


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    _create_test_tables(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture()
def users(db):
    current = UserAccount(
        email="current@example.com",
        display_name="Current User",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    other = UserAccount(
        email="other@example.com",
        display_name="Other User",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    db.add_all([current, other])
    db.commit()
    db.refresh(current)
    db.refresh(other)
    return current, other


@pytest.fixture()
def client(db, users):
    current, _other = users
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/notifications")

    def _override_db():
        yield db

    def _override_current_user():
        return current

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user] = _override_current_user

    with TestClient(app) as test_client:
        yield test_client


def _notification(db, user, *, title, read=False):
    item = UserNotification(
        user_id=user.id,
        notification_type="candidate_applied",
        title=title,
        body=f"{title} body",
        target_url="/candidates/resume-1",
        payload={"source": "test"},
    )
    if read:
        from datetime import datetime, timezone

        item.read_at = datetime.now(timezone.utc)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def test_list_notifications_returns_current_user_items_and_unread_count(db, users, client):
    current, other = users
    _notification(db, current, title="Unread current")
    _notification(db, current, title="Read current", read=True)
    _notification(db, other, title="Unread other")

    response = client.get("/api/v1/notifications/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["unread_count"] == 1
    assert {item["title"] for item in payload["items"]} == {"Read current", "Unread current"}
    assert {item["user_id"] for item in payload["items"]} == {str(current.id)}


def test_mark_read_rejects_other_users_notification(db, users, client):
    _current, other = users
    other_notification = _notification(db, other, title="Other user")

    response = client.post(f"/api/v1/notifications/{other_notification.id}/read")

    assert response.status_code == 404
    assert response.json() == {"detail": "Notification not found"}


def test_mark_all_read_updates_current_user_unread_only(db, users, client):
    current, other = users
    current_unread = _notification(db, current, title="Current unread")
    current_read = _notification(db, current, title="Current read", read=True)
    other_unread = _notification(db, other, title="Other unread")

    response = client.post("/api/v1/notifications/read-all")

    assert response.status_code == 200
    assert response.json() == {"updated_count": 1}
    db.refresh(current_unread)
    db.refresh(current_read)
    db.refresh(other_unread)
    assert current_unread.read_at is not None
    assert current_read.read_at is not None
    assert other_unread.read_at is None
