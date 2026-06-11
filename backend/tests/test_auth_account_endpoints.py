import sys
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.v1.endpoints.auth import (  # noqa: E402
    LoginRequest,
    RegisterRequest,
    UpdateProfileRequest,
    get_me,
    login,
    register,
    router,
    update_me,
)
from src.models.deps import get_current_user, get_db  # noqa: E402
from src.models.base import Base  # noqa: E402
from src.models.enums import RoleName, UserStatus  # noqa: E402
from src.models.oauth_identity import OAuthIdentity  # noqa: E402
from src.models.user_account import RoleAssignment, UserAccount  # noqa: E402


def _create_test_tables(engine):
    tables = [
        Base.metadata.tables["user_accounts"],
        Base.metadata.tables["oauth_identities"],
        Base.metadata.tables["role_assignments"],
    ]
    Base.metadata.create_all(engine, tables=tables)


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
def auth_client(db):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/auth")

    def _override_db():
        yield db

    def _override_current_user():
        user = db.query(UserAccount).filter_by(email="current@example.com").one()
        return user

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user] = _override_current_user

    with TestClient(app) as client:
        yield client


def test_register_creates_active_user_and_recruiter_role(db, monkeypatch):
    monkeypatch.setattr(
        "src.api.v1.endpoints.auth.create_access_token",
        lambda **kwargs: "signed-token",
    )

    response = register(
        RegisterRequest(
            email="new.user@example.com",
            password="secret123",
            display_name="New User",
        ),
        db=db,
    )

    user = db.query(UserAccount).filter_by(email="new.user@example.com").one()
    roles = db.query(RoleAssignment).filter_by(user_id=user.id).all()

    assert response.access_token == "signed-token"
    assert user.display_name == "New User"
    assert user.status == UserStatus.ACTIVE
    assert user.password_hash == "hashed:secret123"
    assert [role.role_name for role in roles] == [RoleName.RECRUITER]


def test_register_rejects_duplicate_email(db):
    db.add(
        UserAccount(
            email="duplicate@example.com",
            display_name="Existing",
            password_hash="hashed:pw",
            status=UserStatus.ACTIVE,
        )
    )
    db.commit()

    with pytest.raises(HTTPException) as exc_info:
        register(
            RegisterRequest(
                email="duplicate@example.com",
                password="secret123",
                display_name="Another User",
            ),
            db=db,
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Email already registered"


def test_login_rejects_invalid_password_for_hashed_user(db):
    db.add(
        UserAccount(
            email="user@example.com",
            display_name="User",
            password_hash="hashed:correct-password",
            status=UserStatus.ACTIVE,
        )
    )
    db.commit()

    with pytest.raises(HTTPException) as exc_info:
        login(
            LoginRequest(email="user@example.com", password="wrong-password"),
            db=db,
        )

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid credentials"


def test_update_me_rejects_email_taken_by_other_user(db):
    current_user = UserAccount(
        email="current@example.com",
        display_name="Current",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    other_user = UserAccount(
        email="taken@example.com",
        display_name="Taken",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    db.add_all([current_user, other_user])
    db.commit()
    db.refresh(current_user)

    with pytest.raises(HTTPException) as exc_info:
        update_me(
            UpdateProfileRequest(email="taken@example.com"),
            current_user=current_user,
            db=db,
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Email already in use"


def test_get_me_returns_gmail_connected_false_without_google_refresh_token(db):
    user = UserAccount(
        email="current@example.com",
        display_name="Current",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    response = get_me(current_user=user)

    assert response.gmail_connected is False


def test_get_me_returns_gmail_connected_true_with_refresh_token_and_scope(db):
    user = UserAccount(
        email="current@example.com",
        display_name="Current",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    identity = OAuthIdentity(
        user_id=user.id,
        provider="google",
        provider_subject="google-sub-1",
        email=user.email,
        refresh_token_encrypted="encrypted-refresh",
        scope="openid email profile https://www.googleapis.com/auth/gmail.send",
    )
    db.add(identity)
    db.commit()
    db.refresh(user)

    response = get_me(current_user=user)

    assert response.gmail_connected is True


def test_update_me_response_does_not_include_gmail_connected(db):
    user = UserAccount(
        email="current@example.com",
        display_name="Current",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    response = update_me(
        UpdateProfileRequest(display_name="Updated"),
        current_user=user,
        db=db,
    )

    assert response.display_name == "Updated"
    assert not hasattr(response, "gmail_connected")


def test_get_me_http_returns_gmail_connected_false_without_google_refresh_token(
    db, auth_client
):
    user = UserAccount(
        email="current@example.com",
        display_name="Current",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    db.add(user)
    db.commit()

    response = auth_client.get("/api/v1/auth/me")

    assert response.status_code == 200
    assert response.json()["gmail_connected"] is False


def test_get_me_http_returns_gmail_connected_true_with_refresh_token_and_scope(
    db, auth_client
):
    user = UserAccount(
        email="current@example.com",
        display_name="Current",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    db.add(
        OAuthIdentity(
            user_id=user.id,
            provider="google",
            provider_subject="google-sub-1",
            email=user.email,
            refresh_token_encrypted="encrypted-refresh",
            scope="openid email profile https://www.googleapis.com/auth/gmail.send",
        )
    )
    db.commit()

    response = auth_client.get("/api/v1/auth/me")

    assert response.status_code == 200
    assert response.json()["gmail_connected"] is True


def test_get_me_patch_http_does_not_expose_gmail_connected(db, auth_client):
    user = UserAccount(
        email="current@example.com",
        display_name="Current",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    db.add(user)
    db.commit()

    response = auth_client.patch(
        "/api/v1/auth/me",
        json={"display_name": "Updated"},
    )

    assert response.status_code == 200
    assert response.json()["display_name"] == "Updated"
    assert "gmail_connected" not in response.json()
