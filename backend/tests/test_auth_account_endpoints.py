import sys
from pathlib import Path

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.v1.endpoints.auth import (  # noqa: E402
    LoginRequest,
    RegisterRequest,
    UpdateProfileRequest,
    login,
    register,
    update_me,
)
from src.models.base import Base  # noqa: E402
from src.models.enums import RoleName, UserStatus  # noqa: E402
from src.models.user_account import RoleAssignment, UserAccount  # noqa: E402


def _create_test_tables(engine):
    tables = [
        Base.metadata.tables["user_accounts"],
        Base.metadata.tables["role_assignments"],
    ]
    Base.metadata.create_all(engine, tables=tables)


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    _create_test_tables(engine)
    with Session(engine) as session:
        yield session


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
