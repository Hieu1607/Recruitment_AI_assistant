import re
import sys
import types
from pathlib import Path

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pydantic_settings  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("pydantic_settings")

    class BaseSettings:
        pass

    stub.BaseSettings = BaseSettings
    sys.modules["pydantic_settings"] = stub

from src.core.config import settings
from src.models.base import Base
from src.models.candidate_profile import CandidateProfile
from src.models.job import Job
from src.models.user_account import UserAccount
from src.models.enums import UserStatus
from src.services.public_job_service import (
    build_public_apply_url,
    generate_public_apply_token,
    resolve_public_job_by_token,
)


TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _create_test_tables(engine):
    tables = [
        Base.metadata.tables["user_accounts"],
        Base.metadata.tables["jobs"],
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


def test_generate_public_apply_token_has_expected_shape_and_length():
    token = generate_public_apply_token()

    assert isinstance(token, str)
    assert len(token) == 43
    assert TOKEN_PATTERN.fullmatch(token)


def test_build_public_apply_url_uses_frontend_base_url(monkeypatch):
    monkeypatch.setattr(settings, "FRONTEND_BASE_URL", "https://frontend.example.com/")

    assert (
        build_public_apply_url("abc123")
        == "https://frontend.example.com/apply/abc123"
    )


def test_resolve_public_job_by_token_returns_job(db):
    owner = UserAccount(
        email="owner@example.com",
        display_name="Owner",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    db.add(owner)
    db.flush()

    job = Job(
        owner_user_id=owner.id,
        title="Backend Engineer",
        public_apply_token="public-token-123",
    )
    db.add(job)
    db.commit()

    resolved = resolve_public_job_by_token(db, "public-token-123")

    assert resolved.id == job.id
    assert resolved.title == "Backend Engineer"


def test_resolve_public_job_by_token_raises_404_for_unknown_token(db):
    with pytest.raises(HTTPException) as exc_info:
        resolve_public_job_by_token(db, "missing-token")

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Public application link not found"


def test_job_public_apply_token_column_uses_generate_public_apply_token_default():
    column = Job.__table__.c.public_apply_token

    assert column.default is not None
    assert callable(column.default.arg)
    assert column.default.arg.__name__ == generate_public_apply_token.__name__
    assert column.default.arg.__module__ == generate_public_apply_token.__module__


def test_job_model_exposes_public_apply_columns():
    column_names = set(Job.__table__.columns.keys())

    assert {
        "public_apply_token",
        "public_apply_enabled",
        "candidate_message",
        "public_apply_created_at",
        "public_apply_disabled_at",
    }.issubset(column_names)


def test_candidate_profile_model_exposes_public_apply_submission_columns():
    column_names = set(CandidateProfile.__table__.columns.keys())

    assert {"submitted_full_name", "submitted_email"}.issubset(column_names)
