"""Tests for src.services.google_oauth — no real network calls."""
import time
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.models.base import Base
from src.models.enums import RoleName, UserStatus
from src.models.oauth_identity import OAuthIdentity
from src.models.user_account import UserAccount, RoleAssignment
from src.services import google_oauth


# ---------------------------------------------------------------------------
# In-memory SQLite DB fixture — only the tables we need
# ---------------------------------------------------------------------------

def _create_test_tables(engine):
    """Create only the auth-related tables (avoids JSONB from other models)."""
    # Import all models so metadata is populated, then selectively create
    from src.models import user_account, oauth_identity  # noqa: F401

    tables = [
        Base.metadata.tables["user_accounts"],
        Base.metadata.tables["role_assignments"],
        Base.metadata.tables["oauth_identities"],
    ]
    Base.metadata.create_all(engine, tables=tables)


@pytest.fixture(scope="function")
def db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    _create_test_tables(engine)
    with Session(engine) as session:
        yield session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_claims(
    sub: str = "google-sub-123",
    email: str = "test@example.com",
    email_verified: bool = True,
    name: str = "Test User",
) -> dict:
    return {
        "sub": sub,
        "email": email,
        "email_verified": email_verified,
        "name": name,
        "iss": "https://accounts.google.com",
        "aud": "test-client-id",
        "exp": int(time.time()) + 3600,
    }


# ---------------------------------------------------------------------------
# State signing / verification tests
# ---------------------------------------------------------------------------

def test_build_then_verify_state_roundtrip():
    url, state = google_oauth.build_authorize_url("/dashboard")
    assert "accounts.google.com" in url
    redirect = google_oauth.verify_state(state)
    assert redirect == "/dashboard"


def test_verify_state_rejects_tampered():
    _, state = google_oauth.build_authorize_url("/dashboard")
    mid = len(state) // 2
    tampered = state[:mid] + ("X" if state[mid] != "X" else "Y") + state[mid + 1:]
    with pytest.raises(ValueError):
        google_oauth.verify_state(tampered)


def test_verify_state_rejects_expired():
    from unittest.mock import patch
    from itsdangerous import SignatureExpired, BadTimeSignature

    _, state = google_oauth.build_authorize_url("/dashboard")

    # Simulate itsdangerous raising SignatureExpired (token too old)
    with patch.object(
        google_oauth._get_serializer().__class__,
        "loads",
        side_effect=SignatureExpired("expired", payload=None),
    ):
        with pytest.raises(ValueError, match="state_expired"):
            google_oauth.verify_state(state)


# ---------------------------------------------------------------------------
# upsert_user_from_google tests
# ---------------------------------------------------------------------------

def test_upsert_creates_new_user_when_not_exists(db):
    claims = _make_claims()
    user = google_oauth.upsert_user_from_google(db, claims)

    assert user.email == "test@example.com"
    assert user.display_name == "Test User"
    assert user.password_hash is None
    assert user.status == UserStatus.ACTIVE

    roles = db.query(RoleAssignment).filter_by(user_id=user.id).all()
    assert any(r.role_name == RoleName.RECRUITER for r in roles)

    identity = db.query(OAuthIdentity).filter_by(
        provider="google", provider_subject="google-sub-123"
    ).one()
    assert identity.user_id == user.id


def test_upsert_links_existing_user_when_email_verified(db):
    existing = UserAccount(
        email="test@example.com",
        display_name="Old Name",
        password_hash="hashed",
        status=UserStatus.ACTIVE,
    )
    db.add(existing)
    db.commit()
    db.refresh(existing)

    claims = _make_claims(email="test@example.com", email_verified=True)
    user = google_oauth.upsert_user_from_google(db, claims)

    assert user.id == existing.id
    assert db.query(UserAccount).count() == 1

    identity = db.query(OAuthIdentity).filter_by(
        provider="google", provider_subject="google-sub-123"
    ).one()
    assert identity.user_id == existing.id


def test_upsert_rejects_when_email_not_verified_and_user_exists(db):
    existing = UserAccount(
        email="victim@example.com",
        display_name="Victim",
        password_hash="hashed",
        status=UserStatus.ACTIVE,
    )
    db.add(existing)
    db.commit()

    claims = _make_claims(email="victim@example.com", email_verified=False)
    with pytest.raises(ValueError, match="google_email_not_verified"):
        google_oauth.upsert_user_from_google(db, claims)

    assert db.query(OAuthIdentity).count() == 0


def test_upsert_returns_existing_identity_user_on_second_login(db):
    """Second login with same Google sub returns same user without duplication."""
    claims = _make_claims()
    user1 = google_oauth.upsert_user_from_google(db, claims)
    user2 = google_oauth.upsert_user_from_google(db, claims)

    assert user1.id == user2.id
    assert db.query(UserAccount).count() == 1
    assert db.query(OAuthIdentity).count() == 1


def test_upsert_stores_encrypted_google_tokens(db, monkeypatch):
    encrypted_values = []
    monkeypatch.setattr(
        google_oauth,
        "encrypt_token",
        lambda value: encrypted_values.append(value) or f"encrypted:{value}",
    )
    claims = _make_claims()
    tokens = {
        "access_token": "access-token",
        "refresh_token": "refresh-token",
        "expires_in": 3600,
        "scope": "openid email profile https://www.googleapis.com/auth/gmail.send",
    }

    user = google_oauth.upsert_user_from_google(db, claims, tokens=tokens)

    identity = db.query(OAuthIdentity).filter_by(user_id=user.id).one()
    assert identity.access_token_encrypted == "encrypted:access-token"
    assert identity.refresh_token_encrypted == "encrypted:refresh-token"
    assert identity.token_expires_at is not None
    expires_at = identity.token_expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    assert expires_at > datetime.now(timezone.utc)
    assert "gmail.send" in identity.scope
    assert encrypted_values == ["access-token", "refresh-token"]


def test_upsert_preserves_existing_refresh_token_when_google_omits_it(db, monkeypatch):
    monkeypatch.setattr(google_oauth, "encrypt_token", lambda value: f"encrypted:{value}")
    claims = _make_claims()
    first_tokens = {
        "access_token": "first-access",
        "refresh_token": "first-refresh",
        "expires_in": 3600,
        "scope": "openid email profile https://www.googleapis.com/auth/gmail.send",
    }
    second_tokens = {
        "access_token": "second-access",
        "expires_in": 3600,
        "scope": "openid email profile https://www.googleapis.com/auth/gmail.send",
    }

    google_oauth.upsert_user_from_google(db, claims, tokens=first_tokens)
    user = google_oauth.upsert_user_from_google(db, claims, tokens=second_tokens)

    identity = db.query(OAuthIdentity).filter_by(user_id=user.id).one()
    assert identity.access_token_encrypted == "encrypted:second-access"
    assert identity.refresh_token_encrypted == "encrypted:first-refresh"
