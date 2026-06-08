"""Endpoint tests for Google OAuth routes — no real Google calls."""
import sys
import types
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

if "minio" not in sys.modules:
    minio_module = types.ModuleType("minio")
    minio_module.Minio = MagicMock()
    sys.modules["minio"] = minio_module
    minio_error_module = types.ModuleType("minio.error")
    minio_error_module.S3Error = Exception
    sys.modules["minio.error"] = minio_error_module

from src.api.v1.endpoints.auth import router as auth_router
from src.models.deps import get_current_user, get_db
from src.services import google_oauth

TEST_USER_ID = uuid.uuid4()

# ---------------------------------------------------------------------------
# Minimal DB override — return a mock session so endpoints don't need Postgres
# ---------------------------------------------------------------------------

def _mock_db():
    yield MagicMock()


def _mock_current_user():
    user = MagicMock()
    user.id = TEST_USER_ID
    user.email = "test@example.com"
    return user


app = FastAPI()
app.include_router(auth_router, prefix="/api/v1/auth")
app.dependency_overrides[get_db] = _mock_db
app.dependency_overrides[get_current_user] = _mock_current_user

client = TestClient(app, follow_redirects=False)


# ---------------------------------------------------------------------------
# /auth/google/login
# ---------------------------------------------------------------------------

class TestGoogleLogin:
    def test_returns_302_to_google(self):
        resp = client.get("/api/v1/auth/google/login")
        assert resp.status_code == 302
        location = resp.headers["location"]
        assert "accounts.google.com/o/oauth2/v2/auth" in location

    def test_missing_config_redirects_back_to_login(self):
        with patch("src.api.v1.endpoints.auth.settings.GOOGLE_CLIENT_ID", ""), patch(
            "src.api.v1.endpoints.auth.settings.GOOGLE_CLIENT_SECRET", "secret"
        ):
            resp = client.get("/api/v1/auth/google/login")

        assert resp.status_code == 302
        assert "/login?error=oauth_not_configured" in resp.headers["location"]

    def test_location_has_required_params(self):
        resp = client.get("/api/v1/auth/google/login")
        location = resp.headers["location"]
        assert "response_type=code" in location
        assert "scope=" in location
        assert "state=" in location
        assert "redirect_uri=" in location

    def test_redirect_param_embedded_in_state(self):
        from itsdangerous import URLSafeTimedSerializer
        from src.core.config import settings

        resp = client.get("/api/v1/auth/google/login?redirect=/candidates")
        location = resp.headers["location"]

        # Extract state from URL
        from urllib.parse import parse_qs, urlparse
        qs = parse_qs(urlparse(location).query)
        state = qs["state"][0]

        serializer = URLSafeTimedSerializer(settings.SECRET_KEY, salt="google-oauth-state")
        payload = serializer.loads(state)
        assert payload["redirect"] == "/candidates"

    def test_open_redirect_sanitised(self):
        from itsdangerous import URLSafeTimedSerializer
        from src.core.config import settings
        from urllib.parse import parse_qs, urlparse

        resp = client.get("/api/v1/auth/google/login?redirect=//evil.com")
        location = resp.headers["location"]
        qs = parse_qs(urlparse(location).query)
        state = qs["state"][0]
        serializer = URLSafeTimedSerializer(settings.SECRET_KEY, salt="google-oauth-state")
        payload = serializer.loads(state)
        assert payload["redirect"] == "/dashboard"

    def test_http_scheme_redirect_sanitised(self):
        from itsdangerous import URLSafeTimedSerializer
        from src.core.config import settings
        from urllib.parse import parse_qs, urlparse

        resp = client.get("/api/v1/auth/google/login?redirect=http://evil.com/steal")
        location = resp.headers["location"]
        qs = parse_qs(urlparse(location).query)
        state = qs["state"][0]
        serializer = URLSafeTimedSerializer(settings.SECRET_KEY, salt="google-oauth-state")
        payload = serializer.loads(state)
        assert payload["redirect"] == "/dashboard"


# ---------------------------------------------------------------------------
# /auth/google/callback
# ---------------------------------------------------------------------------

def _build_valid_state(
    redirect: str = "/dashboard",
    flow: str = "login",
    initiating_user_id: str | None = None,
) -> str:
    from itsdangerous import URLSafeTimedSerializer
    from src.core.config import settings
    import secrets as _secrets

    serializer = URLSafeTimedSerializer(settings.SECRET_KEY, salt="google-oauth-state")
    payload = {"redirect": redirect, "flow": flow, "nonce": _secrets.token_urlsafe(8)}
    if initiating_user_id is not None:
        payload["initiating_user_id"] = initiating_user_id
    return serializer.dumps(payload)


class TestGoogleCallback:
    def test_google_connect_gmail_returns_authorize_url_with_gmail_scope(self):
        resp = client.get("/api/v1/auth/google/connect-gmail?redirect=/outreach")
        assert resp.status_code == 200
        authorize_url = resp.json()["authorize_url"]
        assert "gmail.send" in authorize_url

    def test_google_connect_gmail_authorize_url_state_includes_initiating_user_id(self):
        from urllib.parse import parse_qs, urlparse

        resp = client.get("/api/v1/auth/google/connect-gmail?redirect=/outreach")
        assert resp.status_code == 200

        state = parse_qs(urlparse(resp.json()["authorize_url"]).query)["state"][0]
        payload = google_oauth.verify_state(state)
        assert payload["redirect"] == "/outreach"
        assert payload["flow"] == "connect_gmail"
        assert payload["initiating_user_id"] == str(TEST_USER_ID)

    def test_error_param_redirects_to_login(self):
        resp = client.get("/api/v1/auth/google/callback?error=access_denied")
        assert resp.status_code == 302
        assert "/login?error=access_denied" in resp.headers["location"]

    def test_google_callback_denied_connect_gmail_returns_to_outreach(self):
        state = _build_valid_state(
            "/outreach", flow="connect_gmail", initiating_user_id=str(TEST_USER_ID)
        )
        resp = client.get(
            f"/api/v1/auth/google/callback?error=access_denied&state={state}",
            follow_redirects=False,
        )
        assert resp.status_code == 302
        assert "/outreach" in resp.headers["location"]
        assert "error=gmail_consent_denied" in resp.headers["location"]

    def test_missing_code_redirects_to_login(self):
        state = _build_valid_state()
        resp = client.get(f"/api/v1/auth/google/callback?state={state}")
        assert resp.status_code == 302
        assert "error=missing_params" in resp.headers["location"]

    def test_missing_state_redirects_to_login(self):
        resp = client.get("/api/v1/auth/google/callback?code=somecode")
        assert resp.status_code == 302
        assert "error=missing_params" in resp.headers["location"]

    def test_tampered_state_redirects_to_login(self):
        resp = client.get("/api/v1/auth/google/callback?code=abc&state=TAMPERED_XYZ")
        assert resp.status_code == 302
        assert "error=invalid_state" in resp.headers["location"]

    def test_happy_path_redirects_to_frontend_with_token(self):
        import uuid
        from src.models.user_account import UserAccount
        from src.models.enums import UserStatus

        fake_user = MagicMock(spec=UserAccount)
        fake_user.id = uuid.uuid4()
        fake_user.email = "test@example.com"
        fake_user.display_name = "Test User"
        fake_user.status = UserStatus.ACTIVE

        state = _build_valid_state("/dashboard")

        with (
            patch(
                "src.api.v1.endpoints.auth.google_oauth.exchange_code_for_tokens",
                new_callable=AsyncMock,
                return_value={"id_token": "fake_id_token", "access_token": "fake_access"},
            ),
            patch(
                "src.api.v1.endpoints.auth.google_oauth.verify_id_token",
                return_value={
                    "sub": "google-sub-999",
                    "email": "test@example.com",
                    "email_verified": True,
                    "name": "Test User",
                },
            ),
            patch(
                "src.api.v1.endpoints.auth.google_oauth.upsert_user_from_google",
                return_value=fake_user,
            ),
        ):
            resp = client.get(f"/api/v1/auth/google/callback?code=real_code&state={state}")

        assert resp.status_code == 302
        location = resp.headers["location"]
        assert "/auth/callback" in location
        assert "token=" in location
        assert "redirect=" in location
        # Token must look like a JWT (three dot-separated base64 segments)
        from urllib.parse import parse_qs, urlparse
        token = parse_qs(urlparse(location).query)["token"][0]
        assert token.count(".") == 2

    def test_google_callback_success_connect_gmail_returns_to_outreach(self):
        import uuid
        from src.models.enums import UserStatus
        from src.models.user_account import UserAccount

        fake_user = MagicMock(spec=UserAccount)
        fake_user.id = uuid.uuid4()
        fake_user.email = "test@example.com"
        fake_user.display_name = "Test User"
        fake_user.status = UserStatus.ACTIVE

        state = _build_valid_state(
            "/outreach", flow="connect_gmail", initiating_user_id=str(TEST_USER_ID)
        )

        connect_identity = MagicMock(return_value=fake_user)
        with (
            patch(
                "src.api.v1.endpoints.auth.google_oauth.exchange_code_for_tokens",
                new_callable=AsyncMock,
                return_value={"id_token": "fake", "refresh_token": "refresh"},
            ),
            patch(
                "src.api.v1.endpoints.auth.google_oauth.verify_id_token",
                return_value={
                    "sub": "google-sub-999",
                    "email": "test@example.com",
                    "email_verified": True,
                    "name": "Test User",
                },
            ),
            patch(
                "src.api.v1.endpoints.auth.google_oauth.connect_google_identity_to_user",
                connect_identity,
            ),
        ):
            resp = client.get(f"/api/v1/auth/google/callback?code=real_code&state={state}")

        assert resp.status_code == 302
        assert "/outreach" in resp.headers["location"]
        assert "gmail_connected=1" in resp.headers["location"]
        connect_identity.assert_called_once()
        assert connect_identity.call_args.kwargs["target_user_id"] == str(TEST_USER_ID)

    def test_google_callback_success_connect_gmail_preserves_existing_redirect_query(self):
        import uuid
        from urllib.parse import parse_qs, urlparse
        from src.models.enums import UserStatus
        from src.models.user_account import UserAccount

        fake_user = MagicMock(spec=UserAccount)
        fake_user.id = uuid.uuid4()
        fake_user.email = "test@example.com"
        fake_user.display_name = "Test User"
        fake_user.status = UserStatus.ACTIVE

        state = _build_valid_state(
            "/outreach?tab=gmail", flow="connect_gmail", initiating_user_id=str(TEST_USER_ID)
        )

        with (
            patch(
                "src.api.v1.endpoints.auth.google_oauth.exchange_code_for_tokens",
                new_callable=AsyncMock,
                return_value={"id_token": "fake", "refresh_token": "refresh"},
            ),
            patch(
                "src.api.v1.endpoints.auth.google_oauth.verify_id_token",
                return_value={
                    "sub": "google-sub-999",
                    "email": "test@example.com",
                    "email_verified": True,
                    "name": "Test User",
                },
            ),
            patch(
                "src.api.v1.endpoints.auth.google_oauth.connect_google_identity_to_user",
                return_value=fake_user,
            ),
        ):
            resp = client.get(f"/api/v1/auth/google/callback?code=real_code&state={state}")

        assert resp.status_code == 302
        location = resp.headers["location"]
        parsed = urlparse(location)
        assert parsed.path.endswith("/outreach")
        assert parse_qs(parsed.query) == {"tab": ["gmail"], "gmail_connected": ["1"]}

    def test_google_callback_connect_gmail_mismatch_returns_error_to_outreach(self):
        state = _build_valid_state(
            "/outreach", flow="connect_gmail", initiating_user_id=str(TEST_USER_ID)
        )

        connect_identity = MagicMock(side_effect=ValueError("google_account_mismatch"))
        with (
            patch(
                "src.api.v1.endpoints.auth.google_oauth.exchange_code_for_tokens",
                new_callable=AsyncMock,
                return_value={"id_token": "fake", "refresh_token": "refresh"},
            ),
            patch(
                "src.api.v1.endpoints.auth.google_oauth.verify_id_token",
                return_value={
                    "sub": "google-sub-999",
                    "email": "mismatch@example.com",
                    "email_verified": True,
                    "name": "Mismatch User",
                },
            ),
            patch(
                "src.api.v1.endpoints.auth.google_oauth.connect_google_identity_to_user",
                connect_identity,
            ),
        ):
            resp = client.get(f"/api/v1/auth/google/callback?code=real_code&state={state}")

        assert resp.status_code == 302
        assert "/outreach" in resp.headers["location"]
        assert "error=google_account_mismatch" in resp.headers["location"]
        connect_identity.assert_called_once()
        assert connect_identity.call_args.kwargs["target_user_id"] == str(TEST_USER_ID)

    def test_email_not_verified_redirects_with_error(self):
        state = _build_valid_state()

        with (
            patch(
                "src.api.v1.endpoints.auth.google_oauth.exchange_code_for_tokens",
                new_callable=AsyncMock,
                return_value={"id_token": "fake_id_token"},
            ),
            patch(
                "src.api.v1.endpoints.auth.google_oauth.verify_id_token",
                return_value={
                    "sub": "google-sub-999",
                    "email": "victim@example.com",
                    "email_verified": False,
                    "name": "Victim",
                },
            ),
            patch(
                "src.api.v1.endpoints.auth.google_oauth.upsert_user_from_google",
                side_effect=ValueError("google_email_not_verified"),
            ),
        ):
            resp = client.get(f"/api/v1/auth/google/callback?code=real_code&state={state}")

        assert resp.status_code == 302
        assert "error=google_email_not_verified" in resp.headers["location"]
