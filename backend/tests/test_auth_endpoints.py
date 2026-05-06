"""Endpoint tests for Google OAuth routes — no real Google calls."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.models.deps import get_db

# ---------------------------------------------------------------------------
# Minimal DB override — return a mock session so endpoints don't need Postgres
# ---------------------------------------------------------------------------

def _mock_db():
    yield MagicMock()


app.dependency_overrides[get_db] = _mock_db

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

def _build_valid_state(redirect: str = "/dashboard") -> str:
    from itsdangerous import URLSafeTimedSerializer
    from src.core.config import settings
    import secrets as _secrets

    serializer = URLSafeTimedSerializer(settings.SECRET_KEY, salt="google-oauth-state")
    return serializer.dumps({"redirect": redirect, "nonce": _secrets.token_urlsafe(8)})


class TestGoogleCallback:
    def test_error_param_redirects_to_login(self):
        resp = client.get("/api/v1/auth/google/callback?error=access_denied")
        assert resp.status_code == 302
        assert "/login?error=access_denied" in resp.headers["location"]

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
