from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.services import gmail_service


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {"id": "gmail-message-id"}
        self.text = "response-text"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise gmail_service.httpx.HTTPStatusError(
                "failed",
                request=SimpleNamespace(),
                response=SimpleNamespace(status_code=self.status_code, text=self.text),
            )

    def json(self):
        return self._payload


def test_build_raw_message_contains_headers():
    raw = gmail_service.build_raw_message(
        sender="recruiter@example.com",
        to_email="candidate@example.com",
        subject="Interview invite",
        body="Please join the interview.",
    )

    decoded = gmail_service.base64.urlsafe_b64decode(raw + "==").decode("utf-8")
    assert "From: recruiter@example.com" in decoded
    assert "To: candidate@example.com" in decoded
    assert "Subject: Interview invite" in decoded
    assert "Please join the interview." in decoded


def test_send_message_uses_existing_access_token(monkeypatch):
    identity = SimpleNamespace(
        access_token_encrypted="encrypted-access",
        refresh_token_encrypted="encrypted-refresh",
        token_expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
    )
    calls = []
    monkeypatch.setattr(gmail_service, "decrypt_token", lambda value: value.replace("encrypted-", ""))
    monkeypatch.setattr(gmail_service, "encrypt_token", lambda value: f"encrypted-{value}")
    monkeypatch.setattr(gmail_service.settings, "GMAIL_SEND_ENABLED", True)

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(payload={"id": "gmail-123"})

    monkeypatch.setattr(gmail_service.httpx, "post", fake_post)

    result = gmail_service.send_gmail_message(
        identity=identity,
        sender="recruiter@example.com",
        to_email="candidate@example.com",
        subject="Hello",
        body="Body",
    )

    assert result["id"] == "gmail-123"
    assert calls[0][0] == "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
    assert calls[0][1]["headers"]["Authorization"] == "Bearer access"


def test_send_message_refreshes_expired_access_token(monkeypatch):
    identity = SimpleNamespace(
        access_token_encrypted="encrypted-old-access",
        refresh_token_encrypted="encrypted-refresh",
        token_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    monkeypatch.setattr(gmail_service, "decrypt_token", lambda value: value.replace("encrypted-", ""))
    monkeypatch.setattr(gmail_service, "encrypt_token", lambda value: f"encrypted-{value}")
    monkeypatch.setattr(gmail_service.settings, "GMAIL_SEND_ENABLED", True)
    monkeypatch.setattr(gmail_service.settings, "GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setattr(gmail_service.settings, "GOOGLE_CLIENT_SECRET", "client-secret")

    def fake_post(url, **kwargs):
        if url == "https://oauth2.googleapis.com/token":
            return FakeResponse(payload={"access_token": "new-access", "expires_in": 3600})
        return FakeResponse(payload={"id": "gmail-456"})

    monkeypatch.setattr(gmail_service.httpx, "post", fake_post)

    result = gmail_service.send_gmail_message(
        identity=identity,
        sender="recruiter@example.com",
        to_email="candidate@example.com",
        subject="Hello",
        body="Body",
    )

    assert result["id"] == "gmail-456"
    assert identity.access_token_encrypted == "encrypted-new-access"
    expires_at = identity.token_expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    assert expires_at > datetime.now(timezone.utc)


def test_send_message_requires_enabled_flag(monkeypatch):
    identity = SimpleNamespace()
    monkeypatch.setattr(gmail_service.settings, "GMAIL_SEND_ENABLED", False)

    with pytest.raises(gmail_service.GmailSendDisabledError):
        gmail_service.send_gmail_message(
            identity=identity,
            sender="recruiter@example.com",
            to_email="candidate@example.com",
            subject="Hello",
            body="Body",
        )
