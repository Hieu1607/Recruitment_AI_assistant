from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Any

import httpx

from src.core.config import settings
from src.models.oauth_identity import OAuthIdentity
from src.services.token_crypto import decrypt_token, encrypt_token

GMAIL_SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"


class GmailSendDisabledError(RuntimeError):
    pass


class GmailTokenMissingError(RuntimeError):
    pass


def build_raw_message(*, sender: str, to_email: str, subject: str, body: str) -> str:
    message = EmailMessage()
    message["From"] = sender
    message["To"] = to_email
    message["Subject"] = subject
    message.set_content(body)
    return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii").rstrip("=")


def _is_expired(token_expires_at: datetime | None) -> bool:
    if token_expires_at is None:
        return True
    now = datetime.now(timezone.utc)
    if token_expires_at.tzinfo is None:
        token_expires_at = token_expires_at.replace(tzinfo=timezone.utc)
    return token_expires_at <= now + timedelta(minutes=2)


def _refresh_access_token(identity: OAuthIdentity) -> str:
    if not identity.refresh_token_encrypted:
        raise GmailTokenMissingError(
            "Google refresh token is missing. Ask the recruiter to reconnect Google."
        )

    refresh_token = decrypt_token(identity.refresh_token_encrypted)
    response = httpx.post(
        GOOGLE_TOKEN_URL,
        data={
            "client_id": settings.GOOGLE_CLIENT_ID,
            "client_secret": settings.GOOGLE_CLIENT_SECRET,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=settings.GMAIL_SEND_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    access_token = payload["access_token"]
    identity.access_token_encrypted = encrypt_token(access_token)
    identity.token_expires_at = datetime.now(timezone.utc) + timedelta(
        seconds=int(payload.get("expires_in", 3600))
    )
    return access_token


def get_access_token(identity: OAuthIdentity) -> str:
    if _is_expired(identity.token_expires_at):
        return _refresh_access_token(identity)
    if not identity.access_token_encrypted:
        return _refresh_access_token(identity)
    return decrypt_token(identity.access_token_encrypted)


def send_gmail_message(
    *,
    identity: OAuthIdentity,
    sender: str,
    to_email: str,
    subject: str,
    body: str,
) -> dict[str, Any]:
    if not settings.GMAIL_SEND_ENABLED:
        raise GmailSendDisabledError("Gmail sending is disabled by GMAIL_SEND_ENABLED.")

    access_token = get_access_token(identity)
    raw = build_raw_message(sender=sender, to_email=to_email, subject=subject, body=body)
    response = httpx.post(
        GMAIL_SEND_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json={"raw": raw},
        timeout=settings.GMAIL_SEND_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()
