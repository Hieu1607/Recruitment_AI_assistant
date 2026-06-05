from __future__ import annotations

from src.models.oauth_identity import OAuthIdentity
from src.services.gmail_service import send_gmail_message


def send_email(
    *,
    sender: str,
    to_email: str,
    subject: str,
    body: str,
    identity: OAuthIdentity,
) -> dict:
    return send_gmail_message(
        identity=identity,
        sender=sender,
        to_email=to_email,
        subject=subject,
        body=body,
    )
