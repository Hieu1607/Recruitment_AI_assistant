from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage

from src.services.observability.audit_logger import audit_log


class EmailSendError(RuntimeError):
    pass


@dataclass
class EmailSendRequest:
    to_email: str
    subject: str
    body: str


def send_email(request: EmailSendRequest) -> None:
    host = os.getenv("SMTP_HOST", "localhost")
    port = int(os.getenv("SMTP_PORT", "25"))
    username = os.getenv("SMTP_USERNAME", "")
    password = os.getenv("SMTP_PASSWORD", "")
    sender = os.getenv("SMTP_FROM", username or "no-reply@recruitment.local")

    message = EmailMessage()
    message["From"] = sender
    message["To"] = request.to_email
    message["Subject"] = request.subject
    message.set_content(request.body)

    try:
        with smtplib.SMTP(host=host, port=port, timeout=15) as smtp:
            smtp.starttls()
            if username and password:
                smtp.login(username, password)
            smtp.send_message(message)
        audit_log(
            "outreach_email_sent",
            {
                "to_email": request.to_email,
                "subject": request.subject,
            },
        )
    except Exception as exc:
        audit_log(
            "outreach_email_failed",
            {
                "to_email": request.to_email,
                "subject": request.subject,
                "error": str(exc),
            },
        )
        raise EmailSendError("Unable to send outreach email") from exc
