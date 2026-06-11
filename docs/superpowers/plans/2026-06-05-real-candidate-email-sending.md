# Real Candidate Email Sending Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow recruiters to send real candidate emails from their own Google account through Gmail API, starting with interview invitations and then outreach drafts.

**Architecture:** Extend the existing backend-mediated Google OAuth flow to request Gmail send permission and store per-user refresh tokens securely. Add a small Gmail transport service, send mail through Celery tasks, and update invitation/outreach status only after Gmail accepts the message. Keep frontend changes scoped to explicit "Send" actions and clear failure states.

**Tech Stack:** FastAPI, SQLAlchemy, Alembic, Celery, Redis, httpx, Google OAuth 2.0, Gmail API `users.messages.send`, React, TanStack Query, pytest, Playwright.

---

## Context Snapshot

Current repo state relevant to this work:

- `backend/src/services/mail_service.py` is a stub that only prints.
- `backend/src/services/google_oauth.py` supports Google sign-in with `openid email profile`, but not Gmail send.
- `backend/src/models/oauth_identity.py` stores provider identity but no OAuth access or refresh tokens.
- `backend/src/services/interview_invitation_service.py` creates public interview URLs via `settings.FRONTEND_BASE_URL` and currently sets `sent_at` when the invitation row is created.
- `backend/src/api/v1/endpoints/outreach.py` stores outreach drafts and allows manual status updates, but it does not send email.
- `frontend/src/components/interviews/InvitationSendDialog.tsx` calls `api.interviewInvitations.create`.
- `frontend/src/routes/outreach.tsx` has "Mark as sent", but no real send action.
- `backend/worker/tasks.py` and `backend/worker/celery_app.py` already provide a Celery worker path suitable for async email sending.

The worktree may contain unrelated user edits. Before changing code, run `git status --short` and avoid reverting files outside this plan.

## File Structure

Create:

- `backend/src/services/token_crypto.py` - encrypt and decrypt OAuth tokens with Fernet.
- `backend/src/services/gmail_service.py` - refresh Google access tokens and call Gmail API.
- `backend/src/services/email_templates.py` - build interview invitation and outreach email bodies.
- `backend/tests/test_token_crypto.py` - token encryption unit tests.
- `backend/tests/test_gmail_service.py` - Gmail transport unit tests with mocked HTTP.
- `backend/tests/test_email_templates.py` - deterministic email template tests.
- `backend/tests/test_interview_invitation_email.py` - invitation email behavior tests.
- `backend/tests/test_outreach_send_endpoint.py` - outreach send endpoint tests.
- `backend/migrations/versions/<revision>_add_google_mail_tokens.py` - Alembic migration generated during implementation.

Modify:

- `backend/requirements.txt` - add `cryptography`.
- `.env.example` - add Gmail/OAuth token configuration placeholders.
- `docker-compose.yml` - pass new env vars to backend and worker containers.
- `backend/src/core/config.py` - add Gmail OAuth settings.
- `backend/src/models/oauth_identity.py` - add encrypted token fields.
- `backend/src/services/google_oauth.py` - request Gmail scope and persist tokens.
- `backend/src/services/mail_service.py` - replace print stub with a small interface that calls Gmail service.
- `backend/src/services/interview_invitation_service.py` - create invitations without pretending email was sent.
- `backend/src/api/v1/endpoints/interview_templates.py` - enqueue invitation email task after invitation creation.
- `backend/src/api/v1/endpoints/outreach.py` - add authenticated send endpoint.
- `backend/worker/tasks.py` - add `send_interview_invitation_email` and `send_outreach_email` tasks.
- `frontend/src/api/endpoints/outreach.ts` - add `send`.
- `frontend/src/components/interviews/InvitationSendDialog.tsx` - update success/failure copy to reflect queued send.
- `frontend/src/routes/outreach.tsx` - add "Send email" action and remove reliance on manual-only status.
- `docs/BACKEND.md` and `QUICKSTART.md` - link to the setup guide.

---

## Task 1: Configuration And Token Encryption

**Files:**

- Modify: `backend/requirements.txt`
- Modify: `.env.example`
- Modify: `docker-compose.yml`
- Modify: `backend/src/core/config.py`
- Create: `backend/src/services/token_crypto.py`
- Test: `backend/tests/test_token_crypto.py`

- [ ] **Step 1: Write the failing token crypto tests**

Create `backend/tests/test_token_crypto.py`:

```python
import base64
import os

import pytest
from cryptography.fernet import Fernet

from src.services import token_crypto


def test_encrypt_then_decrypt_roundtrip(monkeypatch):
    key = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv("GOOGLE_TOKEN_ENCRYPTION_KEY", key)
    token_crypto.get_fernet.cache_clear()

    encrypted = token_crypto.encrypt_token("refresh-token-value")

    assert encrypted != "refresh-token-value"
    assert token_crypto.decrypt_token(encrypted) == "refresh-token-value"


def test_encrypt_rejects_missing_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_TOKEN_ENCRYPTION_KEY", raising=False)
    token_crypto.get_fernet.cache_clear()

    with pytest.raises(RuntimeError, match="GOOGLE_TOKEN_ENCRYPTION_KEY"):
        token_crypto.encrypt_token("token")


def test_generate_dev_key_shape():
    key = token_crypto.generate_fernet_key()
    raw = base64.urlsafe_b64decode(key.encode("ascii"))

    assert len(raw) == 32
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
pytest backend/tests/test_token_crypto.py -v
```

Expected: fail because `src.services.token_crypto` does not exist.

- [ ] **Step 3: Add dependency and settings**

Add to `backend/requirements.txt`:

```text
cryptography>=42.0.0
```

Add to `.env.example`:

```dotenv
# Google OAuth + Gmail API
GOOGLE_OAUTH_SCOPES=openid email profile https://www.googleapis.com/auth/gmail.send
GOOGLE_OAUTH_ACCESS_TYPE=offline
GOOGLE_OAUTH_PROMPT=consent
GOOGLE_TOKEN_ENCRYPTION_KEY=generate-with-python-command-in-docs
GMAIL_SEND_ENABLED=false
GMAIL_SEND_TIMEOUT_SECONDS=20
```

Add these environment entries to both `backend` and `worker` services in `docker-compose.yml`:

```yaml
      - GOOGLE_OAUTH_SCOPES=${GOOGLE_OAUTH_SCOPES:-openid email profile https://www.googleapis.com/auth/gmail.send}
      - GOOGLE_OAUTH_ACCESS_TYPE=${GOOGLE_OAUTH_ACCESS_TYPE:-offline}
      - GOOGLE_OAUTH_PROMPT=${GOOGLE_OAUTH_PROMPT:-consent}
      - GOOGLE_TOKEN_ENCRYPTION_KEY=${GOOGLE_TOKEN_ENCRYPTION_KEY:-}
      - GMAIL_SEND_ENABLED=${GMAIL_SEND_ENABLED:-false}
      - GMAIL_SEND_TIMEOUT_SECONDS=${GMAIL_SEND_TIMEOUT_SECONDS:-20}
```

Add fields to `Settings` in `backend/src/core/config.py`:

```python
    GOOGLE_OAUTH_SCOPES: str = os.getenv(
        "GOOGLE_OAUTH_SCOPES",
        "openid email profile https://www.googleapis.com/auth/gmail.send",
    )
    GOOGLE_OAUTH_ACCESS_TYPE: str = os.getenv("GOOGLE_OAUTH_ACCESS_TYPE", "offline")
    GOOGLE_OAUTH_PROMPT: str = os.getenv("GOOGLE_OAUTH_PROMPT", "consent")
    GOOGLE_TOKEN_ENCRYPTION_KEY: str = os.getenv("GOOGLE_TOKEN_ENCRYPTION_KEY", "")
    GMAIL_SEND_ENABLED: bool = os.getenv("GMAIL_SEND_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    GMAIL_SEND_TIMEOUT_SECONDS: int = int(os.getenv("GMAIL_SEND_TIMEOUT_SECONDS", "20"))
```

- [ ] **Step 4: Implement token crypto**

Create `backend/src/services/token_crypto.py`:

```python
from __future__ import annotations

from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from src.core.config import settings


def generate_fernet_key() -> str:
    return Fernet.generate_key().decode("ascii")


@lru_cache(maxsize=1)
def get_fernet() -> Fernet:
    key = settings.GOOGLE_TOKEN_ENCRYPTION_KEY
    if not key:
        raise RuntimeError("GOOGLE_TOKEN_ENCRYPTION_KEY is required to store Google OAuth tokens.")
    return Fernet(key.encode("ascii"))


def encrypt_token(token: str) -> str:
    return get_fernet().encrypt(token.encode("utf-8")).decode("ascii")


def decrypt_token(encrypted_token: str) -> str:
    try:
        return get_fernet().decrypt(encrypted_token.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise RuntimeError("Stored Google OAuth token could not be decrypted.") from exc
```

- [ ] **Step 5: Run tests and commit**

Run:

```powershell
pytest backend/tests/test_token_crypto.py -v
```

Expected: all tests pass.

Commit:

```powershell
git add backend/requirements.txt .env.example docker-compose.yml backend/src/core/config.py backend/src/services/token_crypto.py backend/tests/test_token_crypto.py
git commit -m "feat(email): add google token encryption config"
```

---

## Task 2: Persist Google Gmail Tokens

**Files:**

- Modify: `backend/src/models/oauth_identity.py`
- Modify: `backend/src/services/google_oauth.py`
- Create: `backend/migrations/versions/<revision>_add_google_mail_tokens.py`
- Modify: `backend/tests/test_google_oauth_service.py`

- [ ] **Step 1: Write failing OAuth token persistence tests**

Append to `backend/tests/test_google_oauth_service.py`:

```python
from datetime import datetime, timezone


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
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
pytest backend/tests/test_google_oauth_service.py -v
```

Expected: fail because `OAuthIdentity` has no token columns and `upsert_user_from_google` does not accept `tokens`.

- [ ] **Step 3: Add token fields to model**

Update `backend/src/models/oauth_identity.py`:

```python
from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint, func
```

Add these columns to `OAuthIdentity`:

```python
    access_token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    refresh_token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    scope: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
```

- [ ] **Step 4: Generate and review migration**

Run:

```powershell
cd backend
alembic revision --autogenerate -m "add google mail tokens"
alembic upgrade head
```

Expected migration operations:

```python
op.add_column("oauth_identities", sa.Column("access_token_encrypted", sa.Text(), nullable=True))
op.add_column("oauth_identities", sa.Column("refresh_token_encrypted", sa.Text(), nullable=True))
op.add_column("oauth_identities", sa.Column("token_expires_at", sa.DateTime(timezone=True), nullable=True))
op.add_column("oauth_identities", sa.Column("scope", sa.Text(), nullable=True))
op.add_column("oauth_identities", sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False))
```

If autogenerate adds unrelated operations, remove those unrelated operations from the migration before running `alembic upgrade head`.

- [ ] **Step 5: Persist tokens in Google OAuth service**

In `backend/src/services/google_oauth.py`, import:

```python
from datetime import datetime, timedelta, timezone
from src.services.token_crypto import encrypt_token
```

Change the authorize params in `build_authorize_url`:

```python
    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": settings.GOOGLE_OAUTH_SCOPES,
        "state": state,
        "access_type": settings.GOOGLE_OAUTH_ACCESS_TYPE,
        "prompt": settings.GOOGLE_OAUTH_PROMPT,
        "include_granted_scopes": "true",
    }
```

Add helper:

```python
def _apply_google_tokens(identity: OAuthIdentity, tokens: dict | None) -> None:
    if not tokens:
        return
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    expires_in = tokens.get("expires_in")
    scope = tokens.get("scope")

    if access_token:
        identity.access_token_encrypted = encrypt_token(access_token)
    if refresh_token:
        identity.refresh_token_encrypted = encrypt_token(refresh_token)
    if expires_in:
        identity.token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
    if scope:
        identity.scope = str(scope)
```

Change the function signature:

```python
def upsert_user_from_google(db: Session, claims: dict, tokens: dict | None = None) -> UserAccount:
```

When an existing identity is found:

```python
    if identity is not None:
        _apply_google_tokens(identity, tokens)
        db.commit()
        return identity.user
```

When a new `OAuthIdentity` is created for an existing user or a new user, call `_apply_google_tokens(new_identity, tokens)` before `db.add(new_identity)` or before `db.commit()`.

In `backend/src/api/v1/endpoints/auth.py`, pass tokens:

```python
        user = google_oauth.upsert_user_from_google(db, claims, tokens=tokens)
```

- [ ] **Step 6: Run tests and commit**

Run:

```powershell
pytest backend/tests/test_google_oauth_service.py -v
```

Expected: all Google OAuth service tests pass.

Commit:

```powershell
git add backend/src/models/oauth_identity.py backend/src/services/google_oauth.py backend/src/api/v1/endpoints/auth.py backend/tests/test_google_oauth_service.py backend/migrations/versions
git commit -m "feat(email): persist google gmail oauth tokens"
```

---

## Task 3: Gmail Transport Service

**Files:**

- Create: `backend/src/services/gmail_service.py`
- Modify: `backend/src/services/mail_service.py`
- Test: `backend/tests/test_gmail_service.py`

- [ ] **Step 1: Write failing Gmail service tests**

Create `backend/tests/test_gmail_service.py`:

```python
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
    assert identity.token_expires_at > datetime.now(timezone.utc)


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
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
pytest backend/tests/test_gmail_service.py -v
```

Expected: fail because `gmail_service.py` does not exist.

- [ ] **Step 3: Implement Gmail service**

Create `backend/src/services/gmail_service.py`:

```python
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
        raise GmailTokenMissingError("Google refresh token is missing. Ask the recruiter to reconnect Google.")

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
    identity.token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(payload.get("expires_in", 3600)))
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
```

- [ ] **Step 4: Replace mail service stub with interface**

Update `backend/src/services/mail_service.py`:

```python
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
```

- [ ] **Step 5: Run tests and commit**

Run:

```powershell
pytest backend/tests/test_gmail_service.py -v
```

Expected: all Gmail service tests pass.

Commit:

```powershell
git add backend/src/services/gmail_service.py backend/src/services/mail_service.py backend/tests/test_gmail_service.py
git commit -m "feat(email): add gmail api transport"
```

---

## Task 4: Email Templates

**Files:**

- Create: `backend/src/services/email_templates.py`
- Test: `backend/tests/test_email_templates.py`

- [ ] **Step 1: Write failing template tests**

Create `backend/tests/test_email_templates.py`:

```python
from src.services.email_templates import build_interview_invitation_email, build_outreach_email


def test_build_interview_invitation_email_contains_candidate_and_url():
    subject, body = build_interview_invitation_email(
        candidate_name="Candidate One",
        job_title="Platform Engineer",
        public_url="http://localhost:5173/interviews/token",
        expires_at_text="2026-06-08 10:00 UTC",
    )

    assert subject == "Interview invitation for Platform Engineer"
    assert "Hi Candidate One," in body
    assert "http://localhost:5173/interviews/token" in body
    assert "2026-06-08 10:00 UTC" in body


def test_build_outreach_email_trims_subject_and_body():
    subject, body = build_outreach_email(subject=" Hello ", body=" Body ")

    assert subject == "Hello"
    assert body == "Body"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
pytest backend/tests/test_email_templates.py -v
```

Expected: fail because `email_templates.py` does not exist.

- [ ] **Step 3: Implement templates**

Create `backend/src/services/email_templates.py`:

```python
from __future__ import annotations


def build_interview_invitation_email(
    *,
    candidate_name: str | None,
    job_title: str,
    public_url: str,
    expires_at_text: str | None,
) -> tuple[str, str]:
    display_name = candidate_name or "there"
    subject = f"Interview invitation for {job_title}"
    expiry_line = f"\nThis link is available until {expires_at_text}." if expires_at_text else ""
    body = (
        f"Hi {display_name},\n\n"
        f"Thank you for your interest in the {job_title} role. "
        "We would like to invite you to complete a short voice interview.\n\n"
        f"Interview link: {public_url}"
        f"{expiry_line}\n\n"
        "Best regards,\n"
        "Recruitment Team"
    )
    return subject, body


def build_outreach_email(*, subject: str, body: str) -> tuple[str, str]:
    return subject.strip(), body.strip()
```

- [ ] **Step 4: Run tests and commit**

Run:

```powershell
pytest backend/tests/test_email_templates.py -v
```

Expected: all template tests pass.

Commit:

```powershell
git add backend/src/services/email_templates.py backend/tests/test_email_templates.py
git commit -m "feat(email): add candidate email templates"
```

---

## Task 5: Send Interview Invitation Email Asynchronously

**Files:**

- Modify: `backend/src/services/interview_invitation_service.py`
- Modify: `backend/src/api/v1/endpoints/interview_templates.py`
- Modify: `backend/worker/tasks.py`
- Test: `backend/tests/test_interview_template_endpoints.py`
- Test: `backend/tests/test_interview_invitation_email.py`

- [ ] **Step 1: Write failing invitation behavior tests**

Create `backend/tests/test_interview_invitation_email.py` with focused service-level tests using existing test fixture patterns:

```python
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from worker import tasks


def test_send_interview_invitation_email_marks_sent_after_success(monkeypatch):
    invitation = SimpleNamespace(
        id="invitation-id",
        sent_at=None,
        status="pending",
        candidate_profile=SimpleNamespace(email="candidate@example.com", full_name="Candidate One"),
        job=SimpleNamespace(title="Platform Engineer"),
        sent_by_user_id="user-id",
        public_token="public-token",
        expires_at=None,
    )
    user = SimpleNamespace(email="recruiter@example.com")
    identity = SimpleNamespace(refresh_token_encrypted="encrypted-refresh")
    committed = {"value": False}

    class FakeDb:
        def get(self, model, key):
            name = getattr(model, "__name__", "")
            if name == "InterviewInvitation":
                return invitation
            if name == "UserAccount":
                return user
            return None
        def execute(self, statement):
            return SimpleNamespace(scalar_one_or_none=lambda: identity)
        def commit(self):
            committed["value"] = True
        def close(self):
            pass

    monkeypatch.setattr("src.models.session.SessionLocal", lambda: FakeDb())
    monkeypatch.setattr(
        "src.services.interview_invitation_service.build_interview_public_url",
        lambda token: f"http://test/interviews/{token}",
    )
    monkeypatch.setattr("src.services.mail_service.send_email", lambda **kwargs: {"id": "gmail-id"})

    result = tasks.send_interview_invitation_email.run("invitation-id")

    assert result == {"sent": True, "gmail_message_id": "gmail-id"}
    assert invitation.sent_at is not None
    assert committed["value"] is True
```

Append to `backend/tests/test_interview_template_endpoints.py`:

```python
def test_create_interview_invitation_does_not_set_sent_at_until_email_success(db_session, seeded_interview_domain):
    from src.schemas.interview_invitation import InterviewInvitationCreateRequest
    from src.services.interview_invitation_service import create_interview_invitation

    invitation = create_interview_invitation(
        db_session,
        user_id=seeded_interview_domain["user_id"],
        body=InterviewInvitationCreateRequest(
            job_id=seeded_interview_domain["primary_job_id"],
            candidate_profile_id=seeded_interview_domain["candidate_id"],
            interview_template_id=seeded_interview_domain["template_id"],
            expires_in_hours=72,
        ),
    )

    assert invitation.sent_at is None
    assert invitation.status == "pending"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
pytest backend/tests/test_interview_invitation_email.py backend/tests/test_interview_template_endpoints.py -v
```

Expected: fail because task does not exist and `create_interview_invitation` sets `sent_at`.

- [ ] **Step 3: Stop setting sent_at on invitation creation**

In `backend/src/services/interview_invitation_service.py`, change the `InterviewInvitation(...)` creation to remove `sent_at=datetime.now(timezone.utc)`.

The invitation should still store `sent_by_user_id=user_id`.

- [ ] **Step 4: Enqueue email task after invitation creation**

In `backend/src/api/v1/endpoints/interview_templates.py`, import:

```python
from worker.tasks import send_interview_invitation_email
```

After creating the invitation:

```python
    invitation = create_interview_invitation(db, user_id=current_user.id, body=body)
    send_interview_invitation_email.delay(str(invitation.id))
    return serialize_interview_invitation(invitation)
```

If importing `worker.tasks` at module import time creates circular import issues, move the import inside the endpoint function directly before `.delay(...)`.

- [ ] **Step 5: Implement Celery task**

In `backend/worker/tasks.py`, add imports inside the task body to avoid startup cycles:

```python
@celery_app.task(
    name="worker.tasks.send_interview_invitation_email",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
)
def send_interview_invitation_email(self, invitation_id: str):
    from datetime import datetime, timezone

    from sqlalchemy import select
    from sqlalchemy.orm import joinedload

    from src.models.interview_invitation import InterviewInvitation
    from src.models.oauth_identity import OAuthIdentity
    from src.models.session import SessionLocal
    from src.models.user_account import UserAccount
    from src.services.email_templates import build_interview_invitation_email
    from src.services.interview_invitation_service import build_interview_public_url
    from src.services.mail_service import send_email

    db = SessionLocal()
    try:
        invitation = (
            db.execute(
                select(InterviewInvitation)
                .options(
                    joinedload(InterviewInvitation.candidate_profile),
                    joinedload(InterviewInvitation.job),
                )
                .where(InterviewInvitation.id == uuid.UUID(invitation_id))
            )
            .scalars()
            .one_or_none()
        )
        if invitation is None:
            return {"sent": False, "reason": "invitation_not_found"}
        if invitation.sent_at is not None:
            return {"sent": True, "reason": "already_sent"}
        candidate_email = invitation.candidate_profile.email if invitation.candidate_profile else None
        if not candidate_email:
            invitation.status = "email_failed"
            db.commit()
            return {"sent": False, "reason": "candidate_email_missing"}
        user = db.get(UserAccount, invitation.sent_by_user_id)
        if user is None:
            invitation.status = "email_failed"
            db.commit()
            return {"sent": False, "reason": "sender_not_found"}
        identity = (
            db.execute(
                select(OAuthIdentity).where(
                    OAuthIdentity.user_id == user.id,
                    OAuthIdentity.provider == "google",
                )
            )
            .scalar_one_or_none()
        )
        if identity is None:
            invitation.status = "email_failed"
            db.commit()
            return {"sent": False, "reason": "google_identity_missing"}

        expires_at_text = invitation.expires_at.isoformat() if invitation.expires_at else None
        subject, body = build_interview_invitation_email(
            candidate_name=invitation.candidate_profile.full_name,
            job_title=invitation.job.title,
            public_url=build_interview_public_url(invitation.public_token),
            expires_at_text=expires_at_text,
        )
        result = send_email(
            sender=user.email,
            to_email=candidate_email,
            subject=subject,
            body=body,
            identity=identity,
        )
        invitation.sent_at = datetime.now(timezone.utc)
        invitation.status = "sent"
        db.commit()
        return {"sent": True, "gmail_message_id": result.get("id")}
    except Exception as exc:
        db.rollback()
        logger.exception("send_interview_invitation_email crashed for %s", invitation_id)
        raise self.retry(exc=exc)
    finally:
        db.close()
```

- [ ] **Step 6: Run tests and commit**

Run:

```powershell
pytest backend/tests/test_interview_invitation_email.py backend/tests/test_interview_template_endpoints.py -v
```

Expected: all targeted tests pass.

Commit:

```powershell
git add backend/src/services/interview_invitation_service.py backend/src/api/v1/endpoints/interview_templates.py backend/worker/tasks.py backend/tests/test_interview_invitation_email.py backend/tests/test_interview_template_endpoints.py
git commit -m "feat(email): send interview invitations through gmail"
```

---

## Task 6: Authenticated Outreach Send Endpoint

**Files:**

- Modify: `backend/src/api/v1/endpoints/outreach.py`
- Modify: `backend/worker/tasks.py`
- Test: `backend/tests/test_outreach_send_endpoint.py`

- [ ] **Step 1: Write failing outreach send tests**

Create `backend/tests/test_outreach_send_endpoint.py`:

```python
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.main import app
from src.models.base import Base
from src.models.candidate_profile import CandidateProfile
from src.models.deps import get_current_user, get_db
from src.models.enums import SentStatus
from src.models.enums import ContentSource, ProfileStatus, UploadStatus, UserStatus
from src.models.job import Job
from src.models.outreach import OutreachMessage
from src.models.resume_document import ResumeDocument
from src.models.user_account import UserAccount


def _create_test_tables(engine):
    tables = [
        Base.metadata.tables["user_accounts"],
        Base.metadata.tables["jobs"],
        Base.metadata.tables["resume_documents"],
        Base.metadata.tables["candidate_profiles"],
        Base.metadata.tables["outreach_messages"],
    ]
    Base.metadata.create_all(engine, tables=tables)


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    _create_test_tables(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    with factory() as session:
        yield session


@pytest.fixture()
def seeded_outreach_message(db_session: Session):
    user = UserAccount(
        email="owner@example.com",
        display_name="Owner",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    db_session.add(user)
    db_session.flush()

    job = Job(owner_user_id=user.id, title="Platform Engineer", status="active")
    db_session.add(job)
    db_session.flush()

    resume = ResumeDocument(
        original_file_name="candidate.pdf",
        storage_uri="s3://bucket/resumes/candidate.pdf",
        upload_status=UploadStatus.PROCESSED,
        job_id=job.id,
        uploaded_by_user_id=user.id,
        retention_expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
    )
    db_session.add(resume)
    db_session.flush()

    candidate = CandidateProfile(
        resume_document_id=resume.id,
        full_name="Candidate One",
        email="candidate@example.com",
        profile_status=ProfileStatus.REVIEWED,
    )
    db_session.add(candidate)
    db_session.flush()

    message = OutreachMessage(
        candidate_profile_id=candidate.id,
        created_by_user_id=user.id,
        content_source=ContentSource.AI_DRAFT,
        subject="Intro call",
        body="Would you be open to a short intro call?",
        sent_status=SentStatus.NOT_SENT,
    )
    db_session.add(message)
    db_session.commit()
    db_session.refresh(user)
    db_session.refresh(message)
    return {"user": user, "message": message}


@pytest.fixture()
def api_client(db_session: Session):
    def _override_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_db
    client = TestClient(app, follow_redirects=False)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture()
def authed_api_client(db_session: Session, seeded_outreach_message):
    def _override_db():
        yield db_session

    def _override_current_user():
        return seeded_outreach_message["user"]

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user] = _override_current_user
    client = TestClient(app, follow_redirects=False)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()


def test_send_outreach_requires_current_user(api_client, seeded_outreach_message):
    message = seeded_outreach_message["message"]
    response = api_client.post(f"/api/v1/outreach/{message.id}/send")

    assert response.status_code in {401, 403}


def test_send_outreach_queues_task_for_owner(authed_api_client, seeded_outreach_message, monkeypatch):
    import worker.tasks as tasks_module

    message = seeded_outreach_message["message"]
    queued = []

    class FakeTask:
        @staticmethod
        def delay(message_id):
            queued.append(message_id)

    monkeypatch.setattr(tasks_module, "send_outreach_email", FakeTask, raising=False)

    response = authed_api_client.post(f"/api/v1/outreach/{message.id}/send")

    assert response.status_code == 202
    assert response.json()["sent_status"] == SentStatus.NOT_SENT.value
    assert queued == [str(message.id)]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
pytest backend/tests/test_outreach_send_endpoint.py -v
```

Expected: fail because `/outreach/{id}/send` does not exist.

- [ ] **Step 3: Add authenticated send endpoint**

In `backend/src/api/v1/endpoints/outreach.py`, import:

```python
from fastapi import Depends, status
from src.models.deps import get_current_user, get_db
from src.models.user_account import UserAccount
```

Add route:

```python
@router.post("/{message_id}/send", response_model=OutreachResponse, status_code=status.HTTP_202_ACCEPTED)
def send_message(
    message_id: uuid.UUID,
    db=Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    msg = _get_or_404(db, OutreachMessage, message_id, "OutreachMessage")
    if msg.created_by_user_id != current_user.id:
        raise HTTPException(status_code=404, detail=f"OutreachMessage '{message_id}' not found")
    if msg.sent_status == SentStatus.SENT:
        return _ser(msg)
    if not msg.candidate_profile or not msg.candidate_profile.email:
        msg.sent_status = SentStatus.FAILED
        db.commit()
        db.refresh(msg)
        return _ser(msg)
    from worker.tasks import send_outreach_email

    send_outreach_email.delay(str(msg.id))
    return _ser(msg)
```

Do not trust `created_by_user_id` from frontend for send authorization. Existing create/list endpoints can be hardened in a later task, but this send endpoint must use `current_user`.

- [ ] **Step 4: Add Celery outreach task**

In `backend/worker/tasks.py`, add:

```python
@celery_app.task(
    name="worker.tasks.send_outreach_email",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
)
def send_outreach_email(self, message_id: str):
    from datetime import datetime, timezone

    from sqlalchemy import select
    from sqlalchemy.orm import joinedload

    from src.models.enums import SentStatus
    from src.models.oauth_identity import OAuthIdentity
    from src.models.outreach import OutreachMessage
    from src.models.session import SessionLocal
    from src.models.user_account import UserAccount
    from src.services.email_templates import build_outreach_email
    from src.services.mail_service import send_email

    db = SessionLocal()
    try:
        message = (
            db.execute(
                select(OutreachMessage)
                .options(joinedload(OutreachMessage.candidate_profile))
                .where(OutreachMessage.id == uuid.UUID(message_id))
            )
            .scalars()
            .one_or_none()
        )
        if message is None:
            return {"sent": False, "reason": "message_not_found"}
        if message.sent_status == SentStatus.SENT:
            return {"sent": True, "reason": "already_sent"}
        candidate_email = message.candidate_profile.email if message.candidate_profile else None
        if not candidate_email:
            message.sent_status = SentStatus.FAILED
            db.commit()
            return {"sent": False, "reason": "candidate_email_missing"}
        user = db.get(UserAccount, message.created_by_user_id)
        identity = (
            db.execute(
                select(OAuthIdentity).where(
                    OAuthIdentity.user_id == message.created_by_user_id,
                    OAuthIdentity.provider == "google",
                )
            )
            .scalar_one_or_none()
        )
        if user is None or identity is None:
            message.sent_status = SentStatus.FAILED
            db.commit()
            return {"sent": False, "reason": "sender_google_identity_missing"}
        subject, body = build_outreach_email(subject=message.subject, body=message.body)
        send_email(
            sender=user.email,
            to_email=candidate_email,
            subject=subject,
            body=body,
            identity=identity,
        )
        message.sent_status = SentStatus.SENT
        message.sent_at = datetime.now(timezone.utc)
        db.commit()
        return {"sent": True}
    except Exception as exc:
        db.rollback()
        logger.exception("send_outreach_email crashed for %s", message_id)
        raise self.retry(exc=exc)
    finally:
        db.close()
```

- [ ] **Step 5: Run tests and commit**

Run:

```powershell
pytest backend/tests/test_outreach_send_endpoint.py backend/tests/test_outreach_endpoints.py -v
```

Expected: targeted outreach tests pass.

Commit:

```powershell
git add backend/src/api/v1/endpoints/outreach.py backend/worker/tasks.py backend/tests/test_outreach_send_endpoint.py
git commit -m "feat(email): add outreach send endpoint"
```

---

## Task 7: Frontend Send Controls

**Files:**

- Modify: `frontend/src/api/endpoints/outreach.ts`
- Modify: `frontend/src/routes/outreach.tsx`
- Modify: `frontend/src/components/interviews/InvitationSendDialog.tsx`

- [ ] **Step 1: Add outreach send API client**

In `frontend/src/api/endpoints/outreach.ts`, add:

```typescript
  async send(messageId: string): Promise<OutreachResponse> {
    const { data } = await client.post<OutreachResponse>(`/outreach/${messageId}/send`);
    return data;
  },
```

No new type is required because the endpoint returns `OutreachResponse`.

- [ ] **Step 2: Replace manual send copy with real send action**

In `frontend/src/routes/outreach.tsx`, rename the `markSentMutation` to `sendMutation` and change:

```typescript
mutationFn: () => api.outreach.send(messageId!),
```

Update success and error to:

```typescript
toast.success("Email queued for sending");
```

```typescript
toast.error("Could not queue email. Check Google/Gmail setup and candidate email.");
```

Change button label from `Mark as sent` to `Send email`.

- [ ] **Step 3: Update invitation dialog success copy**

In `frontend/src/components/interviews/InvitationSendDialog.tsx`, change:

```typescript
toast.success("Interview invitation email queued");
```

If the endpoint continues to return the invitation immediately, keep the modal close behavior unchanged.

- [ ] **Step 4: Run frontend checks and commit**

Run:

```powershell
cd frontend
npm run typecheck
npm run build
```

Expected: both commands pass.

Commit:

```powershell
git add frontend/src/api/endpoints/outreach.ts frontend/src/routes/outreach.tsx frontend/src/components/interviews/InvitationSendDialog.tsx
git commit -m "feat(email): add frontend send controls"
```

---

## Task 8: Documentation And End-To-End Verification

**Files:**

- Modify: `docs/BACKEND.md`
- Modify: `QUICKSTART.md`
- Modify: `docs/GOOGLE_OAUTH_GMAIL_API_SETUP.md`

- [ ] **Step 1: Link setup guide from docs**

In `QUICKSTART.md`, add a short section after Google OAuth setup:

```markdown
## Configure Gmail Sending

To send candidate emails from the recruiter's real Gmail account, follow [docs/GOOGLE_OAUTH_GMAIL_API_SETUP.md](docs/GOOGLE_OAUTH_GMAIL_API_SETUP.md). Gmail sending requires `GMAIL_SEND_ENABLED=true`, `GOOGLE_TOKEN_ENCRYPTION_KEY`, and a Google OAuth consent screen that includes `https://www.googleapis.com/auth/gmail.send`.
```

In `docs/BACKEND.md`, add:

```markdown
## Candidate Email Sending

Candidate email uses Google OAuth and Gmail API. Recruiters sign in with Google and grant `gmail.send`. The backend stores encrypted OAuth tokens in `oauth_identities`, sends mail from Celery tasks, and updates `interview_invitations.sent_at` or `outreach_messages.sent_status` only after Gmail API accepts the message.
```

- [ ] **Step 2: Run backend and frontend test suites**

Run:

```powershell
pytest backend/tests/test_token_crypto.py backend/tests/test_google_oauth_service.py backend/tests/test_gmail_service.py backend/tests/test_email_templates.py backend/tests/test_interview_invitation_email.py backend/tests/test_outreach_send_endpoint.py -v
```

Expected: all targeted backend tests pass.

Run:

```powershell
cd frontend
npm run typecheck
npm run build
```

Expected: frontend typecheck and build pass.

- [ ] **Step 3: Manual smoke test with real Gmail**

Set root `.env`:

```dotenv
GOOGLE_OAUTH_SCOPES=openid email profile https://www.googleapis.com/auth/gmail.send
GOOGLE_OAUTH_ACCESS_TYPE=offline
GOOGLE_OAUTH_PROMPT=consent
GMAIL_SEND_ENABLED=true
GOOGLE_TOKEN_ENCRYPTION_KEY=<fernet-key-from-setup-guide>
```

Run:

```powershell
docker compose up --build
```

Manual verification:

1. Sign out of the app.
2. Sign in with Google and approve the Gmail send scope.
3. Upload or use an existing candidate profile with a real test recipient email.
4. Create an interview invitation.
5. Confirm the recipient receives the email from the recruiter's Gmail account.
6. Confirm `interview_invitations.sent_at` is set only after the worker sends.
7. Create an outreach draft.
8. Click `Send email`.
9. Confirm `outreach_messages.sent_status` becomes `sent` and `sent_at` is populated.

- [ ] **Step 4: Security review**

Run:

```powershell
rg -n -i "print\\(|logger\\.|token|secret|password" backend/src
```

Review results and confirm:

- No access token, refresh token, Gmail raw payload, client secret, or app JWT is logged.
- `.env` is not staged.
- `redirect` handling still rejects external URLs.
- `gmail.send` is the only Gmail scope requested.

- [ ] **Step 5: Commit docs**

Run:

```powershell
git add docs/BACKEND.md QUICKSTART.md docs/GOOGLE_OAUTH_GMAIL_API_SETUP.md
git commit -m "docs(email): document gmail sending setup"
```

---

## Rollback Plan

1. Set `GMAIL_SEND_ENABLED=false` in `.env` and restart backend/worker to stop real email sending immediately.
2. Revert frontend send controls if needed with `git revert <frontend-send-commit>`.
3. Revert backend email tasks and Gmail service commits if needed with `git revert`.
4. Keep token columns unless there is a compliance reason to remove them. If removal is required, run an Alembic downgrade for the token migration after confirming no active users depend on Google Gmail sending.
5. In Google Cloud Console, remove `https://www.googleapis.com/auth/gmail.send` from OAuth consent screen scopes if the feature is abandoned.

## Prompt For The Next Codex

Use this prompt in a new Codex session to execute the plan:

```text
Bạn đang làm việc trong repo C:\Users\HP\Desktop\Recruitment_AI_assistant.

Hãy triển khai chức năng gửi email thật cho ứng viên bằng Gmail API theo kế hoạch trong docs/superpowers/plans/2026-06-05-real-candidate-email-sending.md.

Yêu cầu:
- Trước khi sửa code, đọc AGENTS.md và dùng code-review-graph để lấy context repo.
- Không revert các thay đổi hiện có không phải do bạn tạo.
- Thực thi theo từng Task trong kế hoạch, ưu tiên TDD: viết test fail, implement, chạy test pass.
- Sau mỗi task hoàn chỉnh, commit riêng với commit message trong kế hoạch.
- Dùng Google OAuth hiện có, mở rộng scope sang https://www.googleapis.com/auth/gmail.send, lưu token đã mã hóa, gửi mail qua Celery worker.
- Không log access token, refresh token, client secret, app JWT, hoặc nội dung email nhạy cảm.
- Nếu gặp khác biệt fixture/test so với kế hoạch, chỉ điều chỉnh tên fixture cho khớp repo, không đổi hành vi mục tiêu.
- Sau cùng chạy targeted backend tests, frontend typecheck/build, và cập nhật tài liệu theo docs/GOOGLE_OAUTH_GMAIL_API_SETUP.md.

Bắt đầu bằng Task 1. Sau mỗi task, báo ngắn gọn test nào đã chạy và commit hash.
```

## Completion Criteria

- Recruiter can connect Google with `gmail.send`.
- Backend stores encrypted Google access/refresh tokens.
- Interview invitation email is sent by worker and `sent_at` reflects real send success.
- Outreach draft can be sent through `/outreach/{message_id}/send`.
- Candidate receives email from the recruiter's Gmail account.
- Test suite covers token encryption, OAuth token persistence, Gmail transport, invitation email, and outreach send.
- Setup guide explains Google Cloud and local `.env` configuration clearly.
