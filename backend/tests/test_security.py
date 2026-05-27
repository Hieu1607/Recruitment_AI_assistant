import base64
import json
from datetime import datetime, timezone

from src.core.config import settings
from src.core.security import create_access_token


def test_create_access_token_uses_repo_default_expiry_window():
    token = create_access_token(subject="user-123")
    payload_segment = token.split(".")[1]
    payload_bytes = base64.urlsafe_b64decode(payload_segment + "=" * (-len(payload_segment) % 4))
    payload = json.loads(payload_bytes.decode("utf-8"))

    expires_at = datetime.fromisoformat(payload["exp"])
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    expires_in_minutes = (expires_at - now).total_seconds() / 60

    assert expires_in_minutes > 470
    assert expires_in_minutes <= 480
