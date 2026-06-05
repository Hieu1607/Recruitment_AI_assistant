import base64

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
