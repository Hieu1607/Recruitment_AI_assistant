from __future__ import annotations

import os
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

def generate_fernet_key() -> str:
    return Fernet.generate_key().decode("ascii")


@lru_cache(maxsize=1)
def get_fernet() -> Fernet:
    key = os.getenv("GOOGLE_TOKEN_ENCRYPTION_KEY")
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
