import logging
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)
from joserfc import jwt as joserfc_jwt
from joserfc.jwk import KeySet
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.config import settings
from src.models.oauth_identity import OAuthIdentity
from src.models.user_account import UserAccount, RoleAssignment
from src.models.enums import RoleName, UserStatus
from src.services.token_crypto import encrypt_token

GOOGLE_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_ISSUERS = {"https://accounts.google.com", "accounts.google.com"}

# In-memory JWKs cache: (keyset, fetched_at)
_jwks_cache: tuple[Optional[object], float] = (None, 0.0)
_JWKS_TTL = 3600.0  # 1 hour


def _get_serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(settings.SECRET_KEY, salt="google-oauth-state")


def build_authorize_url(redirect_path: str) -> tuple[str, str]:
    """Returns (authorize_url, state). redirect_path is the frontend path after login."""
    payload = {"redirect": redirect_path, "nonce": secrets.token_urlsafe(16)}
    state = _get_serializer().dumps(payload)

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
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{GOOGLE_AUTH_URL}?{query}"
    return url, state


def verify_state(state: str) -> str:
    """Returns the original redirect_path. Raises ValueError if invalid/expired."""
    try:
        payload = _get_serializer().loads(state, max_age=settings.OAUTH_STATE_TTL_SECONDS)
        return payload["redirect"]
    except SignatureExpired:
        raise ValueError("state_expired")
    except BadSignature:
        raise ValueError("state_invalid")
    except Exception:
        raise ValueError("state_invalid")


async def exchange_code_for_tokens(code: str) -> dict:
    """POSTs to Google token endpoint. Returns Google token response dict."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "redirect_uri": settings.GOOGLE_REDIRECT_URI,
            },
        )
    resp.raise_for_status()
    return resp.json()


def _fetch_jwks() -> KeySet:
    import httpx as _httpx
    resp = _httpx.get(GOOGLE_JWKS_URL, timeout=10)
    resp.raise_for_status()
    return KeySet.import_key_set(resp.json())


def _get_jwks() -> KeySet:
    global _jwks_cache
    keyset, fetched_at = _jwks_cache
    if keyset is None or (time.time() - fetched_at) > _JWKS_TTL:
        keyset = _fetch_jwks()
        _jwks_cache = (keyset, time.time())
    return keyset


def verify_id_token(id_token: str) -> dict:
    """Verifies signature against Google JWKs, checks aud, iss, exp. Returns claims dict."""
    keyset = _get_jwks()
    try:
        token = joserfc_jwt.decode(id_token, keyset)
    except Exception:
        # JWKs may be stale — refresh once and retry
        global _jwks_cache
        _jwks_cache = (None, 0.0)
        keyset = _get_jwks()
        token = joserfc_jwt.decode(id_token, keyset)

    claims = token.claims

    aud = claims.get("aud")
    if isinstance(aud, list):
        if settings.GOOGLE_CLIENT_ID not in aud:
            raise ValueError("id_token: aud mismatch")
    elif aud != settings.GOOGLE_CLIENT_ID:
        raise ValueError("id_token: aud mismatch")

    if claims.get("iss") not in GOOGLE_ISSUERS:
        raise ValueError("id_token: iss invalid")

    return dict(claims)


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


def upsert_user_from_google(db: Session, claims: dict, tokens: dict | None = None) -> UserAccount:
    """Implements email linking rule. Raises ValueError('google_email_not_verified') when appropriate."""
    google_sub = claims["sub"]
    email = claims["email"]
    email_verified = claims.get("email_verified", False)

    # 1. Look up by provider identity
    identity = db.execute(
        select(OAuthIdentity).where(
            OAuthIdentity.provider == "google",
            OAuthIdentity.provider_subject == google_sub,
        )
    ).scalar_one_or_none()

    if identity is not None:
        _apply_google_tokens(identity, tokens)
        db.commit()
        return identity.user

    # 2. Look up by email
    user = db.execute(
        select(UserAccount).where(UserAccount.email == email)
    ).scalar_one_or_none()

    if user is not None:
        if not email_verified:
            logger.warning("Google OAuth rejected: email not verified for email=%s", email)
            raise ValueError("google_email_not_verified")
        # Auto-link existing account
        new_identity = OAuthIdentity(
            user_id=user.id,
            provider="google",
            provider_subject=google_sub,
            email=email,
        )
        _apply_google_tokens(new_identity, tokens)
        db.add(new_identity)
        db.commit()
        logger.info("Google OAuth: linked google identity to existing user_id=%s email=%s", user.id, email)
        return user

    # 3. Create new user
    display_name = claims.get("name") or email.split("@")[0]
    new_user = UserAccount(
        email=email,
        display_name=display_name,
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    db.add(new_user)
    db.flush()  # get new_user.id

    role = RoleAssignment(user_id=new_user.id, role_name=RoleName.RECRUITER)
    db.add(role)

    identity = OAuthIdentity(
        user_id=new_user.id,
        provider="google",
        provider_subject=google_sub,
        email=email,
    )
    _apply_google_tokens(identity, tokens)
    db.add(identity)
    db.commit()
    db.refresh(new_user)
    logger.info("Google OAuth: created new user user_id=%s email=%s", new_user.id, email)
    return new_user
