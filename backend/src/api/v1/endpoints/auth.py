import logging
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session
from src.core.config import settings
from src.core.security import create_access_token, get_password_hash, verify_password
from src.models.deps import get_current_user, get_db
from src.models.enums import RoleName, UserStatus
from src.models.oauth_identity import GMAIL_SEND_SCOPE
from src.models.user_account import RoleAssignment, UserAccount
from src.services import google_oauth

logger = logging.getLogger(__name__)

router = APIRouter()


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    display_name: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str


class MeResponse(UserResponse):
    gmail_connected: bool


class GoogleAuthorizeUrlResponse(BaseModel):
    authorize_url: str


class UpdateProfileRequest(BaseModel):
    display_name: Optional[str] = None
    email: Optional[str] = None


def _is_gmail_connected(current_user: UserAccount) -> bool:
    for identity in current_user.oauth_identities:
        if identity.provider != "google":
            continue
        if not identity.refresh_token_encrypted:
            continue
        if not identity.has_scope(GMAIL_SEND_SCOPE):
            continue
        return True
    return False


def _build_frontend_redirect(base_url: str, redirect_path: str, **params: str) -> str:
    destination = urlsplit(f"{base_url}{redirect_path}")
    query = dict(parse_qsl(destination.query, keep_blank_values=True))
    query.update(params)
    return urlunsplit(
        (
            destination.scheme,
            destination.netloc,
            destination.path,
            urlencode(query),
            destination.fragment,
        )
    )


@router.post(
    "/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED
)
def register(body: RegisterRequest, db: Session = Depends(get_db)) -> TokenResponse:
    existing = db.execute(
        select(UserAccount).where(UserAccount.email == body.email)
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Email already registered"
        )

    user = UserAccount(
        email=body.email,
        display_name=body.display_name,
        password_hash=get_password_hash(body.password),
        status=UserStatus.ACTIVE,
    )
    db.add(user)
    db.flush()
    db.add(RoleAssignment(user_id=user.id, role_name=RoleName.RECRUITER))
    db.commit()
    db.refresh(user)

    token = create_access_token(
        subject=str(user.id), email=user.email, display_name=user.display_name
    )
    return TokenResponse(access_token=token)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.execute(
        select(UserAccount).where(UserAccount.email == body.email)
    ).scalar_one_or_none()

    if user is None or user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )

    # If password_hash is set, verify it; legacy/seed users with no hash are accepted as-is (dev mode)
    if user.password_hash is not None and not verify_password(
        body.password, user.password_hash
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )

    token = create_access_token(
        subject=str(user.id), email=user.email, display_name=user.display_name
    )
    return TokenResponse(access_token=token)


@router.get("/me", response_model=MeResponse)
def get_me(current_user: UserAccount = Depends(get_current_user)) -> MeResponse:
    return MeResponse(
        id=str(current_user.id),
        email=current_user.email,
        display_name=current_user.display_name,
        gmail_connected=_is_gmail_connected(current_user),
    )


@router.patch("/me", response_model=UserResponse)
def update_me(
    body: UpdateProfileRequest,
    current_user: UserAccount = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserResponse:
    if body.display_name is not None:
        current_user.display_name = body.display_name
    if body.email is not None:
        existing = db.execute(
            select(UserAccount).where(
                UserAccount.email == body.email,
                UserAccount.id != current_user.id,
            )
        ).scalar_one_or_none()
        if existing is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="Email already in use"
            )
        current_user.email = body.email
    db.commit()
    db.refresh(current_user)
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        display_name=current_user.display_name,
    )


@router.get("/google/login")
def google_login(redirect: str = "/dashboard") -> RedirectResponse:
    frontend = settings.FRONTEND_BASE_URL.rstrip("/")
    if not redirect.startswith("/") or redirect.startswith("//"):
        redirect = "/dashboard"
    if not settings.GOOGLE_CLIENT_ID or not settings.GOOGLE_CLIENT_SECRET:
        logger.error("Google OAuth is not configured: missing client ID or secret")
        return RedirectResponse(
            f"{frontend}/login?error=oauth_not_configured",
            status_code=302,
        )
    url, _ = google_oauth.build_authorize_url(redirect)
    return RedirectResponse(url, status_code=302)


@router.get("/google/connect-gmail", response_model=GoogleAuthorizeUrlResponse)
def google_connect_gmail(
    redirect: str = "/outreach",
    current_user: UserAccount = Depends(get_current_user),
) -> GoogleAuthorizeUrlResponse:
    frontend = settings.FRONTEND_BASE_URL.rstrip("/")
    if not redirect.startswith("/") or redirect.startswith("//"):
        redirect = "/outreach"
    if not settings.GOOGLE_CLIENT_ID or not settings.GOOGLE_CLIENT_SECRET:
        logger.error("Google OAuth is not configured: missing client ID or secret")
        return GoogleAuthorizeUrlResponse(
            authorize_url=f"{frontend}{redirect}?error=oauth_not_configured"
        )
    url, _ = google_oauth.build_authorize_url(
        redirect,
        flow="connect_gmail",
        initiating_user_id=str(current_user.id),
    )
    return GoogleAuthorizeUrlResponse(authorize_url=url)


@router.get("/google/callback")
async def google_callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    db: Session = Depends(get_db),
) -> RedirectResponse:
    frontend = settings.FRONTEND_BASE_URL.rstrip("/")

    state_payload = None
    if state:
        try:
            state_payload = google_oauth.verify_state(state)
        except Exception:
            return RedirectResponse(f"{frontend}/login?error=invalid_state", status_code=302)

    if error:
        logger.warning("Google OAuth callback error: %s", error)
        if state_payload and state_payload["flow"] == "connect_gmail":
            return RedirectResponse(
                _build_frontend_redirect(
                    frontend,
                    state_payload["redirect"],
                    error="gmail_consent_denied",
                ),
                status_code=302,
            )
        return RedirectResponse(f"{frontend}/login?error={error}", status_code=302)
    if not code or not state:
        return RedirectResponse(f"{frontend}/login?error=missing_params", status_code=302)

    redirect_path = state_payload["redirect"]
    flow = state_payload["flow"]
    initiating_user_id = state_payload.get("initiating_user_id")

    try:
        tokens = await google_oauth.exchange_code_for_tokens(code)
        claims = google_oauth.verify_id_token(tokens["id_token"])
        if flow == "connect_gmail":
            if not initiating_user_id:
                raise ValueError("invalid_state")
            user = google_oauth.connect_google_identity_to_user(
                db,
                target_user_id=initiating_user_id,
                claims=claims,
                tokens=tokens,
            )
        else:
            user = google_oauth.upsert_user_from_google(db, claims, tokens=tokens)
    except ValueError as exc:
        logger.warning("Google OAuth upsert failed: %s", exc)
        if flow == "connect_gmail":
            return RedirectResponse(
                _build_frontend_redirect(frontend, redirect_path, error=str(exc)),
                status_code=302,
            )
        return RedirectResponse(f"{frontend}/login?error={exc}", status_code=302)
    except Exception:
        logger.exception("Google OAuth unexpected failure")
        if flow == "connect_gmail":
            return RedirectResponse(
                _build_frontend_redirect(frontend, redirect_path, error="oauth_failed"),
                status_code=302,
            )
        return RedirectResponse(f"{frontend}/login?error=oauth_failed", status_code=302)

    logger.info(
        "Google OAuth login: user_id=%s email=%s",
        user.id,
        user.email,
    )
    if flow == "connect_gmail":
        return RedirectResponse(
            _build_frontend_redirect(frontend, redirect_path, gmail_connected="1"),
            status_code=302,
        )
    app_token = create_access_token(
        subject=str(user.id), email=user.email, display_name=user.display_name
    )
    qs = urlencode({"token": app_token, "redirect": redirect_path})
    return RedirectResponse(f"{frontend}/auth/callback?{qs}", status_code=302)
