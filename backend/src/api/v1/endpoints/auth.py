import logging
from typing import Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session
from src.core.config import settings
from src.core.security import create_access_token, get_password_hash, verify_password
from src.models.deps import get_current_user, get_db
from src.models.enums import RoleName, UserStatus
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


class UpdateProfileRequest(BaseModel):
    display_name: Optional[str] = None
    email: Optional[str] = None


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


@router.get("/me", response_model=UserResponse)
def get_me(current_user: UserAccount = Depends(get_current_user)) -> UserResponse:
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        display_name=current_user.display_name,
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


@router.get("/google/callback")
async def google_callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    db: Session = Depends(get_db),
) -> RedirectResponse:
    frontend = settings.FRONTEND_BASE_URL.rstrip("/")

    if error:
        logger.warning("Google OAuth callback error: %s", error)
        return RedirectResponse(f"{frontend}/login?error={error}", status_code=302)
    if not code or not state:
        return RedirectResponse(f"{frontend}/login?error=missing_params", status_code=302)

    try:
        redirect_path = google_oauth.verify_state(state)
    except Exception:
        return RedirectResponse(f"{frontend}/login?error=invalid_state", status_code=302)

    try:
        tokens = await google_oauth.exchange_code_for_tokens(code)
        claims = google_oauth.verify_id_token(tokens["id_token"])
        user = google_oauth.upsert_user_from_google(db, claims, tokens=tokens)
    except ValueError as exc:
        logger.warning("Google OAuth upsert failed: %s", exc)
        return RedirectResponse(f"{frontend}/login?error={exc}", status_code=302)
    except Exception:
        logger.exception("Google OAuth unexpected failure")
        return RedirectResponse(f"{frontend}/login?error=oauth_failed", status_code=302)

    logger.info(
        "Google OAuth login: user_id=%s email=%s",
        user.id,
        user.email,
    )
    app_token = create_access_token(
        subject=str(user.id), email=user.email, display_name=user.display_name
    )
    qs = urlencode({"token": app_token, "redirect": redirect_path})
    return RedirectResponse(f"{frontend}/auth/callback?{qs}", status_code=302)
