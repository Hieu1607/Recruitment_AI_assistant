from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header

from src.api.errors import ForbiddenError
from src.api.security.hardening import normalize_role_header


@dataclass
class CurrentUser:
    user_id: str
    role: str


def _get_role_from_headers(x_role: Annotated[str | None, Header()] = None) -> str:
    fallback_role = os.getenv("DEFAULT_ROLE", "viewer")
    return normalize_role_header(x_role, fallback=fallback_role)


def get_current_user(role: Annotated[str, Depends(_get_role_from_headers)]) -> CurrentUser:
    return CurrentUser(user_id="00000000-0000-0000-0000-000000000000", role=role)


def require_roles(*allowed_roles: str):
    allowed = {role.lower() for role in allowed_roles}

    def _guard(current_user: Annotated[CurrentUser, Depends(get_current_user)]) -> CurrentUser:
        if current_user.role not in allowed:
            raise ForbiddenError("You do not have permission to perform this action")
        return current_user

    return _guard
