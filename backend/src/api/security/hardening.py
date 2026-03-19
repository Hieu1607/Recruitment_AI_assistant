from __future__ import annotations

import logging
import os


LOGGER = logging.getLogger("recruitment.security")

_ALLOWED_ROLES = {"admin", "recruiter", "viewer"}
_PLACEHOLDER_VALUES = {
    "changeme",
    "replace_me",
    "your_value_here",
    "example",
    "dummy",
    "test",
}


def normalize_role_header(role_header: str | None, fallback: str = "viewer") -> str:
    """Normalize role input and fail closed to viewer for unsupported values."""
    role = (role_header or fallback or "viewer").strip().lower()
    if role in _ALLOWED_ROLES:
        return role
    return "viewer"


def validate_runtime_security() -> None:
    """Validate critical security settings at startup in production mode."""
    app_env = os.getenv("APP_ENV", "development").strip().lower()
    if app_env != "production":
        return

    required_secrets = [
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "SMTP_PASSWORD",
    ]

    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    if provider == "groq":
        required_secrets.append("GROQ_API_KEY")

    issues: list[str] = []
    for key in required_secrets:
        value = os.getenv(key)
        if value is None or not value.strip():
            issues.append(f"{key} is missing")
            continue
        if value.strip().lower() in _PLACEHOLDER_VALUES:
            issues.append(f"{key} uses a placeholder value")

    default_role = os.getenv("DEFAULT_ROLE", "viewer")
    normalized_role = normalize_role_header(default_role, fallback="viewer")
    if normalized_role != default_role.strip().lower():
        LOGGER.warning("DEFAULT_ROLE was invalid and normalized to viewer")

    if issues:
        message = "Production security validation failed: " + "; ".join(issues)
        raise RuntimeError(message)
