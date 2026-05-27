from __future__ import annotations

import os
import secrets
from typing import TYPE_CHECKING

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from src.models.job import Job


def generate_public_apply_token() -> str:
    return secrets.token_urlsafe(32)


def build_public_apply_url(token: str) -> str:
    try:
        from src.core.config import settings

        base_url = settings.FRONTEND_BASE_URL.rstrip("/")
    except Exception:
        base_url = os.getenv("FRONTEND_BASE_URL", "http://localhost:5173").rstrip("/")
    return f"{base_url}/apply/{token}"


def resolve_public_job_by_token(db: Session, token: str) -> Job:
    from src.models.job import Job

    job = db.execute(select(Job).where(Job.public_apply_token == token)).scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Public application link not found")
    return job


def require_public_job_enabled(job: Job) -> Job:
    if not job.public_apply_enabled:
        raise HTTPException(status_code=410, detail="Public application link is disabled")
    return job
