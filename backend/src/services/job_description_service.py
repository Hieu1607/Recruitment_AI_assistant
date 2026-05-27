"""
job_description_service.py
Service layer for CRUD operations on JobDescription records.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.models.job_matching import JobDescription


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jd_to_dict(jd: JobDescription) -> Dict[str, Any]:
    return {
        "id": str(jd.id),
        "job_id": str(jd.job_id),
        "title": jd.title,
        "jd_text": jd.jd_text,
        "hidden_text": getattr(jd, "hidden_text", "") or "",
        "created_by_user_id": str(jd.created_by_user_id),
        "created_at": jd.created_at.isoformat(),
        "is_active": jd.is_active,
    }


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------

def create_job_description(
    *,
    db: Session,
    job_id: uuid.UUID,
    jd_text: str,
    created_by_user_id: uuid.UUID,
    title: Optional[str] = None,
    hidden_text: str = "",
) -> Dict[str, Any]:
    """Persist a new JobDescription and return its dict representation."""
    jd = JobDescription(
        job_id=job_id,
        title=title,
        jd_text=jd_text.strip() if jd_text else "",
        hidden_text=hidden_text.strip() if hidden_text else "",
        created_by_user_id=created_by_user_id,
        is_active=True,
    )
    db.add(jd)
    db.commit()
    db.refresh(jd)
    return _jd_to_dict(jd)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_job_description(
    *,
    db: Session,
    jd_id: uuid.UUID,
    job_id: Optional[uuid.UUID] = None,
) -> Optional[Dict[str, Any]]:
    """Return a single JobDescription dict, or None if not found."""
    jd = db.get(JobDescription, jd_id)
    if jd is None:
        return None
    if job_id is not None and jd.job_id != job_id:
        return None
    return _jd_to_dict(jd)


def list_job_descriptions(
    *,
    db: Session,
    job_id: Optional[uuid.UUID] = None,
    is_active: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Return a paginated list of JobDescription dicts.

    Args:
        is_active: When provided, filters records by ``is_active`` flag.
        limit: Maximum number of records to return (default 50).
        offset: Number of records to skip (default 0).
    """
    query = db.query(JobDescription)
    if job_id is not None:
        query = query.filter(JobDescription.job_id == job_id)
    if is_active is not None:
        query = query.filter(JobDescription.is_active == is_active)
    query = query.order_by(JobDescription.created_at.desc())
    jds = query.offset(offset).limit(limit).all()
    return [_jd_to_dict(jd) for jd in jds]


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

def update_job_description(
    *,
    db: Session,
    jd_id: uuid.UUID,
    title: Optional[str] = None,
    jd_text: Optional[str] = None,
    hidden_text: Optional[str] = None,
    is_active: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Partially update a JobDescription.

    Only fields that are explicitly passed (not None) are changed.
    Returns the updated dict, or None if the record was not found.

    Raises:
        ValueError: If ``jd_text`` is provided but blank.
    """
    jd = db.get(JobDescription, jd_id)
    if jd is None:
        return None

    if title is not None:
        jd.title = title

    if jd_text is not None:
        if not jd_text.strip():
            raise ValueError("jd_text must not be empty")
        jd.jd_text = jd_text.strip()

    if hidden_text is not None:
        jd.hidden_text = hidden_text.strip()

    if is_active is not None:
        jd.is_active = is_active

    db.add(jd)
    db.commit()
    db.refresh(jd)
    return _jd_to_dict(jd)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def delete_job_description(
    *,
    db: Session,
    jd_id: uuid.UUID,
) -> bool:
    """Hard-delete a JobDescription.

    Returns True if deleted, False if not found.
    """
    jd = db.get(JobDescription, jd_id)
    if jd is None:
        return False
    db.delete(jd)
    db.commit()
    return True
