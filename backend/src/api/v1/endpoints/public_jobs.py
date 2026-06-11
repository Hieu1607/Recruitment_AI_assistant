from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session
from src.models.deps import get_db
from src.services.object_storage import build_object_key, get_object_storage
from src.services.public_job_service import (
    require_public_job_enabled,
    resolve_public_job_by_token,
)
from src.services.resume_service import create_resume_document
from worker.tasks import process_resume

router = APIRouter()
# Legacy test hook; uploads now go to object storage.
PROJECT_ROOT = Path(__file__).resolve().parents[5]
PDF_STORAGE_DIR = PROJECT_ROOT / "pdfs"
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_FULL_NAME_LENGTH = 255
MAX_EMAIL_LENGTH = 320


class PublicJobResponse(BaseModel):
    job_title: str
    candidate_message: str | None
    public_apply_enabled: bool


class PublicResumeUploadResponse(BaseModel):
    submitted: bool
    resume_document_id: str | None = None
    status: str = "queued"
    task_id: str | None = None


def _validate_full_name(full_name: str) -> str:
    candidate = full_name.strip()
    if not candidate:
        raise HTTPException(status_code=422, detail="full_name is required")
    if len(candidate) > MAX_FULL_NAME_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"full_name must be <= {MAX_FULL_NAME_LENGTH} characters",
        )
    return candidate


def _validate_email(email: str) -> str:
    candidate = email.strip()
    if not candidate:
        raise HTTPException(status_code=422, detail="email is required")
    if len(candidate) > MAX_EMAIL_LENGTH:
        raise HTTPException(
            status_code=422, detail=f"email must be <= {MAX_EMAIL_LENGTH} characters"
        )
    if not EMAIL_PATTERN.match(candidate):
        raise HTTPException(
            status_code=422, detail="email must be a valid email address"
        )
    return candidate


def _validate_pdf(file: UploadFile) -> str:
    if not file.filename:
        raise HTTPException(status_code=400, detail="file is required")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    if file.content_type and file.content_type not in {
        "application/pdf",
        "application/octet-stream",
    }:
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    return Path(file.filename).name


@router.get("/jobs/{token}", response_model=PublicJobResponse)
def get_public_job(
    token: str,
    db: Session = Depends(get_db),
):
    job = require_public_job_enabled(resolve_public_job_by_token(db, token))
    return PublicJobResponse(
        job_title=job.title,
        candidate_message=job.candidate_message,
        public_apply_enabled=job.public_apply_enabled,
    )


@router.post(
    "/jobs/{token}/resumes", response_model=PublicResumeUploadResponse, status_code=202
)
async def upload_public_resume(
    token: str,
    full_name: str = Form(...),
    email: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    job = require_public_job_enabled(resolve_public_job_by_token(db, token))
    normalized_full_name = _validate_full_name(full_name)
    normalized_email = _validate_email(email)
    original_name = _validate_pdf(file)

    storage_uri = get_object_storage().upload_bytes(
        data=await file.read(),
        object_key=build_object_key(
            prefix=f"resumes/{job.id}",
            original_filename=original_name,
        ),
        content_type=file.content_type or "application/pdf",
    )

    resume = create_resume_document(
        db=db,
        storage_uri=storage_uri,
        original_file_name=original_name,
        job_id=job.id,
        uploaded_by_user_id=job.owner_user_id,
    )
    task = process_resume.delay(
        str(resume.id),
        submitted_full_name=normalized_full_name,
        submitted_email=normalized_email,
    )
    return PublicResumeUploadResponse(
        submitted=True,
        resume_document_id=str(resume.id),
        status="queued",
        task_id=task.id,
    )
