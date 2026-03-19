from __future__ import annotations

from datetime import datetime
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.dependencies.auth import CurrentUser, require_roles
from src.api.errors import AppError
from src.models.candidate import ParseStatus, ResumeDocument, UploadStatus
from src.repositories.db import get_session
from src.services.observability.audit_logger import audit_log
from src.services.parsing.resume_extractor import ResumeExtractionError
from src.services.storage.minio_client import MinioStorageClient
from worker.jobs.resume_ingestion_job import ingest_resume_payload

router = APIRouter(prefix="/v1/resumes", tags=["resumes"])


class UploadAccepted(BaseModel):
    jobId: str
    acceptedCount: int


@router.post("/upload", response_model=UploadAccepted, status_code=202)
def upload_resumes(
    files: Annotated[list[UploadFile], File(...)],
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[CurrentUser, Depends(require_roles("admin", "recruiter"))],
) -> UploadAccepted:
    if not files:
        raise AppError(code="bad_request", message="At least one PDF must be uploaded", status_code=400)

    storage = MinioStorageClient()
    accepted_count = 0
    upload_job_id = str(uuid.uuid4())

    try:
        user_id = uuid.UUID(current_user.user_id)
    except ValueError:
        user_id = uuid.UUID("00000000-0000-0000-0000-000000000000")

    for file in files:
        content_type = (file.content_type or "").lower()
        if "pdf" not in content_type and not file.filename.lower().endswith(".pdf"):
            continue

        payload = file.file.read()
        if not payload:
            continue

        object_key = f"resumes/{uuid.uuid4()}-{file.filename}"
        storage_uri = storage.upload_bytes(object_name=object_key, payload=payload)

        resume_document = ResumeDocument(
            original_file_name=file.filename,
            storage_provider="minio",
            minio_bucket=storage.bucket,
            minio_object_key=object_key,
            storage_uri=storage_uri,
            mime_type="application/pdf",
            upload_status=UploadStatus.UPLOADED,
            parse_status=ParseStatus.NOT_STARTED,
            uploaded_by_user_id=user_id,
        )
        session.add(resume_document)
        session.flush()

        try:
            ingest_resume_payload(session, resume_document=resume_document, payload=payload)
        except ResumeExtractionError as exc:
            resume_document.upload_status = UploadStatus.FAILED
            resume_document.parse_status = ParseStatus.FAILED
            resume_document.processed_at = datetime.utcnow()
            session.add(resume_document)
            audit_log(
                "resume_ingestion_failed",
                {
                    "resume_document_id": str(resume_document.id),
                    "reason": str(exc),
                },
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            resume_document.upload_status = UploadStatus.FAILED
            resume_document.parse_status = ParseStatus.FAILED
            resume_document.processed_at = datetime.utcnow()
            session.add(resume_document)
            audit_log(
                "resume_ingestion_failed",
                {
                    "resume_document_id": str(resume_document.id),
                    "reason": f"unexpected_error: {exc}",
                },
            )

        accepted_count += 1

    if accepted_count == 0:
        raise AppError(code="bad_request", message="No valid PDF files were accepted", status_code=400)

    session.commit()
    return UploadAccepted(jobId=upload_job_id, acceptedCount=accepted_count)
