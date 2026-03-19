from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from src.models.candidate import CandidateProfile, ParseStatus, ResumeDocument, UploadStatus
from src.services.observability.audit_logger import audit_log
from src.services.parsing.extraction_trace_service import extraction_trace_service
from src.services.parsing.profile_normalizer import normalize_profile
from src.services.parsing.resume_extractor import extract_resume


def ingest_resume_payload(
    session: Session,
    *,
    resume_document: ResumeDocument,
    payload: bytes,
) -> CandidateProfile:
    resume_document.upload_status = UploadStatus.PROCESSING
    session.add(resume_document)
    session.flush()

    extraction = extract_resume(payload)
    normalized = normalize_profile(extraction)

    candidate = CandidateProfile(
        resume_document_id=resume_document.id,
        full_name=normalized.full_name,
        phone=normalized.phone,
        email=normalized.email,
        location_normalized=normalized.location_normalized,
        contact=normalized.contact,
        current_job_title=normalized.current_job_title,
        educated=normalized.educated,
        ever_studied_abroad=normalized.ever_studied_abroad,
        education_text=normalized.education_text,
        experience_text=normalized.experience_text,
        skills_text=normalized.skills_text,
        summary_text=normalized.summary_text,
    )
    session.add(candidate)
    session.flush()

    extraction_trace_service.persist_blocks(
        session,
        resume_document_id=resume_document.id,
        candidate_profile_id=candidate.id,
        blocks=extraction.blocks,
    )

    resume_document.upload_status = UploadStatus.PROCESSED
    resume_document.parse_status = ParseStatus.NORMALIZED
    resume_document.processed_at = datetime.utcnow()
    session.add(resume_document)

    audit_log(
        "resume_ingested",
        {
            "resume_document_id": str(resume_document.id),
            "candidate_profile_id": str(candidate.id),
            "used_ocr_fallback": extraction.used_ocr_fallback,
        },
    )
    return candidate
