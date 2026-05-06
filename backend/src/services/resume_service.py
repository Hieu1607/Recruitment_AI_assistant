import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import fitz  # PyMuPDF
import requests
from sqlalchemy.orm import Session
from src.core.config import settings
from src.models.candidate_profile import CandidateProfile
from src.models.enums import ProfileStatus, UploadStatus
from src.models.resume_document import ExtractionTrace, ResumeDocument
from src.prompts.build_prompts import build_prompts
from src.services.llm_service import LLMProvider


def extract_text_from_pdf(filepath: str) -> str:
    text = ""
    try:
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as exc:
        print(f"Error extracting text from {filepath}: {exc}")
        return ""


def extract_text_via_hf_ocr(filepath: str) -> str:
    """Fallback for image-based PDFs: submit to HF Tesseract OCR space and return text."""
    base_url = settings.HF_OCR_BASE_URL.rstrip("/")
    with open(filepath, "rb") as f:
        resp = requests.post(
            f"{base_url}/ocr/submit",
            files={"file": (Path(filepath).name, f, "application/pdf")},
            timeout=60,
        )
    resp.raise_for_status()
    job_id = resp.json()["job_id"]

    deadline = time.monotonic() + settings.HF_OCR_POLL_TIMEOUT
    while time.monotonic() < deadline:
        status_resp = requests.get(f"{base_url}/ocr/status/{job_id}", timeout=10)
        status_resp.raise_for_status()
        data = status_resp.json()
        if data["status"] == "done":
            requests.delete(f"{base_url}/ocr/job/{job_id}", timeout=10)
            return data.get("text") or ""
        if data["status"] == "error":
            raise RuntimeError(f"HF OCR job failed: {data.get('error')}")
        time.sleep(settings.HF_OCR_POLL_INTERVAL)

    raise TimeoutError(
        f"HF OCR job {job_id} timed out after {settings.HF_OCR_POLL_TIMEOUT}s"
    )


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    content = (raw_text or "").strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            content = "\n".join(lines[1:-1]).strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM did not return a valid JSON object")

    candidate = content[start : end + 1]
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Parsed LLM response is not a JSON object")
    return parsed


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def _normalize_decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _build_profile_from_parsed(
    resume_document_id: uuid.UUID, parsed: Dict[str, Any]
) -> CandidateProfile:
    full_name = parsed.get("name") or "Unknown Candidate"

    return CandidateProfile(
        resume_document_id=resume_document_id,
        full_name=str(full_name),
        phone=parsed.get("phone"),
        email=parsed.get("email"),
        location_normalized=parsed.get("location"),
        contact=parsed.get("contact"),
        current_job_title=parsed.get("current_job_title"),
        educated=_normalize_bool(parsed.get("educated")),
        ever_studied_abroad=_normalize_bool(parsed.get("ever_studied_abroad")),
        major=parsed.get("major"),
        cpa=parsed.get("cpa"),
        education_text=parsed.get("education"),
        experience_text=parsed.get("experience"),
        experience_years=_normalize_decimal(parsed.get("experience_years")),
        skills_text=parsed.get("skills"),
        languages_text=parsed.get("languages"),
        projects_text=parsed.get("projects"),
        summary_text=parsed.get("summary"),
        achievements_text=parsed.get("achievements"),
        publications_text=parsed.get("publications"),
        certifications_text=parsed.get("certifications"),
        references_text=parsed.get("references"),
        other_text=parsed.get("other"),
        profile_status=ProfileStatus.REVIEWED.value,
    )


def parse_pdf_to_sections(
    filepaths: Sequence[str],
    db: Session,
    job_id: uuid.UUID,
    uploaded_by_user_id: Optional[uuid.UUID] = None,
    retention_days: int = 365,
    original_filenames: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Extract CV text from PDFs, parse with LLM prompt, and persist results."""
    if not filepaths:
        return []

    actor_id = uploaded_by_user_id or uuid.UUID("00000000-0000-0000-0000-000000000000")
    llm_provider = LLMProvider()
    results: List[Dict[str, Any]] = []

    for idx, raw_path in enumerate(filepaths):
        source_path = Path(raw_path)
        display_name = (
            original_filenames[idx]
            if original_filenames and idx < len(original_filenames)
            else source_path.name
        )
        resume = ResumeDocument(
            original_file_name=display_name,
            storage_uri=str(source_path),
            upload_status=UploadStatus.UPLOADED.value,
            job_id=job_id,
            uploaded_by_user_id=actor_id,
            retention_expires_at=datetime.now(timezone.utc)
            + timedelta(days=retention_days),
        )
        db.add(resume)
        db.commit()
        db.refresh(resume)

        try:
            resume.upload_status = UploadStatus.PROCESSING.value
            db.add(
                ExtractionTrace(
                    resume_document_id=resume.id,
                    stage="pipeline",
                    status="processing",
                    message="Started CV extraction and parsing",
                )
            )
            db.commit()

            cv_text = extract_text_from_pdf(str(source_path))
            if not cv_text.strip():
                cv_text = extract_text_via_hf_ocr(str(source_path))
            prompt = build_prompts.build_cv_parsing_prompt(cv_text)
            llm_response = llm_provider.generate(prompt)
            parsed_payload = _extract_json_object(llm_response.text)

            profile = _build_profile_from_parsed(resume.id, parsed_payload)
            resume.upload_status = UploadStatus.PROCESSED.value
            resume.processed_at = datetime.now(timezone.utc)

            db.add(profile)
            db.add(
                ExtractionTrace(
                    resume_document_id=resume.id,
                    stage="cv_parsing",
                    status="success",
                    message="CV parsed and profile created",
                    payload={
                        "candidateName": profile.full_name,
                        "llmProvider": llm_response.provider,
                        "llmModel": llm_response.model,
                    },
                )
            )
            db.commit()
            db.refresh(profile)

            results.append(
                {
                    "file_name": display_name,
                    "resume_document_id": str(resume.id),
                    "candidate_profile_id": str(profile.id),
                    "status": "processed",
                }
            )
        except Exception as exc:
            db.rollback()
            resume_db = db.get(ResumeDocument, resume.id)
            if resume_db is not None:
                resume_db.upload_status = UploadStatus.FAILED.value
                db.add(
                    ExtractionTrace(
                        resume_document_id=resume_db.id,
                        stage="cv_parsing",
                        status="failed",
                        message=str(exc),
                    )
                )
                db.commit()

            results.append(
                {
                    "file_name": display_name,
                    "resume_document_id": str(resume.id),
                    "candidate_profile_id": None,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    return results


def batch_score_CVs():
    # Placeholder for a function that would take the parsed sections and score them against a job description
    pass


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def _resume_to_dict(resume: ResumeDocument) -> Dict[str, Any]:
    return {
        "id": str(resume.id),
        "job_id": str(resume.job_id),
        "original_file_name": resume.original_file_name,
        "storage_uri": resume.storage_uri,
        "upload_status": (
            resume.upload_status.value
            if hasattr(resume.upload_status, "value")
            else resume.upload_status
        ),
        "duplicate_group_key": resume.duplicate_group_key,
        "uploaded_by_user_id": str(resume.uploaded_by_user_id),
        "uploaded_at": resume.uploaded_at.isoformat() if resume.uploaded_at else None,
        "processed_at": (
            resume.processed_at.isoformat() if resume.processed_at else None
        ),
        "retention_expires_at": (
            resume.retention_expires_at.isoformat()
            if resume.retention_expires_at
            else None
        ),
    }


def get_resume(
    *,
    db: Session,
    resume_id: uuid.UUID,
    job_id: Optional[uuid.UUID] = None,
) -> Optional[Dict[str, Any]]:
    """Return a single ResumeDocument dict, or None if not found."""
    resume = db.get(ResumeDocument, resume_id)
    if resume is None:
        return None
    if job_id is not None and resume.job_id != job_id:
        return None
    return _resume_to_dict(resume)


def list_resumes(
    *,
    db: Session,
    job_id: Optional[uuid.UUID] = None,
    upload_status: Optional[str] = None,
    uploaded_by_user_id: Optional[uuid.UUID] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Return a paginated list of ResumeDocument dicts.

    Args:
        upload_status: Filter by status string (e.g. 'processed', 'failed').
        uploaded_by_user_id: Filter by uploader UUID.
        limit: Max records (default 50).
        offset: Records to skip (default 0).
    """
    query = db.query(ResumeDocument)
    if job_id is not None:
        query = query.filter(ResumeDocument.job_id == job_id)
    if upload_status is not None:
        query = query.filter(ResumeDocument.upload_status == upload_status)
    if uploaded_by_user_id is not None:
        query = query.filter(ResumeDocument.uploaded_by_user_id == uploaded_by_user_id)
    query = query.order_by(ResumeDocument.uploaded_at.desc())
    resumes = query.offset(offset).limit(limit).all()
    return [_resume_to_dict(r) for r in resumes]


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


def update_resume(
    *,
    db: Session,
    resume_id: uuid.UUID,
    original_file_name: Optional[str] = None,
    upload_status: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Partially update a ResumeDocument.

    Only ``original_file_name`` and ``upload_status`` may be changed through
    the API (storage_uri is managed internally).

    Returns the updated dict, or None if not found.

    Raises:
        ValueError: if ``upload_status`` is not a valid UploadStatus value.
    """
    resume = db.get(ResumeDocument, resume_id)
    if resume is None:
        return None

    if original_file_name is not None:
        if not original_file_name.strip():
            raise ValueError("original_file_name must not be empty")
        resume.original_file_name = original_file_name.strip()

    if upload_status is not None:
        valid_statuses = {s.value for s in UploadStatus}
        if upload_status not in valid_statuses:
            raise ValueError(f"upload_status must be one of: {sorted(valid_statuses)}")
        resume.upload_status = upload_status

    db.add(resume)
    db.commit()
    db.refresh(resume)
    return _resume_to_dict(resume)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def delete_resume(
    *,
    db: Session,
    resume_id: uuid.UUID,
    delete_file: bool = False,
) -> bool:
    """Hard-delete a ResumeDocument (and its cascade relations).

    Args:
        delete_file: When True, also removes the physical PDF from disk.

    Returns True if deleted, False if not found.
    """
    resume = db.get(ResumeDocument, resume_id)
    if resume is None:
        return False

    storage_uri = resume.storage_uri
    db.delete(resume)
    db.commit()

    if delete_file and storage_uri:
        try:
            Path(storage_uri).unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not delete file {storage_uri}: {exc}")

    return True
