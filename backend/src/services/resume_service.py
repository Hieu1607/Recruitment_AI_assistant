import json
import logging
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
from src.services.object_storage import get_object_storage, parse_storage_uri

_HF_OCR_MAX_ATTEMPTS = 3
_HF_OCR_RETRYABLE_STATUS_CODES = {500, 502, 503, 504}
_VISION_FALLBACK_MAX_PAGES = 3
_RESUME_PARSE_MAX_TOKENS = 2500
_STRUCTURED_SECTION_TEXT_FIELDS = {
    "experience": "experience",
    "education": "education",
    "projects": "projects",
    "skills": "skills",
    "languages": "languages",
    "achievements": "achievements",
    "publications": "publications",
    "certifications": "certifications",
    "references": "references",
    "other": "other",
}
logger = logging.getLogger(__name__)


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    normalized = value.strip()
    return normalized or None


def _resume_llm_provider() -> LLMProvider:
    return LLMProvider(max_tokens=max(settings.LLM_MAX_TOKENS, _RESUME_PARSE_MAX_TOKENS))


def _normalize_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    items: List[str] = []
    for item in value:
        normalized = _normalize_text(item)
        if normalized:
            items.append(normalized)
    return items


def _normalize_structured_link(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, str):
        normalized_url = _normalize_text(value)
        if normalized_url:
            return {"url": normalized_url, "label": None}
        return None
    if not isinstance(value, dict):
        return None
    normalized_url = _normalize_text(value.get("url"))
    if not normalized_url:
        return None
    return {
        "url": normalized_url,
        "label": _normalize_text(value.get("label")),
    }


def _normalize_structured_links(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    links: List[Dict[str, Any]] = []
    for item in value:
        normalized = _normalize_structured_link(item)
        if normalized is not None:
            links.append(normalized)
    return links


def _normalize_structured_entry(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None

    entry = {
        "title": _normalize_text(value.get("title")),
        "subtitle": _normalize_text(value.get("subtitle")),
        "role": _normalize_text(value.get("role")),
        "location": _normalize_text(value.get("location")),
        "dateRange": _normalize_text(value.get("dateRange")),
        "description": _normalize_text(value.get("description")),
        "bullets": _normalize_string_list(value.get("bullets")),
        "links": _normalize_structured_links(value.get("links")),
        "metadata": _normalize_string_list(value.get("metadata")),
    }

    if any(
        [
            entry["title"],
            entry["subtitle"],
            entry["role"],
            entry["location"],
            entry["dateRange"],
            entry["description"],
            entry["bullets"],
            entry["links"],
            entry["metadata"],
        ]
    ):
        return entry
    return None


def _normalize_structured_section(
    value: Any,
    *,
    fallback_text: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    raw_text = _normalize_text(fallback_text)

    if isinstance(value, dict):
        if isinstance(value.get("entries"), list):
            for item in value.get("entries", []):
                normalized_item = _normalize_structured_entry(item)
                if normalized_item is not None:
                    entries.append(normalized_item)
        raw_text = _normalize_text(value.get("rawText")) or raw_text

    if not entries and raw_text is None:
        return None

    return {
        "entries": entries,
        "rawText": raw_text,
    }


def _normalize_structured_summary(
    value: Any,
    *,
    fallback_text: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    text = _normalize_text(fallback_text)
    links: List[Dict[str, Any]] = []

    if isinstance(value, dict):
        text = _normalize_text(value.get("text")) or text
        links = _normalize_structured_links(value.get("links"))

    if text is None and not links:
        return None

    return {
        "text": text,
        "links": links,
    }


def _normalize_structured_profile(
    value: Any,
    parsed: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    payload = value if isinstance(value, dict) else {}
    structured: Dict[str, Any] = {}

    summary = _normalize_structured_summary(
        payload.get("summary"),
        fallback_text=_normalize_text(parsed.get("summary")),
    )
    if summary is not None:
        structured["summary"] = summary

    for section_name, parsed_key in _STRUCTURED_SECTION_TEXT_FIELDS.items():
        section = _normalize_structured_section(
            payload.get(section_name),
            fallback_text=_normalize_text(parsed.get(parsed_key)),
        )
        if section is not None:
            structured[section_name] = section

    return structured or None


def extract_text_from_pdf(pdf_source: bytes | str) -> str:
    text = ""
    source_name = pdf_source if isinstance(pdf_source, str) else "<in-memory-pdf>"
    try:
        if isinstance(pdf_source, (bytes, bytearray)):
            doc = fitz.open(stream=bytes(pdf_source), filetype="pdf")
        else:
            doc = fitz.open(pdf_source)
        with doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as exc:
        print(f"Error extracting text from {source_name}: {exc}")
        return ""


def extract_text_via_hf_ocr(
    pdf_source: bytes | str, filename: Optional[str] = None
) -> str:
    """Fallback for image-based PDFs: submit to HF Tesseract OCR space and return text."""
    base_url = settings.HF_OCR_BASE_URL.rstrip("/")
    if isinstance(pdf_source, (bytes, bytearray)):
        upload_name = Path(filename or "resume.pdf").name
        pdf_bytes = bytes(pdf_source)
    else:
        upload_name = Path(pdf_source).name
        pdf_bytes = Path(pdf_source).read_bytes()

    def _should_retry_http_error(exc: requests.HTTPError) -> bool:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        return status_code in _HF_OCR_RETRYABLE_STATUS_CODES

    def _post_submit() -> requests.Response:
        return requests.post(
            f"{base_url}/ocr/submit",
            files={"file": (upload_name, pdf_bytes, "application/pdf")},
            timeout=60,
        )

    def _get_status(job_id: str) -> requests.Response:
        return requests.get(f"{base_url}/ocr/status/{job_id}", timeout=10)

    resp = None
    last_error: Exception | None = None
    for attempt in range(1, _HF_OCR_MAX_ATTEMPTS + 1):
        try:
            resp = _post_submit()
            resp.raise_for_status()
            last_error = None
            break
        except requests.HTTPError as exc:
            last_error = exc
            if attempt >= _HF_OCR_MAX_ATTEMPTS or not _should_retry_http_error(exc):
                raise
            time.sleep(attempt)
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= _HF_OCR_MAX_ATTEMPTS:
                raise
            time.sleep(attempt)

    if resp is None:
        raise RuntimeError(f"HF OCR submit failed without a response: {last_error}")

    job_id = resp.json()["job_id"]

    deadline = time.monotonic() + settings.HF_OCR_POLL_TIMEOUT
    while time.monotonic() < deadline:
        try:
            status_resp = _get_status(job_id)
            status_resp.raise_for_status()
        except requests.HTTPError as exc:
            if not _should_retry_http_error(exc):
                raise
            time.sleep(settings.HF_OCR_POLL_INTERVAL)
            continue
        except requests.RequestException:
            time.sleep(settings.HF_OCR_POLL_INTERVAL)
            continue
        data = status_resp.json()
        if data["status"] == "done":
            try:
                requests.delete(f"{base_url}/ocr/job/{job_id}", timeout=10)
            except requests.RequestException:
                pass
            return data.get("text") or ""
        if data["status"] == "error":
            raise RuntimeError(f"HF OCR job failed: {data.get('error')}")
        time.sleep(settings.HF_OCR_POLL_INTERVAL)

    raise TimeoutError(
        f"HF OCR job {job_id} timed out after {settings.HF_OCR_POLL_TIMEOUT}s"
    )


def _render_pdf_pages_as_images(
    pdf_source: bytes | str, max_pages: int = _VISION_FALLBACK_MAX_PAGES
) -> List[bytes]:
    images: List[bytes] = []
    if isinstance(pdf_source, (bytes, bytearray)):
        doc = fitz.open(stream=bytes(pdf_source), filetype="pdf")
    else:
        doc = fitz.open(pdf_source)

    with doc:
        for page_index, page in enumerate(doc):
            if page_index >= max_pages:
                break
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            images.append(pixmap.tobytes("jpeg"))

    if not images:
        raise ValueError("No PDF pages available for vision fallback")
    return images


def _parse_resume_payload(
    pdf_source: bytes | str,
    display_name: str,
    llm_provider: LLMProvider,
) -> tuple[Dict[str, Any], Any, str]:
    cv_text = extract_text_from_pdf(pdf_source)
    if cv_text.strip():
        logger.info("Resume %s extracted via embedded PDF text", display_name)
        prompt = build_prompts.build_cv_parsing_prompt(cv_text)
        llm_response = llm_provider.generate(prompt)
        return _extract_json_object(llm_response.text), llm_response, "text"

    vision_error: Exception | None = None
    try:
        logger.info("Resume %s has no embedded text; trying vision fallback", display_name)
        images = _render_pdf_pages_as_images(pdf_source)
        vision_prompt = build_prompts.build_cv_vision_prompt()
        llm_response = llm_provider.generate_with_images(vision_prompt, images)
        return _extract_json_object(llm_response.text), llm_response, "vision"
    except Exception as exc:
        vision_error = exc
        logger.warning(
            "Vision fallback failed for %s: %s. Falling back to OCR.",
            display_name,
            exc,
        )

    logger.info("Resume %s falling back to OCR extraction", display_name)
    cv_text = extract_text_via_hf_ocr(pdf_source, display_name)
    if not cv_text.strip() and vision_error is not None:
        raise RuntimeError(
            f"Vision fallback failed ({vision_error}) and OCR returned empty text"
        ) from vision_error
    prompt = build_prompts.build_cv_parsing_prompt(cv_text)
    llm_response = llm_provider.generate(prompt)
    return _extract_json_object(llm_response.text), llm_response, "ocr"


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


def _pick_candidate_name(parsed_name: Any, submitted_full_name: Optional[str]) -> str:
    return (
        _normalize_text(parsed_name)
        or _normalize_text(submitted_full_name)
        or "Unknown Candidate"
    )


def _pick_candidate_email(
    parsed_email: Any, submitted_email: Optional[str]
) -> Optional[str]:
    return _normalize_text(parsed_email) or _normalize_text(submitted_email)


def _display_name_from_storage_uri(storage_uri: str) -> str:
    if storage_uri.startswith("s3://"):
        _, object_key = parse_storage_uri(storage_uri)
        return Path(object_key).name
    return Path(storage_uri).name


def _fallback_trace_payload(
    parsed: Optional[Dict[str, Any]],
    profile: CandidateProfile,
) -> Dict[str, Any]:
    parsed_name = _normalize_text(parsed.get("name")) if parsed else None
    parsed_email = _normalize_text(parsed.get("email")) if parsed else None
    return {
        "candidateName": profile.full_name,
        "candidateEmail": profile.email,
        "parsedName": parsed_name,
        "parsedEmail": parsed_email,
        "submittedFullName": profile.submitted_full_name,
        "submittedEmail": profile.submitted_email,
        "usedSubmittedFullName": parsed_name is None
        and profile.submitted_full_name is not None,
        "usedSubmittedEmail": parsed_email is None
        and profile.submitted_email is not None,
    }


def _build_failure_profile(
    resume_document_id: uuid.UUID,
    *,
    submitted_full_name: Optional[str] = None,
    submitted_email: Optional[str] = None,
) -> CandidateProfile:
    return CandidateProfile(
        resume_document_id=resume_document_id,
        full_name=_pick_candidate_name(None, submitted_full_name),
        submitted_full_name=_normalize_text(submitted_full_name),
        email=_pick_candidate_email(None, submitted_email),
        submitted_email=_normalize_text(submitted_email),
        profile_status=ProfileStatus.DRAFT.value,
    )


def _build_profile_from_parsed(
    resume_document_id: uuid.UUID,
    parsed: Dict[str, Any],
    *,
    submitted_full_name: Optional[str] = None,
    submitted_email: Optional[str] = None,
) -> CandidateProfile:
    normalized_submitted_full_name = _normalize_text(submitted_full_name)
    normalized_submitted_email = _normalize_text(submitted_email)
    structured_profile = _normalize_structured_profile(
        parsed.get("structured_profile"),
        parsed,
    )

    return CandidateProfile(
        resume_document_id=resume_document_id,
        full_name=_pick_candidate_name(
            parsed.get("name"), normalized_submitted_full_name
        ),
        submitted_full_name=normalized_submitted_full_name,
        phone=_normalize_text(parsed.get("phone")),
        email=_pick_candidate_email(parsed.get("email"), normalized_submitted_email),
        submitted_email=normalized_submitted_email,
        location_normalized=_normalize_text(parsed.get("location")),
        contact=_normalize_text(parsed.get("contact")),
        current_job_title=_normalize_text(parsed.get("current_job_title")),
        educated=_normalize_bool(parsed.get("educated")),
        ever_studied_abroad=_normalize_bool(parsed.get("ever_studied_abroad")),
        major=_normalize_text(parsed.get("major")),
        cpa=_normalize_text(parsed.get("cpa")),
        education_text=_normalize_text(parsed.get("education")),
        experience_text=_normalize_text(parsed.get("experience")),
        experience_years=_normalize_decimal(parsed.get("experience_years")),
        skills_text=_normalize_text(parsed.get("skills")),
        languages_text=_normalize_text(parsed.get("languages")),
        projects_text=_normalize_text(parsed.get("projects")),
        summary_text=_normalize_text(parsed.get("summary")),
        achievements_text=_normalize_text(parsed.get("achievements")),
        publications_text=_normalize_text(parsed.get("publications")),
        certifications_text=_normalize_text(parsed.get("certifications")),
        references_text=_normalize_text(parsed.get("references")),
        other_text=_normalize_text(parsed.get("other")),
        structured_profile=structured_profile,
        profile_status=ProfileStatus.REVIEWED.value,
    )


def _latest_extraction_mode_from_traces(
    traces: Sequence[ExtractionTrace],
) -> Optional[str]:
    for trace in reversed(list(traces)):
        payload = trace.payload if isinstance(trace.payload, dict) else {}
        mode = _normalize_text(payload.get("extractionMode")) if payload else None
        if mode:
            return mode
    return None


def _get_resume_extraction_mode(
    db: Session,
    resume_document_id: uuid.UUID,
) -> Optional[str]:
    traces = (
        db.query(ExtractionTrace)
        .filter(ExtractionTrace.resume_document_id == resume_document_id)
        .order_by(ExtractionTrace.created_at.asc())
        .all()
    )
    return _latest_extraction_mode_from_traces(traces)


# ---------------------------------------------------------------------------
# Async-friendly helpers (Celery path)
# ---------------------------------------------------------------------------


def create_resume_document(
    *,
    db: Session,
    storage_uri: str,
    original_file_name: str,
    job_id: uuid.UUID,
    uploaded_by_user_id: uuid.UUID,
    retention_days: int = 365,
) -> ResumeDocument:
    """Create a ResumeDocument row with status=uploaded (no parsing)."""
    resume = ResumeDocument(
        original_file_name=original_file_name,
        storage_uri=storage_uri,
        upload_status=UploadStatus.UPLOADED.value,
        job_id=job_id,
        uploaded_by_user_id=uploaded_by_user_id,
        retention_expires_at=datetime.now(timezone.utc)
        + timedelta(days=retention_days),
    )
    db.add(resume)
    db.commit()
    db.refresh(resume)
    return resume


def process_single_resume(
    resume_document_id: uuid.UUID,
    *,
    submitted_full_name: Optional[str] = None,
    submitted_email: Optional[str] = None,
) -> Dict[str, Any]:
    """Run heavy parsing for one ResumeDocument **in its own DB session**.

    Designed to be called from a Celery worker.  Transitions
    ``upload_status`` through processing → processed / failed and
    creates ``CandidateProfile`` + ``ExtractionTrace`` rows.
    """
    from src.models.session import SessionLocal

    db: Session = SessionLocal()
    try:
        resume = db.get(ResumeDocument, resume_document_id)
        if resume is None:
            return {
                "resume_document_id": str(resume_document_id),
                "status": "failed",
                "error": "ResumeDocument not found",
            }

        display_name = resume.original_file_name or "unknown.pdf"
        storage_uri = resume.storage_uri

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

            object_storage = get_object_storage()
            if storage_uri.startswith("s3://"):
                pdf_source: bytes | str = object_storage.download_bytes(storage_uri)
            else:
                pdf_source = storage_uri

            llm_provider = _resume_llm_provider()
            parsed_payload, llm_response, extraction_mode = _parse_resume_payload(
                pdf_source,
                display_name,
                llm_provider,
            )

            profile = _build_profile_from_parsed(
                resume.id,
                parsed_payload,
                submitted_full_name=submitted_full_name,
                submitted_email=submitted_email,
            )
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
                        **_fallback_trace_payload(parsed_payload, profile),
                        "extractionMode": extraction_mode,
                        "llmProvider": llm_response.provider,
                        "llmModel": llm_response.model,
                    },
                )
            )
            db.commit()
            db.refresh(profile)

            return {
                "file_name": display_name,
                "resume_document_id": str(resume.id),
                "candidate_profile_id": str(profile.id),
                "status": "processed",
                "extraction_mode": extraction_mode,
            }

        except Exception as exc:
            db.rollback()
            resume_db = db.get(ResumeDocument, resume_document_id)
            failure_profile_id = None
            if resume_db is not None:
                resume_db.upload_status = UploadStatus.FAILED.value
                minimal_profile = None
                if submitted_full_name is not None or submitted_email is not None:
                    minimal_profile = _build_failure_profile(
                        resume_db.id,
                        submitted_full_name=submitted_full_name,
                        submitted_email=submitted_email,
                    )
                    db.add(minimal_profile)
                db.add(
                    ExtractionTrace(
                        resume_document_id=resume_db.id,
                        stage="cv_parsing",
                        status="failed",
                        message=str(exc),
                        payload={
                            **(
                                _fallback_trace_payload(None, minimal_profile)
                                if minimal_profile is not None
                                else {
                                    "submittedFullName": _normalize_text(
                                        submitted_full_name
                                    ),
                                    "submittedEmail": _normalize_text(submitted_email),
                                    "usedSubmittedFullName": False,
                                    "usedSubmittedEmail": False,
                                }
                            ),
                            "createdFallbackProfile": minimal_profile is not None,
                        },
                    )
                )
                db.commit()
                if minimal_profile is not None:
                    db.refresh(minimal_profile)
                    failure_profile_id = str(minimal_profile.id)

            return {
                "file_name": display_name,
                "resume_document_id": str(resume_document_id),
                "candidate_profile_id": failure_profile_id,
                "status": "failed",
                "error": str(exc),
            }
    finally:
        db.close()


def parse_pdf_to_sections(
    storage_uris: Optional[Sequence[str]] = None,
    *,
    db: Session,
    job_id: uuid.UUID,
    uploaded_by_user_id: Optional[uuid.UUID] = None,
    retention_days: int = 365,
    original_filenames: Optional[List[str]] = None,
    submitted_full_names: Optional[List[str]] = None,
    submitted_emails: Optional[List[str]] = None,
    filepaths: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Extract CV text from PDFs, parse with LLM prompt, and persist results."""
    source_locations = list(storage_uris or filepaths or [])
    if not source_locations:
        return []

    actor_id = uploaded_by_user_id or uuid.UUID("00000000-0000-0000-0000-000000000000")
    llm_provider = _resume_llm_provider()
    object_storage = get_object_storage()
    results: List[Dict[str, Any]] = []

    for idx, storage_uri in enumerate(source_locations):
        display_name = (
            original_filenames[idx]
            if original_filenames and idx < len(original_filenames)
            else _display_name_from_storage_uri(storage_uri)
        )
        submitted_full_name = (
            submitted_full_names[idx]
            if submitted_full_names and idx < len(submitted_full_names)
            else None
        )
        submitted_email = (
            submitted_emails[idx]
            if submitted_emails and idx < len(submitted_emails)
            else None
        )
        resume = ResumeDocument(
            original_file_name=display_name,
            storage_uri=storage_uri,
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

            if storage_uri.startswith("s3://"):
                pdf_source: bytes | str = object_storage.download_bytes(storage_uri)
            else:
                pdf_source = storage_uri

            parsed_payload, llm_response, extraction_mode = _parse_resume_payload(
                pdf_source,
                display_name,
                llm_provider,
            )

            profile = _build_profile_from_parsed(
                resume.id,
                parsed_payload,
                submitted_full_name=submitted_full_name,
                submitted_email=submitted_email,
            )
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
                        **_fallback_trace_payload(parsed_payload, profile),
                        "extractionMode": extraction_mode,
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
                    "extraction_mode": extraction_mode,
                }
            )
        except Exception as exc:
            db.rollback()
            resume_db = db.get(ResumeDocument, resume.id)
            failure_profile_id = None
            if resume_db is not None:
                resume_db.upload_status = UploadStatus.FAILED.value
                minimal_profile = None
                if submitted_full_name is not None or submitted_email is not None:
                    minimal_profile = _build_failure_profile(
                        resume_db.id,
                        submitted_full_name=submitted_full_name,
                        submitted_email=submitted_email,
                    )
                    db.add(minimal_profile)
                db.add(
                    ExtractionTrace(
                        resume_document_id=resume_db.id,
                        stage="cv_parsing",
                        status="failed",
                        message=str(exc),
                        payload={
                            **(
                                _fallback_trace_payload(None, minimal_profile)
                                if minimal_profile is not None
                                else {
                                    "submittedFullName": _normalize_text(
                                        submitted_full_name
                                    ),
                                    "submittedEmail": _normalize_text(submitted_email),
                                    "usedSubmittedFullName": False,
                                    "usedSubmittedEmail": False,
                                }
                            ),
                            "createdFallbackProfile": minimal_profile is not None,
                        },
                    )
                )
                db.commit()
                if minimal_profile is not None:
                    db.refresh(minimal_profile)
                    failure_profile_id = str(minimal_profile.id)

            results.append(
                {
                    "file_name": display_name,
                    "resume_document_id": str(resume.id),
                    "candidate_profile_id": failure_profile_id,
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


def _resume_to_dict(
    resume: ResumeDocument,
    *,
    extraction_mode: Optional[str] = None,
) -> Dict[str, Any]:
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
        "extraction_mode": extraction_mode,
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
    return _resume_to_dict(
        resume,
        extraction_mode=_get_resume_extraction_mode(db, resume.id),
    )


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
    return [
        _resume_to_dict(
            r,
            extraction_mode=_get_resume_extraction_mode(db, r.id),
        )
        for r in resumes
    ]


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
    return _resume_to_dict(
        resume,
        extraction_mode=_get_resume_extraction_mode(db, resume.id),
    )


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
        delete_file: When True, also removes the stored PDF object or legacy local file.

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
            if storage_uri.startswith("s3://"):
                parse_storage_uri(storage_uri)
                get_object_storage().delete_object(storage_uri)
            else:
                Path(storage_uri).unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not delete file {storage_uri}: {exc}")

    return True
