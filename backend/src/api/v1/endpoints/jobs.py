from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from src.models.candidate_profile import CandidateProfile
from src.models.deps import get_current_user, get_db
from src.models.job import Job
from src.models.enums import MatchRunStatus, UploadStatus
from src.models.job_matching import JobDescription, MatchRun
from src.models.query_shortlist import QuerySession, QueryTurn
from src.models.resume_document import ResumeDocument
from src.models.user_account import UserAccount
from src.services.ai_agent.graph import get_graph
from src.services.job_description_service import _jd_to_dict
from src.services.job_scope import (
    apply_public_job_settings,
    get_current_user_owned_job,
    get_job_scoped_candidate,
    get_job_scoped_jd,
    get_job_scoped_resume,
    require_job_scoped_jd,
    serialize_job,
    serialize_job_application_settings,
)
from src.services.object_storage import build_object_key, get_object_storage
from src.services.public_job_service import generate_public_apply_token
from src.services.ai_agent.langgraph_trace import format_exception_payload, get_trace_logger
from src.services.resume_service import (
    _resume_to_dict,
    create_resume_document,
    parse_pdf_to_sections,
)
from src.services.score_candidate import score_candidates
from src.services.scoring_errors import ScoringProviderLimitError
from worker.tasks import process_resume

router = APIRouter()
TOTAL_CANDIDATE_COUNT_PATTERNS = (
    r"\bhow many candidates\b",
    r"\bhow many applicants\b",
    r"\bnumber of candidates\b",
    r"\bnumber of applicants\b",
    r"\bcandidate count\b",
    r"\bapplicant count\b",
    r"bao nhiêu ứng viên",
    r"số lượng ứng viên",
    r"có mấy ứng viên",
)


class JobCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    status: str = Field(default="active", min_length=1, max_length=50)
    candidate_message: Optional[str] = None
    public_apply_enabled: bool = True


class JobUpdateRequest(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=255)
    status: Optional[str] = Field(default=None, min_length=1, max_length=50)
    candidate_message: Optional[str] = None
    public_apply_enabled: Optional[bool] = None


class JobResponse(BaseModel):
    id: str
    owner_user_id: str
    title: str
    status: str
    candidate_message: Optional[str]
    public_apply_enabled: bool
    public_apply_url: str
    created_at: str
    updated_at: str
    archived_at: Optional[str]


class JobApplicationLinkResponse(BaseModel):
    public_apply_enabled: bool
    public_apply_url: str
    candidate_message: Optional[str]


class JobListResponse(BaseModel):
    items: list[JobResponse]
    total: int


class JobDescriptionRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=255)
    jd_text: str = Field(..., min_length=1)
    hidden_text: str = ""
    is_active: bool = True


class JobDescriptionPatchRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=255)
    jd_text: Optional[str] = None
    hidden_text: Optional[str] = None
    is_active: Optional[bool] = None


class ResumeResponse(BaseModel):
    id: str
    job_id: str
    original_file_name: str
    candidate_profile_id: Optional[str] = None
    candidate_display_name: Optional[str] = None
    storage_uri: str
    upload_status: str
    extraction_mode: Optional[str] = None
    duplicate_group_key: Optional[str]
    uploaded_by_user_id: str
    uploader_display_name: Optional[str] = None
    uploaded_at: Optional[str]
    processed_at: Optional[str]
    retention_expires_at: Optional[str]


class ResumeListResponse(BaseModel):
    items: list[ResumeResponse]
    total: int


class ResumeUpdateRequest(BaseModel):
    original_file_name: Optional[str] = None
    upload_status: Optional[str] = None


class CandidateResponse(BaseModel):
    id: str
    resume_document_id: str
    full_name: str
    email: Optional[str]
    current_job_title: Optional[str]
    summary_text: Optional[str]
    skills_text: Optional[str]
    experience_text: Optional[str]
    experience_years: Optional[float]
    education_text: Optional[str]


class CandidateListResponse(BaseModel):
    items: list[CandidateResponse]
    total: int


class ScoreRequest(BaseModel):
    score_threshold: float = Field(50.0, ge=0, le=100)
    candidate_profile_ids: Optional[list[uuid.UUID]] = None
    section_weights: Optional[dict[str, float]] = None
    batch_size: int = Field(10, ge=1, le=50)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    candidate_limit: int = Field(500, ge=1, le=2000)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    candidates_in_scope: int
    session: Optional["ChatSessionResponse"] = None
    turn: Optional["ChatTurnResponse"] = None


class ChatSessionCreateRequest(BaseModel):
    session_title: Optional[str] = Field(default=None, max_length=255)


class ChatSessionUpdateRequest(BaseModel):
    session_title: Optional[str] = Field(default=None, max_length=255)


class ChatSessionResponse(BaseModel):
    id: str
    user_id: str
    job_id: str
    session_title: Optional[str]
    turn_count: int
    created_at: datetime
    updated_at: datetime


class ChatSessionListResponse(BaseModel):
    items: list[ChatSessionResponse]
    total: int


class ChatTurnResponse(BaseModel):
    id: str
    query_session_id: str
    user_question: str
    answer_text: str
    matched_candidate_ids: Optional[list[str]]
    matched_count: Optional[int]
    tool_trace_masked: Optional[dict[str, Any]]
    created_at: datetime


class JobSetupStatusResponse(BaseModel):
    job_id: str
    resume_count: int
    processed_candidate_count: int
    has_uploaded_resumes: bool
    has_processed_candidates: bool
    has_active_job_description: bool
    has_completed_score_run: bool
    has_chat_turn: bool
    completed_score_run_count: int
    chat_session_count: int
    chat_turn_count: int
    latest_job_description_id: Optional[str] = None
    latest_score_run_id: Optional[str] = None
    latest_score_run_at: Optional[datetime] = None
    latest_chat_session_id: Optional[str] = None
    latest_chat_turn_at: Optional[datetime] = None


ChatResponse.model_rebuild()


def _serialize_candidate(profile: CandidateProfile) -> CandidateResponse:
    return CandidateResponse(
        id=str(profile.id),
        resume_document_id=str(profile.resume_document_id),
        full_name=profile.full_name,
        email=profile.email,
        current_job_title=profile.current_job_title,
        summary_text=profile.summary_text,
        skills_text=profile.skills_text,
        experience_text=profile.experience_text,
        experience_years=(
            float(profile.experience_years)
            if profile.experience_years is not None
            else None
        ),
        education_text=profile.education_text,
    )


def _resume_response_payload(
    resume: ResumeDocument,
    *,
    candidate_profile_id: Optional[uuid.UUID] = None,
    candidate_display_name: Optional[str] = None,
    uploader_display_name: Optional[str] = None,
) -> dict[str, Any]:
    payload = _resume_to_dict(resume)
    payload["candidate_profile_id"] = (
        str(candidate_profile_id) if candidate_profile_id is not None else None
    )
    payload["candidate_display_name"] = candidate_display_name
    payload["uploader_display_name"] = uploader_display_name
    return payload


def _get_resume_display_metadata(
    db: Session,
    resume_id: uuid.UUID,
) -> tuple[Optional[uuid.UUID], Optional[str], Optional[str]]:
    row = db.execute(
        select(CandidateProfile.id, CandidateProfile.full_name, UserAccount.display_name)
        .select_from(ResumeDocument)
        .outerjoin(
            CandidateProfile,
            CandidateProfile.resume_document_id == ResumeDocument.id,
        )
        .outerjoin(UserAccount, UserAccount.id == ResumeDocument.uploaded_by_user_id)
        .where(ResumeDocument.id == resume_id)
    ).first()
    if row is None:
        return None, None, None
    return row[0], row[1], row[2]


def _load_job_candidates(
    db: Session, job_id: uuid.UUID, limit: int
) -> list[dict[str, Any]]:
    rows = (
        db.execute(
            select(CandidateProfile)
            .join(
                ResumeDocument, ResumeDocument.id == CandidateProfile.resume_document_id
            )
            .where(ResumeDocument.job_id == job_id)
            .limit(limit)
        )
        .scalars()
        .all()
    )
    return [
        {
            "id": str(r.id),
            "full_name": r.full_name,
            "phone": r.phone,
            "email": r.email,
            "location_normalized": r.location_normalized,
            "contact": r.contact,
            "current_job_title": r.current_job_title,
            "graduation_status": r.graduation_status,
            "ever_studied_abroad": r.ever_studied_abroad,
            "major": r.major,
            "cpa": r.cpa,
            "education_text": r.education_text,
            "experience_text": r.experience_text,
            "experience_years": (
                float(r.experience_years) if r.experience_years is not None else None
            ),
            "skills_text": r.skills_text,
            "languages_text": r.languages_text,
            "projects_text": r.projects_text,
            "summary_text": r.summary_text,
            "achievements_text": r.achievements_text,
            "publications_text": r.publications_text,
            "certifications_text": r.certifications_text,
            "references_text": r.references_text,
            "other_text": r.other_text,
        }
        for r in rows
    ]


def _build_job_chat_context(
    *,
    job: Job,
    job_description: JobDescription | None,
) -> dict[str, Any]:
    return {
        "job_id": str(job.id),
        "job_title": (job.title or "").strip() or None,
        "job_description_id": str(job_description.id) if job_description is not None else None,
        "job_description_title": (job_description.title or "").strip()
        if job_description is not None and job_description.title
        else None,
        "job_description_text": (job_description.jd_text or "").strip()
        if job_description is not None and job_description.jd_text
        else None,
        "job_hidden_text": (job_description.hidden_text or "").strip()
        if job_description is not None and job_description.hidden_text
        else None,
    }


def _is_total_candidate_count_question(question: str) -> bool:
    normalized = question.strip().lower()
    if not normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in TOTAL_CANDIDATE_COUNT_PATTERNS)


def _question_language(question: str) -> str:
    normalized = (question or "").strip().lower()
    if not normalized:
        return "vi"

    if re.search(r"[àáạảãăằắặẳẵâầấậẩẫđèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ]", normalized):
        return "vi"

    vietnamese_markers = (
        "ứng viên",
        "bao nhiêu",
        "có mấy",
        "công việc này",
        "kinh nghiệm",
        "kỹ năng",
        "đại học",
    )
    if any(marker in normalized for marker in vietnamese_markers):
        return "vi"

    return "en"


def _total_candidate_count_answer(question: str, count: int) -> str:
    if _question_language(question) == "en":
        return (
            f"There is {count} candidate in this job."
            if count == 1
            else f"There are {count} candidates in this job."
        )
    return (
        f"Có {count} ứng viên trong job này."
        if count != 1
        else f"Có {count} ứng viên trong job này."
    )


def _default_chat_title(message: str) -> str:
    title = " ".join(message.strip().split())
    if not title:
        return "New Conversation"
    return title[:48] + ("..." if len(title) > 48 else "")


def _messages_from_turns(turns: list[QueryTurn]) -> list[Any]:
    messages: list[Any] = []
    for turn in turns:
        messages.append(HumanMessage(content=turn.user_question))
        messages.append(AIMessage(content=turn.answer_text))
    return messages


def _serialize_chat_session(db: Session, session: QuerySession) -> ChatSessionResponse:
    turn_count = (
        db.query(func.count(QueryTurn.id))
        .filter(QueryTurn.query_session_id == session.id)
        .scalar()
        or 0
    )
    return ChatSessionResponse(
        id=str(session.id),
        user_id=str(session.user_id),
        job_id=str(session.job_id),
        session_title=session.session_title,
        turn_count=int(turn_count),
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def _serialize_chat_turn(turn: QueryTurn) -> ChatTurnResponse:
    return ChatTurnResponse(
        id=str(turn.id),
        query_session_id=str(turn.query_session_id),
        user_question=turn.user_question,
        answer_text=turn.answer_text,
        matched_candidate_ids=turn.matched_candidate_ids,
        matched_count=turn.matched_count,
        tool_trace_masked=turn.tool_trace_masked,
        created_at=turn.created_at,
    )


def _get_job_chat_session_or_404(
    db: Session,
    *,
    current_user_id: uuid.UUID,
    job_id: uuid.UUID,
    session_id: uuid.UUID,
) -> QuerySession:
    session = (
        db.query(QuerySession)
        .filter(
            QuerySession.id == session_id,
            QuerySession.user_id == current_user_id,
            QuerySession.job_id == job_id,
        )
        .one_or_none()
    )
    if session is None:
        raise HTTPException(status_code=404, detail=f"Chat session '{session_id}' not found")
    return session


def _resolve_chat_session(
    db: Session,
    *,
    current_user_id: uuid.UUID,
    job_id: uuid.UUID,
    body: ChatRequest,
) -> QuerySession:
    if body.session_id:
        try:
            session_id = uuid.UUID(body.session_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="session_id must be a UUID") from exc
        return _get_job_chat_session_or_404(
            db,
            current_user_id=current_user_id,
            job_id=job_id,
            session_id=session_id,
        )

    session = QuerySession(
        user_id=current_user_id,
        job_id=job_id,
        session_title=_default_chat_title(body.message),
    )
    db.add(session)
    db.flush()
    return session


def _persist_chat_turn(
    db: Session,
    *,
    session: QuerySession,
    question: str,
    answer: str,
    candidates_in_scope: int,
    candidate_limit: int,
    matched_candidate_ids: list[str] | None,
    route: str,
) -> QueryTurn:
    turn = QueryTurn(
        query_session_id=session.id,
        user_question=question,
        answer_text=answer,
        matched_candidate_ids=matched_candidate_ids,
        matched_count=candidates_in_scope,
        tool_trace_masked={
            "route": route,
            "candidate_limit": candidate_limit,
            "candidates_in_scope": candidates_in_scope,
        },
    )
    session.updated_at = datetime.now(timezone.utc)
    db.add(turn)
    db.commit()
    db.refresh(session)
    db.refresh(turn)
    return turn


@router.post("/", response_model=JobResponse, status_code=201)
def create_job(
    body: JobCreateRequest,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    job = Job(
        owner_user_id=current_user.id,
        title=body.title.strip(),
        status=body.status.strip(),
        public_apply_enabled=body.public_apply_enabled,
    )
    apply_public_job_settings(
        job,
        candidate_message=body.candidate_message,
        public_apply_enabled=body.public_apply_enabled,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return JobResponse(**serialize_job(job))


@router.get("/", response_model=JobListResponse)
def list_jobs(
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    jobs = (
        db.execute(
            select(Job)
            .where(Job.owner_user_id == current_user.id)
            .order_by(Job.updated_at.desc())
        )
        .scalars()
        .all()
    )
    return JobListResponse(
        items=[JobResponse(**serialize_job(job)) for job in jobs], total=len(jobs)
    )


@router.get("/{job_id}", response_model=JobResponse)
def get_job(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    return JobResponse(
        **serialize_job(get_current_user_owned_job(db, current_user.id, job_id))
    )


@router.patch("/{job_id}", response_model=JobResponse)
def update_job(
    job_id: uuid.UUID,
    body: JobUpdateRequest,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    job = get_current_user_owned_job(db, current_user.id, job_id)
    if body.title is not None:
        job.title = body.title.strip()
    if body.status is not None:
        job.status = body.status.strip()
    if "candidate_message" in body.model_fields_set:
        apply_public_job_settings(job, candidate_message=body.candidate_message)
    if "public_apply_enabled" in body.model_fields_set:
        apply_public_job_settings(job, public_apply_enabled=body.public_apply_enabled)
    db.commit()
    db.refresh(job)
    return JobResponse(**serialize_job(job))


@router.delete("/{job_id}")
def delete_job(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    job = get_current_user_owned_job(db, current_user.id, job_id)
    db.delete(job)
    db.commit()
    return {"deleted": True, "job_id": str(job_id)}


@router.get("/{job_id}/application-link", response_model=JobApplicationLinkResponse)
def get_job_application_link(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    job = get_current_user_owned_job(db, current_user.id, job_id)
    return JobApplicationLinkResponse(**serialize_job_application_settings(job))


@router.post(
    "/{job_id}/application-link/rotate", response_model=JobApplicationLinkResponse
)
def rotate_job_application_link(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    job = get_current_user_owned_job(db, current_user.id, job_id)
    job.public_apply_token = generate_public_apply_token()
    job.public_apply_created_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(job)
    return JobApplicationLinkResponse(**serialize_job_application_settings(job))


@router.get("/{job_id}/job-description")
def get_job_description(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    get_current_user_owned_job(db, current_user.id, job_id)
    jd = require_job_scoped_jd(db, current_user.id, job_id)
    return _jd_to_dict(jd)


@router.post("/{job_id}/job-description", status_code=201)
def create_or_replace_job_description(
    job_id: uuid.UUID,
    body: JobDescriptionRequest,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    get_current_user_owned_job(db, current_user.id, job_id)
    existing = get_job_scoped_jd(db, current_user.id, job_id)
    if existing is None:
        existing = JobDescription(
            job_id=job_id,
            title=body.title,
            jd_text=body.jd_text.strip(),
            hidden_text=body.hidden_text.strip(),
            created_by_user_id=current_user.id,
            is_active=body.is_active,
        )
        db.add(existing)
    else:
        existing.title = body.title
        existing.jd_text = body.jd_text.strip()
        existing.hidden_text = body.hidden_text.strip()
        existing.is_active = body.is_active
    db.commit()
    db.refresh(existing)
    return _jd_to_dict(existing)


@router.patch("/{job_id}/job-description")
def patch_job_description(
    job_id: uuid.UUID,
    body: JobDescriptionPatchRequest,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    jd = require_job_scoped_jd(db, current_user.id, job_id)
    if body.title is not None:
        jd.title = body.title
    if body.jd_text is not None:
        if not body.jd_text.strip():
            raise HTTPException(status_code=422, detail="jd_text must not be empty")
        jd.jd_text = body.jd_text.strip()
    if body.hidden_text is not None:
        jd.hidden_text = body.hidden_text.strip()
    if body.is_active is not None:
        jd.is_active = body.is_active
    db.commit()
    db.refresh(jd)
    return _jd_to_dict(jd)


@router.get("/{job_id}/resumes", response_model=ResumeListResponse)
def list_job_resumes(
    job_id: uuid.UUID,
    upload_status: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    get_current_user_owned_job(db, current_user.id, job_id)
    query = select(
        ResumeDocument,
        CandidateProfile.id,
        CandidateProfile.full_name,
        UserAccount.display_name,
    ).outerjoin(
        CandidateProfile,
        CandidateProfile.resume_document_id == ResumeDocument.id,
    ).outerjoin(
        UserAccount,
        UserAccount.id == ResumeDocument.uploaded_by_user_id,
    ).where(
        ResumeDocument.job_id == job_id,
    )
    if upload_status is not None:
        query = query.where(ResumeDocument.upload_status == upload_status)
    rows = (
        db.execute(
            query.order_by(ResumeDocument.uploaded_at.desc())
            .offset(offset)
            .limit(limit)
        )
        .all()
    )
    return ResumeListResponse(
        items=[
            ResumeResponse(
                **_resume_response_payload(
                    resume,
                    candidate_profile_id=candidate_profile_id,
                    candidate_display_name=candidate_display_name,
                    uploader_display_name=uploader_display_name,
                )
            )
            for resume, candidate_profile_id, candidate_display_name, uploader_display_name in rows
        ],
        total=len(rows),
    )


@router.post("/{job_id}/resumes", status_code=202)
async def upload_job_resumes(
    job_id: uuid.UUID,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    get_current_user_owned_job(db, current_user.id, job_id)
    object_storage = get_object_storage()
    items: list[dict[str, Any]] = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400, detail=f"Invalid file type: {file.filename}"
            )
        original_name = Path(file.filename).name
        storage_uri = object_storage.upload_bytes(
            data=await file.read(),
            object_key=build_object_key(
                prefix=f"resumes/{job_id}",
                original_filename=original_name,
            ),
            content_type=file.content_type or "application/pdf",
        )
        resume = create_resume_document(
            db=db,
            storage_uri=storage_uri,
            original_file_name=original_name,
            job_id=job_id,
            uploaded_by_user_id=current_user.id,
        )
        task = process_resume.delay(str(resume.id))
        items.append(
            {
                "file_name": original_name,
                "resume_document_id": str(resume.id),
                "status": "queued",
                "task_id": task.id,
            }
        )
    return {
        "total_files": len(items),
        "queued_files": len(items),
        "items": items,
    }


@router.get("/{job_id}/resumes/{resume_id}", response_model=ResumeResponse)
def get_job_resume(
    job_id: uuid.UUID,
    resume_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    resume = get_job_scoped_resume(db, current_user.id, job_id, resume_id)
    candidate_profile_id, candidate_display_name, uploader_display_name = _get_resume_display_metadata(
        db, resume.id
    )
    return ResumeResponse(
        **_resume_response_payload(
            resume,
            candidate_profile_id=candidate_profile_id,
            candidate_display_name=candidate_display_name,
            uploader_display_name=uploader_display_name,
        )
    )


@router.patch("/{job_id}/resumes/{resume_id}", response_model=ResumeResponse)
def update_job_resume(
    job_id: uuid.UUID,
    resume_id: uuid.UUID,
    body: ResumeUpdateRequest,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    resume = get_job_scoped_resume(db, current_user.id, job_id, resume_id)
    if body.original_file_name is not None:
        resume.original_file_name = body.original_file_name.strip()
    if body.upload_status is not None:
        resume.upload_status = body.upload_status
    db.commit()
    db.refresh(resume)
    candidate_profile_id, candidate_display_name, uploader_display_name = _get_resume_display_metadata(
        db, resume.id
    )
    return ResumeResponse(
        **_resume_response_payload(
            resume,
            candidate_profile_id=candidate_profile_id,
            candidate_display_name=candidate_display_name,
            uploader_display_name=uploader_display_name,
        )
    )


@router.delete("/{job_id}/resumes/{resume_id}")
def delete_job_resume(
    job_id: uuid.UUID,
    resume_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    get_current_user_owned_job(db, current_user.id, job_id)
    resume = (
        db.execute(
            select(ResumeDocument).where(
                ResumeDocument.id == resume_id,
                ResumeDocument.job_id == job_id,
            )
        )
        .scalars()
        .first()
    )
    if resume is not None:
        db.delete(resume)
        db.commit()
    return {"deleted": True, "resume_id": str(resume_id)}


@router.get("/{job_id}/candidates", response_model=CandidateListResponse)
def list_job_candidates(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    get_current_user_owned_job(db, current_user.id, job_id)
    rows = (
        db.execute(
            select(CandidateProfile)
            .join(
                ResumeDocument, ResumeDocument.id == CandidateProfile.resume_document_id
            )
            .where(ResumeDocument.job_id == job_id)
            .order_by(CandidateProfile.created_at.desc())
        )
        .scalars()
        .all()
    )
    return CandidateListResponse(
        items=[_serialize_candidate(row) for row in rows], total=len(rows)
    )


@router.post("/{job_id}/score")
def score_job_candidates(
    job_id: uuid.UUID,
    body: ScoreRequest,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    jd = require_job_scoped_jd(db, current_user.id, job_id)
    if body.candidate_profile_ids:
        for candidate_id in body.candidate_profile_ids:
            get_job_scoped_candidate(db, current_user.id, job_id, candidate_id)
    try:
        return score_candidates(
            db=db,
            job_description_id=jd.id,
            initiated_by_user_id=current_user.id,
            score_threshold=Decimal(str(body.score_threshold)),
            candidate_profile_ids=body.candidate_profile_ids,
            section_weights=body.section_weights,
            batch_size=body.batch_size,
        )
    except ScoringProviderLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if "not found" in detail.lower() else 422
        raise HTTPException(status_code=status_code, detail=detail) from exc


@router.get("/{job_id}/setup-status", response_model=JobSetupStatusResponse)
def get_job_setup_status(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    get_current_user_owned_job(db, current_user.id, job_id)

    resume_count = (
        db.query(func.count(ResumeDocument.id))
        .filter(ResumeDocument.job_id == job_id)
        .scalar()
        or 0
    )
    processed_candidate_count = (
        db.query(func.count(CandidateProfile.id))
        .join(ResumeDocument, ResumeDocument.id == CandidateProfile.resume_document_id)
        .filter(
            ResumeDocument.job_id == job_id,
            ResumeDocument.upload_status == UploadStatus.PROCESSED,
        )
        .scalar()
        or 0
    )
    latest_jd = (
        db.query(JobDescription)
        .filter(JobDescription.job_id == job_id, JobDescription.is_active.is_(True))
        .order_by(JobDescription.created_at.desc())
        .first()
    )
    completed_score_runs = (
        db.query(MatchRun)
        .join(JobDescription, JobDescription.id == MatchRun.job_description_id)
        .filter(
            JobDescription.job_id == job_id,
            MatchRun.run_status == MatchRunStatus.COMPLETED,
        )
        .order_by(MatchRun.completed_at.desc().nullslast(), MatchRun.created_at.desc())
        .all()
    )
    chat_sessions_query = db.query(QuerySession).filter(
        QuerySession.job_id == job_id,
        QuerySession.user_id == current_user.id,
    )
    chat_session_count = chat_sessions_query.count()
    latest_chat_session = chat_sessions_query.order_by(QuerySession.updated_at.desc()).first()
    chat_turn_count = (
        db.query(func.count(QueryTurn.id))
        .join(QuerySession, QuerySession.id == QueryTurn.query_session_id)
        .filter(QuerySession.job_id == job_id, QuerySession.user_id == current_user.id)
        .scalar()
        or 0
    )
    latest_chat_turn = (
        db.query(QueryTurn)
        .join(QuerySession, QuerySession.id == QueryTurn.query_session_id)
        .filter(QuerySession.job_id == job_id, QuerySession.user_id == current_user.id)
        .order_by(QueryTurn.created_at.desc())
        .first()
    )
    latest_score_run = completed_score_runs[0] if completed_score_runs else None

    return JobSetupStatusResponse(
        job_id=str(job_id),
        resume_count=int(resume_count),
        processed_candidate_count=int(processed_candidate_count),
        has_uploaded_resumes=resume_count > 0,
        has_processed_candidates=processed_candidate_count > 0,
        has_active_job_description=latest_jd is not None,
        has_completed_score_run=latest_score_run is not None,
        has_chat_turn=chat_turn_count > 0,
        completed_score_run_count=len(completed_score_runs),
        chat_session_count=chat_session_count,
        chat_turn_count=int(chat_turn_count),
        latest_job_description_id=str(latest_jd.id) if latest_jd else None,
        latest_score_run_id=str(latest_score_run.id) if latest_score_run else None,
        latest_score_run_at=(
            latest_score_run.completed_at or latest_score_run.created_at
            if latest_score_run
            else None
        ),
        latest_chat_session_id=str(latest_chat_session.id) if latest_chat_session else None,
        latest_chat_turn_at=latest_chat_turn.created_at if latest_chat_turn else None,
    )


@router.post(
    "/{job_id}/chat/sessions",
    response_model=ChatSessionResponse,
    status_code=201,
)
def create_job_chat_session(
    job_id: uuid.UUID,
    body: ChatSessionCreateRequest,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    get_current_user_owned_job(db, current_user.id, job_id)
    session = QuerySession(
        user_id=current_user.id,
        job_id=job_id,
        session_title=body.session_title,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return _serialize_chat_session(db, session)


@router.get("/{job_id}/chat/sessions", response_model=ChatSessionListResponse)
def list_job_chat_sessions(
    job_id: uuid.UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    get_current_user_owned_job(db, current_user.id, job_id)
    query = db.query(QuerySession).filter(
        QuerySession.user_id == current_user.id,
        QuerySession.job_id == job_id,
    )
    total = query.count()
    rows = (
        query.order_by(QuerySession.updated_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return ChatSessionListResponse(
        items=[_serialize_chat_session(db, session) for session in rows],
        total=total,
    )


@router.get("/{job_id}/chat/sessions/{session_id}", response_model=ChatSessionResponse)
def get_job_chat_session(
    job_id: uuid.UUID,
    session_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    get_current_user_owned_job(db, current_user.id, job_id)
    session = _get_job_chat_session_or_404(
        db,
        current_user_id=current_user.id,
        job_id=job_id,
        session_id=session_id,
    )
    return _serialize_chat_session(db, session)


@router.patch("/{job_id}/chat/sessions/{session_id}", response_model=ChatSessionResponse)
def update_job_chat_session(
    job_id: uuid.UUID,
    session_id: uuid.UUID,
    body: ChatSessionUpdateRequest,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    get_current_user_owned_job(db, current_user.id, job_id)
    session = _get_job_chat_session_or_404(
        db,
        current_user_id=current_user.id,
        job_id=job_id,
        session_id=session_id,
    )
    if body.session_title is not None:
        session.session_title = body.session_title.strip() or None
        session.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(session)
    return _serialize_chat_session(db, session)


@router.delete("/{job_id}/chat/sessions/{session_id}", status_code=204)
def delete_job_chat_session(
    job_id: uuid.UUID,
    session_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    get_current_user_owned_job(db, current_user.id, job_id)
    session = _get_job_chat_session_or_404(
        db,
        current_user_id=current_user.id,
        job_id=job_id,
        session_id=session_id,
    )
    db.delete(session)
    db.commit()


@router.get(
    "/{job_id}/chat/sessions/{session_id}/turns",
    response_model=list[ChatTurnResponse],
)
def list_job_chat_turns(
    job_id: uuid.UUID,
    session_id: uuid.UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    get_current_user_owned_job(db, current_user.id, job_id)
    _get_job_chat_session_or_404(
        db,
        current_user_id=current_user.id,
        job_id=job_id,
        session_id=session_id,
    )
    turns = (
        db.query(QueryTurn)
        .filter(QueryTurn.query_session_id == session_id)
        .order_by(QueryTurn.created_at.asc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [_serialize_chat_turn(turn) for turn in turns]


@router.post("/{job_id}/chat", response_model=ChatResponse)
def chat_about_job(
    job_id: uuid.UUID,
    body: ChatRequest,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    job = get_current_user_owned_job(db, current_user.id, job_id)
    job_description = get_job_scoped_jd(db, current_user.id, job_id)
    session = _resolve_chat_session(
        db,
        current_user_id=current_user.id,
        job_id=job_id,
        body=body,
    )
    saved_turns = (
        db.query(QueryTurn)
        .filter(QueryTurn.query_session_id == session.id)
        .order_by(QueryTurn.created_at.asc())
        .all()
    )
    history = _messages_from_turns(saved_turns)
    candidates = _load_job_candidates(db, job_id, body.candidate_limit)
    current_job = _build_job_chat_context(job=job, job_description=job_description)
    if _is_total_candidate_count_question(body.message):
        answer = _total_candidate_count_answer(body.message, len(candidates))
        turn = _persist_chat_turn(
            db,
            session=session,
            question=body.message,
            answer=answer,
            candidates_in_scope=len(candidates),
            candidate_limit=body.candidate_limit,
            matched_candidate_ids=[str(candidate["id"]) for candidate in candidates],
            route="candidate_count",
        )
        return ChatResponse(
            session_id=str(session.id),
            answer=answer,
            candidates_in_scope=len(candidates),
            session=_serialize_chat_session(db, session),
            turn=_serialize_chat_turn(turn),
        )
    history = list(history) + [HumanMessage(content=body.message)]
    graph_input = {
        "messages": history,
        "current_candidates": candidates,
        "current_job": current_job,
        "question": body.message,
        "router_output": None,
        "dsl_candidates": None,
        "llm_result": None,
        "answer": "",
        "trace_id": str(uuid.uuid4()),
        "trace_metadata": {
            "endpoint": "job_chat",
            "job_id": str(job_id),
            "session_id": str(session.id),
            "user_id": str(current_user.id),
        },
    }
    trace_id = graph_input["trace_id"]
    get_trace_logger().start_trace(
        trace_id=trace_id,
        metadata={
            "endpoint": "job_chat",
            "job_id": str(job_id),
            "session_id": str(session.id),
            "user_id": str(current_user.id),
            "question": body.message,
            "candidate_limit": body.candidate_limit,
            "current_job": current_job,
        },
        graph_input=graph_input,
    )
    try:
        result = get_graph().invoke(graph_input)
    except Exception as exc:
        get_trace_logger().finalize_trace(
            trace_id=trace_id,
            status="error",
            error=format_exception_payload(exc),
        )
        raise HTTPException(
            status_code=500, detail=f"Graph execution error: {exc}"
        ) from exc
    get_trace_logger().finalize_trace(
        trace_id=trace_id,
        status="success",
        graph_output=result,
    )
    dsl_pool = result.get("dsl_candidates")
    candidates_in_scope = len(dsl_pool) if dsl_pool is not None else len(candidates)
    matched_candidate_ids = None
    if dsl_pool is not None:
        matched_candidate_ids = [
            str(candidate["id"])
            for candidate in dsl_pool
            if isinstance(candidate, dict) and candidate.get("id") is not None
        ]
    answer = result.get("answer") or ""
    turn = _persist_chat_turn(
        db,
        session=session,
        question=body.message,
        answer=answer,
        candidates_in_scope=candidates_in_scope,
        candidate_limit=body.candidate_limit,
        matched_candidate_ids=matched_candidate_ids,
        route="graph",
    )
    return ChatResponse(
        session_id=str(session.id),
        answer=answer,
        candidates_in_scope=candidates_in_scope,
        session=_serialize_chat_session(db, session),
        turn=_serialize_chat_turn(turn),
    )
