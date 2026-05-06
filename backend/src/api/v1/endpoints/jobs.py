from __future__ import annotations

import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models.candidate_profile import CandidateProfile
from src.models.deps import get_current_user, get_db
from src.models.job import Job
from src.models.job_matching import JobDescription
from src.models.resume_document import ResumeDocument
from src.models.user_account import UserAccount
from src.services.ai_agent.graph import get_graph
from src.services.job_description_service import _jd_to_dict
from src.services.job_scope import (
    get_current_user_owned_job,
    get_job_scoped_candidate,
    get_job_scoped_jd,
    get_job_scoped_resume,
    require_job_scoped_jd,
    serialize_job,
)
from src.services.resume_service import _resume_to_dict, parse_pdf_to_sections
from src.services.score_candidate import score_candidates

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[5]
PDF_STORAGE_DIR = PROJECT_ROOT / "pdfs"
_sessions: dict[str, list[Any]] = {}


class JobCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    status: str = Field(default="active", min_length=1, max_length=50)


class JobUpdateRequest(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=255)
    status: Optional[str] = Field(default=None, min_length=1, max_length=50)


class JobResponse(BaseModel):
    id: str
    owner_user_id: str
    title: str
    status: str
    created_at: str
    updated_at: str
    archived_at: Optional[str]


class JobListResponse(BaseModel):
    items: list[JobResponse]
    total: int


class JobDescriptionRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=255)
    jd_text: str = Field(..., min_length=1)
    is_active: bool = True


class JobDescriptionPatchRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=255)
    jd_text: Optional[str] = None
    is_active: Optional[bool] = None


class ResumeResponse(BaseModel):
    id: str
    job_id: str
    original_file_name: str
    storage_uri: str
    upload_status: str
    duplicate_group_key: Optional[str]
    uploaded_by_user_id: str
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
        experience_years=float(profile.experience_years) if profile.experience_years is not None else None,
        education_text=profile.education_text,
    )


def _load_job_candidates(db: Session, job_id: uuid.UUID, limit: int) -> list[dict[str, Any]]:
    rows = db.execute(
        select(CandidateProfile)
        .join(ResumeDocument, ResumeDocument.id == CandidateProfile.resume_document_id)
        .where(ResumeDocument.job_id == job_id)
        .limit(limit)
    ).scalars().all()
    return [
        {
            "id": str(r.id),
            "full_name": r.full_name,
            "phone": r.phone,
            "email": r.email,
            "location_normalized": r.location_normalized,
            "contact": r.contact,
            "current_job_title": r.current_job_title,
            "educated": r.educated,
            "ever_studied_abroad": r.ever_studied_abroad,
            "major": r.major,
            "cpa": r.cpa,
            "education_text": r.education_text,
            "experience_text": r.experience_text,
            "experience_years": float(r.experience_years) if r.experience_years is not None else None,
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


@router.post("/", response_model=JobResponse, status_code=201)
def create_job(
    body: JobCreateRequest,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    job = Job(owner_user_id=current_user.id, title=body.title.strip(), status=body.status.strip())
    db.add(job)
    db.commit()
    db.refresh(job)
    return JobResponse(**serialize_job(job))


@router.get("/", response_model=JobListResponse)
def list_jobs(
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    jobs = db.execute(
        select(Job).where(Job.owner_user_id == current_user.id).order_by(Job.updated_at.desc())
    ).scalars().all()
    return JobListResponse(items=[JobResponse(**serialize_job(job)) for job in jobs], total=len(jobs))


@router.get("/{job_id}", response_model=JobResponse)
def get_job(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    return JobResponse(**serialize_job(get_current_user_owned_job(db, current_user.id, job_id)))


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
            created_by_user_id=current_user.id,
            is_active=body.is_active,
        )
        db.add(existing)
    else:
        existing.title = body.title
        existing.jd_text = body.jd_text.strip()
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
    query = db.query(ResumeDocument).filter(ResumeDocument.job_id == job_id)
    if upload_status is not None:
        query = query.filter(ResumeDocument.upload_status == upload_status)
    rows = query.order_by(ResumeDocument.uploaded_at.desc()).offset(offset).limit(limit).all()
    return ResumeListResponse(items=[ResumeResponse(**_resume_to_dict(r)) for r in rows], total=len(rows))


@router.post("/{job_id}/resumes", status_code=201)
async def upload_job_resumes(
    job_id: uuid.UUID,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    get_current_user_owned_job(db, current_user.id, job_id)
    PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    stored_paths: list[str] = []
    original_filenames: list[str] = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}")
        original_name = Path(file.filename).name
        safe_name = f"{uuid.uuid4()}_{original_name}"
        target_path = PDF_STORAGE_DIR / safe_name
        target_path.write_bytes(await file.read())
        stored_paths.append(str(target_path))
        original_filenames.append(original_name)
    items = parse_pdf_to_sections(
        filepaths=stored_paths,
        db=db,
        job_id=job_id,
        uploaded_by_user_id=current_user.id,
        original_filenames=original_filenames,
    )
    processed_files = sum(1 for item in items if item.get("status") == "processed")
    failed_files = sum(1 for item in items if item.get("status") == "failed")
    return {
        "total_files": len(items),
        "processed_files": processed_files,
        "failed_files": failed_files,
        "items": items,
    }


@router.get("/{job_id}/resumes/{resume_id}", response_model=ResumeResponse)
def get_job_resume(
    job_id: uuid.UUID,
    resume_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    return ResumeResponse(**_resume_to_dict(get_job_scoped_resume(db, current_user.id, job_id, resume_id)))


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
    return ResumeResponse(**_resume_to_dict(resume))


@router.delete("/{job_id}/resumes/{resume_id}")
def delete_job_resume(
    job_id: uuid.UUID,
    resume_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    resume = get_job_scoped_resume(db, current_user.id, job_id, resume_id)
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
    rows = db.execute(
        select(CandidateProfile)
        .join(ResumeDocument, ResumeDocument.id == CandidateProfile.resume_document_id)
        .where(ResumeDocument.job_id == job_id)
        .order_by(CandidateProfile.created_at.desc())
    ).scalars().all()
    return CandidateListResponse(items=[_serialize_candidate(row) for row in rows], total=len(rows))


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
    return score_candidates(
        db=db,
        job_description_id=jd.id,
        initiated_by_user_id=current_user.id,
        score_threshold=Decimal(str(body.score_threshold)),
        candidate_profile_ids=body.candidate_profile_ids,
        section_weights=body.section_weights,
        batch_size=body.batch_size,
    )


@router.post("/{job_id}/chat", response_model=ChatResponse)
def chat_about_job(
    job_id: uuid.UUID,
    body: ChatRequest,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    get_current_user_owned_job(db, current_user.id, job_id)
    session_id = body.session_id or str(uuid.uuid4())
    history = _sessions.get(f"{job_id}:{session_id}", [])
    candidates = _load_job_candidates(db, job_id, body.candidate_limit)
    history = list(history) + [HumanMessage(content=body.message)]
    graph_input = {
        "messages": history,
        "current_candidates": candidates,
        "question": body.message,
        "router_output": None,
        "dsl_candidates": None,
        "llm_result": None,
        "answer": "",
    }
    try:
        result = get_graph().invoke(graph_input)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Graph execution error: {exc}") from exc
    _sessions[f"{job_id}:{session_id}"] = result.get("messages") or history
    dsl_pool = result.get("dsl_candidates")
    return ChatResponse(
        session_id=session_id,
        answer=result.get("answer") or "",
        candidates_in_scope=len(dsl_pool) if dsl_pool is not None else len(candidates),
    )
