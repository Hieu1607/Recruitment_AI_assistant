import uuid
from pathlib import Path
from typing import Annotated, List, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from src.models.candidate_profile import CandidateProfile
from src.models.deps import get_current_user, get_db
from src.models.job import Job
from src.models.resume_document import ResumeDocument
from src.models.user_account import UserAccount
from src.services.object_storage import build_object_key, get_object_storage
from src.services.resume_service import (
    _get_resume_extraction_mode,
    create_resume_document,
    delete_resume,
    get_resume,
    list_resumes,
    parse_pdf_to_sections,
    update_resume,
)
from worker.tasks import process_resume

router = APIRouter()


def _build_pdf_content_disposition(filename: str) -> str:
    safe_name = Path(filename or "resume.pdf").name or "resume.pdf"
    ascii_fallback = "".join(char if ord(char) < 128 else "_" for char in safe_name)
    ascii_fallback = ascii_fallback.replace('"', "_")
    if not ascii_fallback:
        ascii_fallback = "resume.pdf"
    encoded_filename = quote(safe_name, safe="")
    return (
        f'inline; filename="{ascii_fallback}"; '
        f"filename*=UTF-8''{encoded_filename}"
    )


class BatchUploadResponse(BaseModel):
    total_files: int
    queued_files: int
    items: List[dict]


class DocumentUploadResponse(BaseModel):
    filename: str
    message: str
    document_id: int


class ResumeResponse(BaseModel):
    id: str
    original_file_name: str
    storage_uri: str
    upload_status: str
    extraction_mode: Optional[str] = None
    duplicate_group_key: Optional[str]
    uploaded_by_user_id: str
    uploaded_at: Optional[str]
    processed_at: Optional[str]
    retention_expires_at: Optional[str]


class ResumeListResponse(BaseModel):
    total: int
    items: List[ResumeResponse]


class ResumeUpdateRequest(BaseModel):
    original_file_name: Optional[str] = Field(
        None,
        min_length=1,
        description="New display name for the file (omit to leave unchanged)",
    )
    upload_status: Optional[str] = Field(
        None, description="Override status: uploaded | processing | processed | failed"
    )


class DeleteResumeResponse(BaseModel):
    deleted: bool
    resume_id: str


class StructuredLinkResponse(BaseModel):
    url: str
    label: Optional[str] = None


class StructuredEntryResponse(BaseModel):
    title: Optional[str] = None
    subtitle: Optional[str] = None
    role: Optional[str] = None
    location: Optional[str] = None
    dateRange: Optional[str] = None
    description: Optional[str] = None
    bullets: List[str] = Field(default_factory=list)
    links: List[StructuredLinkResponse] = Field(default_factory=list)
    metadata: List[str] = Field(default_factory=list)


class StructuredSectionResponse(BaseModel):
    entries: List[StructuredEntryResponse] = Field(default_factory=list)
    rawText: Optional[str] = None


class StructuredSummaryResponse(BaseModel):
    text: Optional[str] = None
    links: List[StructuredLinkResponse] = Field(default_factory=list)


class StructuredProfileResponse(BaseModel):
    summary: Optional[StructuredSummaryResponse] = None
    experience: Optional[StructuredSectionResponse] = None
    education: Optional[StructuredSectionResponse] = None
    projects: Optional[StructuredSectionResponse] = None
    skills: Optional[StructuredSectionResponse] = None
    languages: Optional[StructuredSectionResponse] = None
    achievements: Optional[StructuredSectionResponse] = None
    publications: Optional[StructuredSectionResponse] = None
    certifications: Optional[StructuredSectionResponse] = None
    references: Optional[StructuredSectionResponse] = None
    other: Optional[StructuredSectionResponse] = None


class CandidateProfileResponse(BaseModel):
    id: str
    resume_document_id: str
    extraction_mode: Optional[str] = None
    full_name: str
    submitted_full_name: Optional[str]
    phone: Optional[str]
    email: Optional[str]
    submitted_email: Optional[str]
    location_normalized: Optional[str]
    contact: Optional[str]
    current_job_title: Optional[str]
    graduation_status: str
    ever_studied_abroad: bool
    major: Optional[str]
    cpa: Optional[str]
    summary_text: Optional[str]
    skills_text: Optional[str]
    experience_text: Optional[str]
    experience_years: Optional[float]
    education_text: Optional[str]
    languages_text: Optional[str]
    projects_text: Optional[str]
    achievements_text: Optional[str]
    publications_text: Optional[str]
    certifications_text: Optional[str]
    references_text: Optional[str]
    other_text: Optional[str]
    structured_profile: Optional[StructuredProfileResponse] = None


# ---------------------------------------------------------------------------
# Candidate profile endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/profiles/{profile_id}",
    response_model=CandidateProfileResponse,
    summary="Get a candidate profile by its own UUID",
)
def get_candidate_profile_by_id(
    profile_id: uuid.UUID,
    db=Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    profile = db.execute(
        select(CandidateProfile)
        .join(ResumeDocument, ResumeDocument.id == CandidateProfile.resume_document_id)
        .join(Job, Job.id == ResumeDocument.job_id)
        .where(CandidateProfile.id == profile_id, Job.owner_user_id == current_user.id)
    ).scalar_one_or_none()
    if profile is None:
        raise HTTPException(status_code=404, detail="Candidate profile not found")
    return CandidateProfileResponse(
        id=str(profile.id),
        resume_document_id=str(profile.resume_document_id),
        extraction_mode=_get_resume_extraction_mode(db, profile.resume_document_id),
        full_name=profile.full_name,
        submitted_full_name=profile.submitted_full_name,
        phone=profile.phone,
        email=profile.email,
        submitted_email=profile.submitted_email,
        location_normalized=profile.location_normalized,
        contact=profile.contact,
        current_job_title=profile.current_job_title,
        graduation_status=profile.graduation_status,
        ever_studied_abroad=profile.ever_studied_abroad,
        major=profile.major,
        cpa=profile.cpa,
        summary_text=profile.summary_text,
        skills_text=profile.skills_text,
        experience_text=profile.experience_text,
        experience_years=(
            float(profile.experience_years)
            if profile.experience_years is not None
            else None
        ),
        education_text=profile.education_text,
        languages_text=profile.languages_text,
        projects_text=profile.projects_text,
        achievements_text=profile.achievements_text,
        publications_text=profile.publications_text,
        certifications_text=profile.certifications_text,
        references_text=profile.references_text,
        other_text=profile.other_text,
        structured_profile=profile.structured_profile,
    )


@router.get(
    "/{resume_id}/profile",
    response_model=CandidateProfileResponse,
    summary="Get the parsed candidate profile for a resume",
)
def get_candidate_profile(
    resume_id: uuid.UUID,
    db=Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    profile = db.execute(
        select(CandidateProfile)
        .join(ResumeDocument, ResumeDocument.id == CandidateProfile.resume_document_id)
        .join(Job, Job.id == ResumeDocument.job_id)
        .where(
            CandidateProfile.resume_document_id == resume_id,
            Job.owner_user_id == current_user.id,
        )
    ).scalar_one_or_none()
    if profile is None:
        raise HTTPException(
            status_code=404, detail="Candidate profile not found for this resume"
        )
    return CandidateProfileResponse(
        id=str(profile.id),
        resume_document_id=str(profile.resume_document_id),
        extraction_mode=_get_resume_extraction_mode(db, profile.resume_document_id),
        full_name=profile.full_name,
        submitted_full_name=profile.submitted_full_name,
        phone=profile.phone,
        email=profile.email,
        submitted_email=profile.submitted_email,
        location_normalized=profile.location_normalized,
        contact=profile.contact,
        current_job_title=profile.current_job_title,
        graduation_status=profile.graduation_status,
        ever_studied_abroad=profile.ever_studied_abroad,
        major=profile.major,
        cpa=profile.cpa,
        summary_text=profile.summary_text,
        skills_text=profile.skills_text,
        experience_text=profile.experience_text,
        experience_years=(
            float(profile.experience_years)
            if profile.experience_years is not None
            else None
        ),
        education_text=profile.education_text,
        languages_text=profile.languages_text,
        projects_text=profile.projects_text,
        achievements_text=profile.achievements_text,
        publications_text=profile.publications_text,
        certifications_text=profile.certifications_text,
        references_text=profile.references_text,
        other_text=profile.other_text,
        structured_profile=profile.structured_profile,
    )


# ---------------------------------------------------------------------------
# Read endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/",
    response_model=ResumeListResponse,
    summary="List resume documents",
)
def list_resume_documents(
    upload_status: Annotated[
        Optional[str],
        Query(
            description="Filter by status: uploaded | processing | processed | failed"
        ),
    ] = None,
    limit: Annotated[
        int, Query(ge=1, le=200, description="Max records to return")
    ] = 50,
    offset: Annotated[int, Query(ge=0, description="Records to skip")] = 0,
    db=Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    first_job = (
        db.execute(
            select(Job)
            .where(Job.owner_user_id == current_user.id)
            .order_by(Job.created_at.asc())
        )
        .scalars()
        .first()
    )
    if first_job is None:
        return ResumeListResponse(total=0, items=[])
    items = list_resumes(
        db=db,
        job_id=first_job.id,
        upload_status=upload_status,
        limit=limit,
        offset=offset,
    )
    return ResumeListResponse(total=len(items), items=items)


@router.get(
    "/{resume_id}",
    response_model=ResumeResponse,
    summary="Get a single resume document",
)
def get_resume_document(
    resume_id: uuid.UUID,
    db=Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    result = (
        db.execute(
            select(ResumeDocument)
            .join(Job, Job.id == ResumeDocument.job_id)
            .where(ResumeDocument.id == resume_id, Job.owner_user_id == current_user.id)
        )
        .scalars()
        .first()
    )
    result = get_resume(db=db, resume_id=resume_id) if result is not None else None
    if result is None:
        raise HTTPException(status_code=404, detail=f"Resume {resume_id} not found")
    return result


@router.get(
    "/{resume_id}/file",
    summary="Download or preview a resume PDF",
)
def get_resume_document_file(
    resume_id: uuid.UUID,
    db=Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    resume = (
        db.execute(
            select(ResumeDocument)
            .join(Job, Job.id == ResumeDocument.job_id)
            .where(ResumeDocument.id == resume_id, Job.owner_user_id == current_user.id)
        )
        .scalars()
        .first()
    )
    if resume is None:
        raise HTTPException(status_code=404, detail=f"Resume {resume_id} not found")

    storage_uri = str(resume.storage_uri or "").strip()
    if not storage_uri:
        raise HTTPException(status_code=404, detail="Resume file is unavailable")

    try:
        if storage_uri.startswith("s3://"):
            pdf_bytes = get_object_storage().download_bytes(storage_uri)
        else:
            pdf_bytes = Path(storage_uri).read_bytes()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Resume file not found") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="Failed to load resume file") from exc

    headers = {
        "Content-Disposition": _build_pdf_content_disposition(
            resume.original_file_name
        ),
        "Cache-Control": "no-store",
    }
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)


# ---------------------------------------------------------------------------
# Update endpoint
# ---------------------------------------------------------------------------


@router.patch(
    "/{resume_id}",
    response_model=ResumeResponse,
    summary="Update a resume document",
    description="Patch `original_file_name` and/or `upload_status`. All fields are optional.",
)
def update_resume_document(
    resume_id: uuid.UUID,
    body: ResumeUpdateRequest,
    db=Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    owner_match = (
        db.execute(
            select(ResumeDocument)
            .join(Job, Job.id == ResumeDocument.job_id)
            .where(ResumeDocument.id == resume_id, Job.owner_user_id == current_user.id)
        )
        .scalars()
        .first()
    )
    if owner_match is None:
        raise HTTPException(status_code=404, detail=f"Resume {resume_id} not found")
    try:
        result = update_resume(
            db=db,
            resume_id=resume_id,
            original_file_name=body.original_file_name,
            upload_status=body.upload_status,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail=f"Resume {resume_id} not found")
    return result


# ---------------------------------------------------------------------------
# Delete endpoint
# ---------------------------------------------------------------------------


@router.delete(
    "/{resume_id}",
    response_model=DeleteResumeResponse,
    summary="Delete a resume document",
    description=(
        "Hard-deletes the ResumeDocument record and all cascade relations "
        "(CandidateProfile, ExtractionTrace). "
        "Pass `delete_file=true` to also remove the stored PDF object."
    ),
)
def delete_resume_document(
    resume_id: uuid.UUID,
    delete_file: Annotated[
        bool,
        Query(description="Also delete the stored PDF object"),
    ] = False,
    db=Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    owner_match = (
        db.execute(
            select(ResumeDocument)
            .join(Job, Job.id == ResumeDocument.job_id)
            .where(ResumeDocument.id == resume_id, Job.owner_user_id == current_user.id)
        )
        .scalars()
        .first()
    )
    deleted = (
        delete_resume(db=db, resume_id=resume_id, delete_file=delete_file)
        if owner_match is not None
        else False
    )
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Resume {resume_id} not found")
    return DeleteResumeResponse(deleted=True, resume_id=str(resume_id))


@router.post(
    "/batch-parse",
    response_model=BatchUploadResponse,
    status_code=202,
    summary="Upload and queue multiple resume PDFs for parsing",
    description="Use the files picker to select multiple PDFs in one request (Ctrl/Shift for multi-select). Parsing runs asynchronously via a background worker.",
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["files"],
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                                "description": "Select one or more PDF files.",
                            },
                            "uploaded_by_user_id": {
                                "type": "string",
                                "nullable": True,
                                "description": "Optional uploader user UUID.",
                            },
                        },
                    }
                }
            },
            "required": True,
        }
    },
)
async def upload_and_parse_resumes(
    files: Annotated[
        list[UploadFile],
        File(
            ...,
            description="Select one or more PDF files.",
        ),
    ],
    db=Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    first_job = (
        db.execute(
            select(Job)
            .where(Job.owner_user_id == current_user.id)
            .order_by(Job.created_at.asc())
        )
        .scalars()
        .first()
    )
    if first_job is None:
        raise HTTPException(
            status_code=400, detail="Create a job before uploading resumes"
        )

    object_storage = get_object_storage()
    items: List[dict] = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400, detail=f"Invalid file type: {file.filename}"
            )

        original_name = Path(file.filename).name
        storage_uri = object_storage.upload_bytes(
            data=await file.read(),
            object_key=build_object_key(
                prefix=f"resumes/{first_job.id}",
                original_filename=original_name,
            ),
            content_type=file.content_type or "application/pdf",
        )
        resume = create_resume_document(
            db=db,
            storage_uri=storage_uri,
            original_file_name=original_name,
            job_id=first_job.id,
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

    return BatchUploadResponse(
        total_files=len(items),
        queued_files=len(items),
        items=items,
    )
