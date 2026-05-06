import uuid
from pathlib import Path
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from src.models.candidate_profile import CandidateProfile
from src.models.deps import get_current_user, get_db
from src.models.job import Job
from src.models.resume_document import ResumeDocument
from src.models.user_account import UserAccount
from src.services.resume_service import (
    delete_resume,
    get_resume,
    list_resumes,
    parse_pdf_to_sections,
    update_resume,
)

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[5]
PDF_STORAGE_DIR = PROJECT_ROOT / "pdfs"


class BatchUploadResponse(BaseModel):
    total_files: int
    processed_files: int
    failed_files: int
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


class CandidateProfileResponse(BaseModel):
    id: str
    resume_document_id: str
    full_name: str
    phone: Optional[str]
    email: Optional[str]
    location_normalized: Optional[str]
    current_job_title: Optional[str]
    summary_text: Optional[str]
    skills_text: Optional[str]
    experience_text: Optional[str]
    experience_years: Optional[float]
    education_text: Optional[str]
    languages_text: Optional[str]
    projects_text: Optional[str]
    achievements_text: Optional[str]
    certifications_text: Optional[str]


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
        full_name=profile.full_name,
        phone=profile.phone,
        email=profile.email,
        location_normalized=profile.location_normalized,
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
        languages_text=profile.languages_text,
        projects_text=profile.projects_text,
        achievements_text=profile.achievements_text,
        certifications_text=profile.certifications_text,
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
        full_name=profile.full_name,
        phone=profile.phone,
        email=profile.email,
        location_normalized=profile.location_normalized,
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
        languages_text=profile.languages_text,
        projects_text=profile.projects_text,
        achievements_text=profile.achievements_text,
        certifications_text=profile.certifications_text,
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
    first_job = db.execute(
        select(Job).where(Job.owner_user_id == current_user.id).order_by(Job.created_at.asc())
    ).scalars().first()
    if first_job is None:
        return ResumeListResponse(total=0, items=[])
    items = list_resumes(db=db, job_id=first_job.id, upload_status=upload_status, limit=limit, offset=offset)
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
    result = db.execute(
        select(ResumeDocument)
        .join(Job, Job.id == ResumeDocument.job_id)
        .where(ResumeDocument.id == resume_id, Job.owner_user_id == current_user.id)
    ).scalars().first()
    result = get_resume(db=db, resume_id=resume_id) if result is not None else None
    if result is None:
        raise HTTPException(status_code=404, detail=f"Resume {resume_id} not found")
    return result


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
    owner_match = db.execute(
        select(ResumeDocument)
        .join(Job, Job.id == ResumeDocument.job_id)
        .where(ResumeDocument.id == resume_id, Job.owner_user_id == current_user.id)
    ).scalars().first()
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
        "Pass `delete_file=true` to also remove the physical PDF from disk."
    ),
)
def delete_resume_document(
    resume_id: uuid.UUID,
    delete_file: Annotated[
        bool,
        Query(description="Also delete the physical PDF file from disk"),
    ] = False,
    db=Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    owner_match = db.execute(
        select(ResumeDocument)
        .join(Job, Job.id == ResumeDocument.job_id)
        .where(ResumeDocument.id == resume_id, Job.owner_user_id == current_user.id)
    ).scalars().first()
    deleted = delete_resume(db=db, resume_id=resume_id, delete_file=delete_file) if owner_match is not None else False
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Resume {resume_id} not found")
    return DeleteResumeResponse(deleted=True, resume_id=str(resume_id))


@router.post(
    "/batch-parse",
    response_model=BatchUploadResponse,
    summary="Upload and parse multiple resume PDFs",
    description="Use the files picker to select multiple PDFs in one request (Ctrl/Shift for multi-select).",
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
    first_job = db.execute(
        select(Job).where(Job.owner_user_id == current_user.id).order_by(Job.created_at.asc())
    ).scalars().first()
    if first_job is None:
        raise HTTPException(status_code=400, detail="Create a job before uploading resumes")

    PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    stored_paths: List[str] = []
    original_filenames: List[str] = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400, detail=f"Invalid file type: {file.filename}"
            )

        original_name = Path(file.filename).name
        safe_name = f"{uuid.uuid4()}_{original_name}"
        target_path = PDF_STORAGE_DIR / safe_name
        content = await file.read()
        target_path.write_bytes(content)
        stored_paths.append(str(target_path))
        original_filenames.append(original_name)

    items = parse_pdf_to_sections(
        filepaths=stored_paths,
        db=db,
        job_id=first_job.id,
        uploaded_by_user_id=current_user.id,
        original_filenames=original_filenames,
    )

    processed_files = sum(1 for item in items if item.get("status") == "processed")
    failed_files = sum(1 for item in items if item.get("status") == "failed")

    return BatchUploadResponse(
        total_files=len(items),
        processed_files=processed_files,
        failed_files=failed_files,
        items=items,
    )
