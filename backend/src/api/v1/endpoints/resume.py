import uuid
from enum import Enum as PyEnum
from pathlib import Path
from typing import Annotated, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from src.models.session import SessionLocal
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
        None, min_length=1, description="New display name for the file (omit to leave unchanged)"
    )
    upload_status: Optional[str] = Field(
        None, description="Override status: uploaded | processing | processed | failed"
    )


class DeleteResumeResponse(BaseModel):
    deleted: bool
    resume_id: str


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
        Query(description="Filter by status: uploaded | processing | processed | failed"),
    ] = None,
    uploaded_by_user_id: Annotated[
        Optional[uuid.UUID],
        Query(description="Filter by uploader UUID"),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=200, description="Max records to return")] = 50,
    offset: Annotated[int, Query(ge=0, description="Records to skip")] = 0,
):
    db = SessionLocal()
    try:
        items = list_resumes(
            db=db,
            upload_status=upload_status,
            uploaded_by_user_id=uploaded_by_user_id,
            limit=limit,
            offset=offset,
        )
    finally:
        db.close()
    return ResumeListResponse(total=len(items), items=items)


@router.get(
    "/{resume_id}",
    response_model=ResumeResponse,
    summary="Get a single resume document",
)
def get_resume_document(resume_id: uuid.UUID):
    db = SessionLocal()
    try:
        result = get_resume(db=db, resume_id=resume_id)
    finally:
        db.close()
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
def update_resume_document(resume_id: uuid.UUID, body: ResumeUpdateRequest):
    db = SessionLocal()
    try:
        result = update_resume(
            db=db,
            resume_id=resume_id,
            original_file_name=body.original_file_name,
            upload_status=body.upload_status,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        db.close()
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
):
    db = SessionLocal()
    try:
        deleted = delete_resume(db=db, resume_id=resume_id, delete_file=delete_file)
    finally:
        db.close()
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
    uploaded_by_user_id: Annotated[
        Optional[str],
        Form(description="Optional uploader user UUID."),
    ] = None,
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    parsed_user_id: Optional[uuid.UUID] = None
    if uploaded_by_user_id:
        try:
            parsed_user_id = uuid.UUID(uploaded_by_user_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="uploaded_by_user_id must be a valid UUID") from exc

    PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    stored_paths: List[str] = []
    original_filenames: List[str] = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}")

        original_name = Path(file.filename).name
        safe_name = f"{uuid.uuid4()}_{original_name}"
        target_path = PDF_STORAGE_DIR / safe_name
        content = await file.read()
        target_path.write_bytes(content)
        stored_paths.append(str(target_path))
        original_filenames.append(original_name)

    db = SessionLocal()
    try:
        items = parse_pdf_to_sections(
            filepaths=stored_paths,
            db=db,
            uploaded_by_user_id=parsed_user_id,
            original_filenames=original_filenames,
        )
    finally:
        db.close()

    processed_files = sum(1 for item in items if item.get("status") == "processed")
    failed_files = sum(1 for item in items if item.get("status") == "failed")

    return BatchUploadResponse(
        total_files=len(items),
        processed_files=processed_files,
        failed_files=failed_files,
        items=items,
    )
