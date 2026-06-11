from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.core.config import settings
from src.models.deps import get_current_user
from src.models.user_account import UserAccount
from src.services.object_storage import build_object_key, get_object_storage


router = APIRouter()


class OutreachAssetUploadResponse(BaseModel):
    storage_uri: str
    asset_url: str
    content_type: str
    filename: str


@router.post("/upload", response_model=OutreachAssetUploadResponse, status_code=201)
async def upload_outreach_asset(
    file: UploadFile = File(...),
    current_user: UserAccount = Depends(get_current_user),
):
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    original_filename = Path(file.filename or "image.png").name
    object_key = build_object_key(
        prefix=f"outreach/{current_user.id}",
        original_filename=original_filename,
    )
    storage = get_object_storage()
    storage_uri = storage.upload_bytes(
        data=payload,
        object_key=object_key,
        content_type=content_type,
        bucket=settings.MINIO_OUTREACH_BUCKET,
    )
    return OutreachAssetUploadResponse(
        storage_uri=storage_uri,
        asset_url=storage.public_object_url(storage_uri),
        content_type=content_type,
        filename=original_filename,
    )
