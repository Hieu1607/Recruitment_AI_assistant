from fastapi import APIRouter, File, UploadFile, HTTPException
from src.schemas.ai_schema import DocumentUploadResponse

router = APIRouter()

@router.post("/", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Simple mockup
    return DocumentUploadResponse(
        filename=file.filename,
        message="Document uploaded successfully",
        document_id=1
    )
