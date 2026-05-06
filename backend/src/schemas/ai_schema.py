from typing import Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


class DocumentUploadResponse(BaseModel):
    filename: str
    message: str
    document_id: int
