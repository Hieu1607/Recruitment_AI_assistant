from pydantic import BaseModel
from typing import List, Optional

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
