from sqlalchemy.orm import Session
from src.models.document import Document
from src.schemas.ai_schema import DocumentUploadResponse

def create_document(db: Session, filename: str, filepath: str, extracted_text: str, user_id: int):
    db_obj = Document(
        filename=filename,
        filepath=filepath,
        extracted_text=extracted_text,
        user_id=user_id
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj

def get_document(db: Session, doc_id: int):
    return db.query(Document).filter(Document.id == doc_id).first()
