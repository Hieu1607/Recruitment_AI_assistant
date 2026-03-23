from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from src.database.base import Base

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    session_id = Column(String, index=True)
    role = Column(String) # user or ai
    content = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
