from fastapi import APIRouter
from src.api.v1.endpoints import upload, chat

api_router = APIRouter()
api_router.include_router(upload.router, prefix="/upload", tags=["upload"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
