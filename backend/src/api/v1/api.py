from fastapi import APIRouter
from src.api.v1.endpoints import chat, interview_questions, outreach, resume, jobDescription, score, shortlist

api_router = APIRouter()
api_router.include_router(resume.router, prefix="/upload", tags=["upload"])
api_router.include_router(jobDescription.router, prefix="/job-descriptions", tags=["job-descriptions"])
api_router.include_router(score.router, prefix="/score", tags=["score"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(shortlist.router, prefix="/shortlist", tags=["shortlist"])
api_router.include_router(interview_questions.router, prefix="/interview-questions", tags=["interview-questions"])
api_router.include_router(outreach.router, prefix="/outreach", tags=["outreach"])
