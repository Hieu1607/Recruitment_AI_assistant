from src.api.routes.candidates import router as candidates_router
from src.api.routes.interview_questions import router as interview_questions_router
from src.api.routes.matching import router as matching_router
from src.api.routes.outreach import router as outreach_router
from src.api.routes.query import router as query_router
from src.api.routes.resumes import router as resumes_router
from src.api.routes.shortlists import router as shortlists_router

__all__ = [
	"candidates_router",
	"matching_router",
	"resumes_router",
	"query_router",
	"shortlists_router",
	"outreach_router",
	"interview_questions_router",
]
