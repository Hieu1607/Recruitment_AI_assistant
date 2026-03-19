from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.api.dependencies.auth import CurrentUser, require_roles
from src.repositories.db import get_session
from src.services.interview.interview_question_service import (
    InterviewQuestionInput,
    interview_question_service,
)

router = APIRouter(prefix="/v1/interview-questions", tags=["interview-questions"])


class InterviewQuestionRequest(BaseModel):
    candidateId: str
    jobDescriptionId: str
    questionCount: int = Field(default=10, ge=3, le=25)


class InterviewQuestionItem(BaseModel):
    prompt: str
    category: str | None = None
    difficulty: str | None = None


class InterviewQuestionResponse(BaseModel):
    id: str
    candidateId: str
    jobDescriptionId: str
    questions: list[InterviewQuestionItem]


@router.post("", response_model=InterviewQuestionResponse, status_code=201)
def create_interview_questions(
    payload: InterviewQuestionRequest,
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[CurrentUser, Depends(require_roles("admin", "recruiter"))],
) -> InterviewQuestionResponse:
    user_id = uuid.UUID(current_user.user_id)
    created = interview_question_service.generate_questions(
        session,
        InterviewQuestionInput(
            candidate_id=uuid.UUID(payload.candidateId),
            job_description_id=uuid.UUID(payload.jobDescriptionId),
            generated_by_user_id=user_id,
            question_count=payload.questionCount,
        ),
    )
    session.commit()

    raw_questions = created.question_payload.get("questions", [])
    return InterviewQuestionResponse(
        id=str(created.id),
        candidateId=str(created.candidate_profile_id),
        jobDescriptionId=str(created.job_description_id),
        questions=[InterviewQuestionItem(**item) for item in raw_questions],
    )
