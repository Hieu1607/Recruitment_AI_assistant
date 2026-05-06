"""CRUD endpoints for InterviewQuestionSet.

Route map
---------
  POST   /interview-questions/           create a question set
  GET    /interview-questions/           list (filter by user / candidate / JD)
  GET    /interview-questions/{id}       get single question set
  PATCH  /interview-questions/{id}       update question_payload
  DELETE /interview-questions/{id}       delete
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from src.models.candidate_profile import CandidateProfile
from src.models.deps import get_current_user
from src.models.job_matching import InterviewQuestionSet, JobDescription
from src.models.session import SessionLocal

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_or_404(db, model, record_id: uuid.UUID, label: str):
    obj = db.get(model, record_id)
    if obj is None:
        raise HTTPException(status_code=404, detail=f"{label} '{record_id}' not found")
    return obj


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class QuestionSetCreateRequest(BaseModel):
    candidate_profile_id: uuid.UUID
    job_description_id: uuid.UUID
    generated_by_user_id: uuid.UUID
    question_payload: Dict[str, Any] = Field(
        ..., description="AI-generated question content"
    )


class GenerateQuestionsRequest(BaseModel):
    candidate_profile_id: uuid.UUID
    job_description_id: uuid.UUID


class QuestionSetUpdateRequest(BaseModel):
    question_payload: Dict[str, Any] = Field(
        ..., description="Updated question content"
    )


class QuestionSetResponse(BaseModel):
    id: str
    candidate_profile_id: str
    candidate_full_name: Optional[str]
    job_description_id: str
    job_description_title: Optional[str]
    generated_by_user_id: str
    question_payload: Dict[str, Any]
    created_at: datetime


class QuestionSetListResponse(BaseModel):
    total: int
    items: List[QuestionSetResponse]


# ---------------------------------------------------------------------------
# Serialiser
# ---------------------------------------------------------------------------


def _ser(qs: InterviewQuestionSet) -> QuestionSetResponse:
    candidate_name: Optional[str] = None
    jd_title: Optional[str] = None

    if qs.candidate_profile is not None:
        candidate_name = qs.candidate_profile.full_name
    if qs.job_description is not None:
        jd_title = qs.job_description.title

    return QuestionSetResponse(
        id=str(qs.id),
        candidate_profile_id=str(qs.candidate_profile_id),
        candidate_full_name=candidate_name,
        job_description_id=str(qs.job_description_id),
        job_description_title=jd_title,
        generated_by_user_id=str(qs.generated_by_user_id),
        question_payload=qs.question_payload,
        created_at=qs.created_at,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/", response_model=QuestionSetResponse, status_code=201)
def create_question_set(body: QuestionSetCreateRequest):
    db = SessionLocal()
    try:
        _get_or_404(db, CandidateProfile, body.candidate_profile_id, "CandidateProfile")
        _get_or_404(db, JobDescription, body.job_description_id, "JobDescription")

        qs = InterviewQuestionSet(
            candidate_profile_id=body.candidate_profile_id,
            job_description_id=body.job_description_id,
            generated_by_user_id=body.generated_by_user_id,
            question_payload=body.question_payload,
        )
        db.add(qs)
        db.commit()
        db.refresh(qs)
        return _ser(qs)
    finally:
        db.close()


@router.get("/", response_model=QuestionSetListResponse)
def list_question_sets(
    generated_by_user_id: Optional[uuid.UUID] = Query(
        None, description="Filter by creator user"
    ),
    candidate_profile_id: Optional[uuid.UUID] = Query(
        None, description="Filter by candidate"
    ),
    job_description_id: Optional[uuid.UUID] = Query(
        None, description="Filter by job description"
    ),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    db = SessionLocal()
    try:
        query = db.query(InterviewQuestionSet)

        if generated_by_user_id is not None:
            query = query.filter(
                InterviewQuestionSet.generated_by_user_id == generated_by_user_id
            )
        if candidate_profile_id is not None:
            query = query.filter(
                InterviewQuestionSet.candidate_profile_id == candidate_profile_id
            )
        if job_description_id is not None:
            query = query.filter(
                InterviewQuestionSet.job_description_id == job_description_id
            )

        total = query.count()
        rows = (
            query.order_by(InterviewQuestionSet.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        return QuestionSetListResponse(total=total, items=[_ser(r) for r in rows])
    finally:
        db.close()


@router.get("/{question_set_id}", response_model=QuestionSetResponse)
def get_question_set(question_set_id: uuid.UUID):
    db = SessionLocal()
    try:
        qs = _get_or_404(
            db, InterviewQuestionSet, question_set_id, "InterviewQuestionSet"
        )
        return _ser(qs)
    finally:
        db.close()


@router.patch("/{question_set_id}", response_model=QuestionSetResponse)
def update_question_set(question_set_id: uuid.UUID, body: QuestionSetUpdateRequest):
    db = SessionLocal()
    try:
        qs = _get_or_404(
            db, InterviewQuestionSet, question_set_id, "InterviewQuestionSet"
        )
        qs.question_payload = body.question_payload
        db.commit()
        db.refresh(qs)
        return _ser(qs)
    finally:
        db.close()


@router.delete("/{question_set_id}", status_code=204)
def delete_question_set(question_set_id: uuid.UUID):
    db = SessionLocal()
    try:
        qs = _get_or_404(
            db, InterviewQuestionSet, question_set_id, "InterviewQuestionSet"
        )
        db.delete(qs)
        db.commit()
    finally:
        db.close()


@router.post("/generate", response_model=QuestionSetResponse, status_code=201)
def generate_question_set(
    body: GenerateQuestionsRequest,
    current_user=Depends(get_current_user),
):
    from src.prompts.build_prompts import build_prompts
    from src.services.llm_service import LLMProvider, LLMProviderError

    db = SessionLocal()
    try:
        candidate = _get_or_404(
            db, CandidateProfile, body.candidate_profile_id, "CandidateProfile"
        )
        jd = _get_or_404(db, JobDescription, body.job_description_id, "JobDescription")

        candidate_data = {
            "full_name": candidate.full_name,
            "current_job_title": candidate.current_job_title,
            "experience": candidate.experience_text,
            "skills": candidate.skills_text,
            "education": candidate.education_text,
            "projects": candidate.projects_text,
            "summary": candidate.summary_text,
        }

        prompt = build_prompts.build_interview_questions_prompt(
            candidate_data=candidate_data,
            job_description_text=jd.jd_text,
        )

        try:
            llm = LLMProvider()
            response = llm.generate(prompt)
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("```", 2)[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.rsplit("```", 1)[0].strip()
            question_payload = json.loads(text)
        except (LLMProviderError, json.JSONDecodeError, Exception) as exc:
            raise HTTPException(status_code=502, detail=f"LLM generation failed: {exc}")

        qs = InterviewQuestionSet(
            candidate_profile_id=body.candidate_profile_id,
            job_description_id=body.job_description_id,
            generated_by_user_id=current_user.id,
            question_payload=question_payload,
        )
        db.add(qs)
        db.commit()
        db.refresh(qs)
        return _ser(qs)
    finally:
        db.close()
