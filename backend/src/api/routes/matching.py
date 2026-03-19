from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.api.dependencies.auth import CurrentUser, require_roles
from src.api.errors import AppError
from src.models.matching import JobDescription, MatchRun, MatchRunStatus
from src.repositories.candidate_repository import candidate_repository
from src.repositories.db import get_session
from src.repositories.match_repository import match_repository
from src.services.matching.batch_llm_scorer import execute_batch_scoring
from src.services.matching.batch_prompt_builder import build_batch_scoring_prompt
from src.services.matching.score_list_parser import persist_match_results

router = APIRouter(prefix="/v1", tags=["matching"])


class MatchRunRequest(BaseModel):
    jobDescriptionText: str = Field(min_length=20)
    candidateIds: list[str] = Field(min_length=1)
    scoringPromptTemplate: str = Field(min_length=20)
    scoreThreshold: float = Field(ge=0, le=100)


class WeightedComponentScore(BaseModel):
    criterionKey: str
    weight: float
    score: float
    weightedScore: float
    evidenceSummary: str | None = None


class MatchScoreItem(BaseModel):
    candidateId: str
    totalScore: float
    componentScores: list[WeightedComponentScore]
    passedThreshold: bool
    rationale: str


class MatchScoresList(BaseModel):
    matchRunId: str
    scoreThreshold: float
    scores: list[MatchScoreItem]


def _to_score_item(raw: Any) -> MatchScoreItem:
    component_scores = [WeightedComponentScore(**item) for item in raw.component_scores]
    return MatchScoreItem(
        candidateId=str(raw.candidate_profile_id),
        totalScore=float(raw.total_score),
        componentScores=component_scores,
        passedThreshold=bool(raw.passed_threshold),
        rationale=raw.rationale_summary,
    )


@router.post("/match-runs", response_model=MatchScoresList)
def create_match_run(
    payload: MatchRunRequest,
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[CurrentUser, Depends(require_roles("admin", "recruiter"))],
) -> MatchScoresList:
    candidate_ids = [uuid.UUID(candidate_id) for candidate_id in payload.candidateIds]
    candidates = candidate_repository.get_candidates_by_ids(session, candidate_ids)
    if len(candidates) != len(candidate_ids):
        raise AppError(code="not_found", message="One or more candidates were not found", status_code=404)

    try:
        user_id = uuid.UUID(current_user.user_id)
    except ValueError:
        user_id = uuid.UUID("00000000-0000-0000-0000-000000000000")

    job_description = JobDescription(jd_text=payload.jobDescriptionText, created_by_user_id=user_id)
    session.add(job_description)
    session.flush()

    match_run = MatchRun(
        job_description_id=job_description.id,
        initiated_by_user_id=user_id,
        scoring_prompt_template=payload.scoringPromptTemplate,
        score_threshold=payload.scoreThreshold,
        run_status=MatchRunStatus.RUNNING,
    )
    session.add(match_run)
    session.flush()

    prompt = build_batch_scoring_prompt(
        job_description_text=payload.jobDescriptionText,
        candidates=candidates,
        scoring_prompt_template=payload.scoringPromptTemplate,
    )
    raw_scores = execute_batch_scoring(prompt, candidates)
    persist_match_results(
        session,
        match_run_id=match_run.id,
        threshold=payload.scoreThreshold,
        raw_items=raw_scores,
    )

    match_run.run_status = MatchRunStatus.COMPLETED
    match_run.completed_at = datetime.utcnow()
    session.add(match_run)
    session.commit()

    scores = match_repository.list_results(
        session,
        match_run_id=match_run.id,
        threshold=payload.scoreThreshold,
    )
    return MatchScoresList(
        matchRunId=str(match_run.id),
        scoreThreshold=payload.scoreThreshold,
        scores=[_to_score_item(item) for item in scores],
    )
