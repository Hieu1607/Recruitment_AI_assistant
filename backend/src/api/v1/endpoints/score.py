import uuid
from decimal import Decimal
from typing import Annotated, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select

from src.models.deps import get_current_user, get_db
from src.models.job import Job
from src.models.job_matching import JobDescription
from src.models.user_account import UserAccount
from src.services.score_candidate import score_candidates
from src.services.scoring_errors import ScoringProviderLimitError

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SectionWeights(BaseModel):
    """
    Per-section scoring weights. Only fields explicitly set by the caller are
    included in the LLM prompt; every omitted section defaults to weight = 0.

    Leave a field as None (or omit it entirely) to exclude it from scoring.
    Set it to a positive float to include it with that relative weight.
    Weights are normalised to sum to 1.0 before being sent to the LLM.
    """
    skills: Optional[float] = Field(None, ge=0, description="Weight for skills section")
    experience: Optional[float] = Field(None, ge=0, description="Weight for experience section")
    education: Optional[float] = Field(None, ge=0, description="Weight for education section")
    projects: Optional[float] = Field(None, ge=0, description="Weight for projects section")
    summary: Optional[float] = Field(None, ge=0, description="Weight for summary section")
    languages: Optional[float] = Field(None, ge=0, description="Weight for languages section")
    achievements: Optional[float] = Field(None, ge=0, description="Weight for achievements section")
    certifications: Optional[float] = Field(None, ge=0, description="Weight for certifications section")
    publications: Optional[float] = Field(None, ge=0, description="Weight for publications section")
    other: Optional[float] = Field(None, ge=0, description="Weight for other section")

    def to_weights_dict(self) -> Optional[Dict[str, float]]:
        """
        Return only the explicitly set (non-None) fields as a dict.
        Returns None if the caller provided no section_weights body at all,
        which tells the service to fall back to DEFAULT_SECTION_WEIGHTS.
        """
        result = {k: v for k, v in self.model_dump().items() if v is not None}
        return result if result else None


class ScoreRequest(BaseModel):
    job_description_id: uuid.UUID = Field(
        ..., description="UUID of the JobDescription to score against"
    )
    score_threshold: float = Field(
        50.0,
        ge=0,
        le=100,
        description="Threshold score out of 100 to pass (0-100)."
    )
    candidate_profile_ids: Optional[List[uuid.UUID]] = Field(
        None,
        description="Explicit list of CandidateProfile UUIDs to score. "
                    "When omitted, all profiles in the database are scored.",
    )
    section_weights: Optional[SectionWeights] = Field(
        None,
        description="Per-section weights. Only explicitly provided fields are used; "
                    "all others default to 0. Omit entirely to use default weights.",
    )
    batch_size: int = Field(
        10,
        ge=1,
        le=50,
        description="Number of candidates sent to the LLM in a single call (1–50). "
                    "Smaller = more accurate context; larger = fewer API calls.",
    )


class ComponentScore(BaseModel):
    criterionKey: str
    criterionType: Optional[str] = None
    evaluationMode: Optional[str] = None
    requirementText: Optional[str] = None
    weight: float
    score: float
    weightedScore: float
    evidenceSummary: str


class CandidateScore(BaseModel):
    candidateId: str
    totalScore: float
    passedThreshold: bool
    rationale: str
    componentScores: List[ComponentScore]


class ScoreResponse(BaseModel):
    match_run_id: str
    job_description_id: str
    total_candidates: int
    total_passed_candidates: int
    batches: int
    scores: List[dict]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/",
    response_model=ScoreResponse,
    summary="Score candidates against a job description",
    description=(
        "Scores one or more candidate profiles against a job description using the LLM. "
        "Supply `section_weights` to control which CV sections matter and by how much — "
        "only the fields you set are included; all others default to 0. "
        "Omit `section_weights` entirely to use the system defaults "
        "(skills 35%, experience 35%, projects 15%, education 10%, summary 5%). "
        "Use `batch_size` to control how many candidates are sent per LLM call."
    ),
)
def score_candidates_endpoint(
    body: ScoreRequest,
    db=Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    # Resolve section_weights: only include keys that were explicitly set
    weights_dict: Optional[Dict[str, float]] = None
    if body.section_weights is not None:
        weights_dict = body.section_weights.to_weights_dict()
        if weights_dict is not None and sum(weights_dict.values()) == 0:
            raise HTTPException(
                status_code=422,
                detail="All provided section_weights are 0 or negative. "
                       "At least one section must have a positive weight.",
            )

    owned = db.execute(
        select(JobDescription)
        .join(Job, Job.id == JobDescription.job_id)
        .where(JobDescription.id == body.job_description_id, Job.owner_user_id == current_user.id)
    ).scalar_one_or_none()
    if owned is None:
        raise HTTPException(status_code=404, detail=f"Job description {body.job_description_id} not found")
    try:
        result = score_candidates(
            db=db,
            job_description_id=body.job_description_id,
            initiated_by_user_id=current_user.id,
            score_threshold=Decimal(str(body.score_threshold)),
            candidate_profile_ids=body.candidate_profile_ids,
            section_weights=weights_dict,
            batch_size=body.batch_size,
        )
    except ScoringProviderLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if "not found" in detail.lower() else 422
        raise HTTPException(status_code=status_code, detail=detail) from exc

    return result
