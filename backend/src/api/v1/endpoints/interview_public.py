from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.models.deps import get_db
from src.schemas.interview_public import (
    PublicInterviewCompleteRequest,
    PublicInterviewCompleteResponse,
    PublicInterviewEventsRequest,
    PublicInterviewEventsResponse,
    PublicInterviewStartRequest,
    PublicInterviewStartResponse,
)
from src.services.interview_session_service import (
    complete_public_interview_session,
    ingest_public_interview_events,
    start_public_interview_session,
)


router = APIRouter()


@router.post("/interview/{token}/start", response_model=PublicInterviewStartResponse)
def start_interview_session(
    token: str,
    body: PublicInterviewStartRequest,
    db: Session = Depends(get_db),
):
    return start_public_interview_session(db, token=token, body=body)


@router.post("/interview/{token}/events", response_model=PublicInterviewEventsResponse, status_code=202)
def ingest_interview_events(
    token: str,
    body: PublicInterviewEventsRequest,
    db: Session = Depends(get_db),
):
    return ingest_public_interview_events(db, token=token, body=body)


@router.post("/interview/{token}/complete", response_model=PublicInterviewCompleteResponse)
def complete_interview_session(
    token: str,
    body: PublicInterviewCompleteRequest,
    db: Session = Depends(get_db),
):
    return complete_public_interview_session(db, token=token, body=body)
