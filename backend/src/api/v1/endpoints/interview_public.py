from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from src.models.deps import get_db
from src.schemas.interview_public import (
    PublicInterviewCompleteRequest,
    PublicInterviewCompleteResponse,
    PublicInterviewEventsRequest,
    PublicInterviewEventsResponse,
    PublicInterviewStartRequest,
    PublicInterviewStartResponse,
    PublicInterviewStatusResponse,
    PublicInterviewTTSRequest,
)
from src.services.interview_session_service import (
    complete_public_interview_session,
    get_public_interview_status,
    ingest_public_interview_events,
    start_public_interview_session,
    synthesize_public_interview_prompt,
)
from src.services.tts_service import TTSProviderError


logger = logging.getLogger(__name__)


router = APIRouter()


@router.get("/interview/{token}", response_model=PublicInterviewStatusResponse)
def get_interview_status(
    token: str,
    db: Session = Depends(get_db),
):
    return get_public_interview_status(db, token=token)


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


@router.post("/interview/{token}/tts")
def synthesize_interview_tts(
    token: str,
    body: PublicInterviewTTSRequest,
    db: Session = Depends(get_db),
):
    try:
        audio, _language_code = synthesize_public_interview_prompt(db, token=token, body=body)
    except TTSProviderError as exc:
        logger.error("TTS synthesis failed for interview token=%s: %s", token, exc)
        raise HTTPException(
            status_code=503,
            detail="Speech synthesis is temporarily unavailable. Please try again.",
        ) from exc
    return Response(content=audio, media_type="audio/mpeg")
