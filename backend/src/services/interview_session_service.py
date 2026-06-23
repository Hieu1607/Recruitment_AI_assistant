from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import cast

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from src.models.interview_invitation import InterviewInvitation
from src.models.interview_session import InterviewSession, InterviewTranscriptTurn
from src.models.interview_template import InterviewTemplate
from src.schemas.interview_public import (
    PublicInterviewAvailabilityPayload,
    PublicInterviewCompleteRequest,
    PublicInterviewCompleteResponse,
    PublicInterviewEventsRequest,
    PublicInterviewEventsResponse,
    PublicInterviewInvitationPayload,
    PublicInterviewSessionPayload,
    PublicInterviewStartRequest,
    PublicInterviewStartResponse,
    PublicInterviewStatusResponse,
    PublicInterviewTemplatePayload,
    PublicInterviewTTSRequest,
)
from src.services.notification_service import create_notification
from src.services.tts_service import synthesize_speech
from src.services.voice_provider import UnsupportedVoiceProviderError, VoiceProvider, get_voice_provider


STARTABLE_INVITATION_STATUSES = {"pending", "in_progress"}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InvitationStartAvailability:
    can_start: bool
    reason: str
    detail: str | None = None
    status_code: int | None = None


def serialize_public_interview_invitation(invitation: InterviewInvitation) -> PublicInterviewInvitationPayload:
    candidate_full_name = invitation.candidate_profile.full_name if invitation.candidate_profile is not None else None
    return PublicInterviewInvitationPayload(
        id=str(invitation.id),
        public_token=invitation.public_token,
        status=invitation.status,
        expires_at=invitation.expires_at,
        max_attempts=invitation.max_attempts,
        attempt_count=invitation.attempt_count,
        candidate_full_name=candidate_full_name,
        completed_at=invitation.completed_at,
    )


def serialize_public_interview_session(session: InterviewSession) -> PublicInterviewSessionPayload:
    return PublicInterviewSessionPayload(
        id=str(session.id),
        provider=session.provider,
        provider_session_id=session.provider_session_id,
        status=session.status,
        started_at=session.started_at,
        completed_at=session.completed_at,
    )


def serialize_public_interview_template(template: InterviewTemplate) -> PublicInterviewTemplatePayload:
    return PublicInterviewTemplatePayload(
        id=str(template.id),
        name=template.name,
        language_code=template.language_code,
        intro_script=template.intro_script,
        closing_script=template.closing_script,
        question_payload=cast(dict, template.question_payload or {}),
    )


def get_public_interview_status(
    db: Session,
    *,
    token: str,
) -> PublicInterviewStatusResponse:
    invitation = _get_public_interview_invitation(db, token)
    template = invitation.interview_template
    assert template is not None
    availability = _evaluate_invitation_start_availability(invitation, _get_active_session(invitation))
    return PublicInterviewStatusResponse(
        invitation=serialize_public_interview_invitation(invitation),
        template=serialize_public_interview_template(template),
        availability=PublicInterviewAvailabilityPayload(
            can_start=availability.can_start,
            reason=availability.reason,
            detail=availability.detail,
        ),
    )


def start_public_interview_session(
    db: Session,
    *,
    token: str,
    body: PublicInterviewStartRequest,
) -> PublicInterviewStartResponse:
    invitation = _get_public_interview_invitation(db, token)
    _ensure_invitation_can_start(invitation)

    provider = _resolve_voice_provider(body.provider)
    now = _utc_now()
    session_record = InterviewSession(
        interview_invitation_id=invitation.id,
        provider=provider.name,
        provider_session_id=body.provider_session_id,
        status="in_progress",
        started_at=now,
        device_metadata=body.device_metadata,
        browser_metadata=body.browser_metadata,
        connection_metadata=body.connection_metadata,
    )
    db.add(session_record)

    invitation.status = "in_progress"
    invitation.attempt_count += 1
    if invitation.opened_at is None:
        invitation.opened_at = now

    db.commit()
    return _build_start_response(db, invitation.id, session_record.id)


def ingest_public_interview_events(
    db: Session,
    *,
    token: str,
    body: PublicInterviewEventsRequest,
) -> PublicInterviewEventsResponse:
    invitation = _get_public_interview_invitation(db, token)
    _ensure_invitation_is_open(invitation)
    session_record = _get_active_session(invitation)
    if session_record is None:
        raise HTTPException(status_code=409, detail="Interview session has not been started")

    provider = _resolve_session_provider(body.provider, session_record)
    normalized_events = provider.normalize_events([event.model_dump() for event in body.events])
    _append_transcript_turns_with_retry(db, session_record.id, normalized_events)
    return PublicInterviewEventsResponse(accepted=True, stored_turns=len(normalized_events))


def complete_public_interview_session(
    db: Session,
    *,
    token: str,
    body: PublicInterviewCompleteRequest,
) -> PublicInterviewCompleteResponse:
    invitation = _get_public_interview_invitation(db, token)
    _ensure_invitation_is_open(invitation)
    session_record = _get_active_session(invitation)
    if session_record is None:
        raise HTTPException(status_code=409, detail="Interview session has not been started")

    provider = _resolve_session_provider(body.provider, session_record)
    now = _utc_now()
    session_record.provider = provider.name
    session_record.status = "completed"
    session_record.completed_at = now
    invitation.status = "completed"
    invitation.completed_at = now
    candidate_name = invitation.candidate_profile.full_name or "Candidate"
    create_notification(
        db=db,
        user_id=invitation.job.owner_user_id,
        notification_type="interview_completed",
        title="Voice interview completed",
        body=f"{candidate_name} completed the voice interview for {invitation.job.title}.",
        target_url=f"/interviews/reports/{session_record.id}",
        metadata={
            "job_id": str(invitation.job_id),
            "candidate_profile_id": str(invitation.candidate_profile_id),
            "interview_session_id": str(session_record.id),
            "interview_invitation_id": str(invitation.id),
        },
        commit=False,
    )

    db.commit()
    try:
        enqueue_interview_report_generation(db, session_record.id)
    except Exception as exc:
        from src.services.interview_report_service import mark_interview_report_failure_in_db

        mark_interview_report_failure_in_db(
            db,
            interview_session_id=session_record.id,
            stage="enqueue",
            message=str(exc),
            retryable=True,
        )
        logger.exception("Failed to enqueue interview report generation for %s", session_record.id)
    refreshed_invitation = _get_public_interview_invitation(db, token)
    refreshed_session = db.get(InterviewSession, session_record.id)
    assert refreshed_session is not None
    return PublicInterviewCompleteResponse(
        invitation=serialize_public_interview_invitation(refreshed_invitation),
        session=serialize_public_interview_session(refreshed_session),
    )


def synthesize_public_interview_prompt(
    db: Session,
    *,
    token: str,
    body: PublicInterviewTTSRequest,
) -> tuple[bytes, str]:
    invitation = _get_public_interview_invitation(db, token)
    template = invitation.interview_template
    language_code = template.language_code if template is not None else "en-US"
    return synthesize_speech(body.text, language_code=language_code), language_code


def _build_start_response(db: Session, invitation_id, session_id) -> PublicInterviewStartResponse:
    invitation = (
        db.execute(
            select(InterviewInvitation)
            .options(
                joinedload(InterviewInvitation.candidate_profile),
                joinedload(InterviewInvitation.interview_template),
            )
            .where(InterviewInvitation.id == invitation_id)
        )
        .scalars()
        .one()
    )
    session_record = db.get(InterviewSession, session_id)
    assert session_record is not None
    template = invitation.interview_template
    assert template is not None
    return PublicInterviewStartResponse(
        invitation=serialize_public_interview_invitation(invitation),
        session=serialize_public_interview_session(session_record),
        template=serialize_public_interview_template(template),
    )


def _get_public_interview_invitation(db: Session, token: str) -> InterviewInvitation:
    invitation = (
        db.execute(
            select(InterviewInvitation)
            .options(
                joinedload(InterviewInvitation.candidate_profile),
                joinedload(InterviewInvitation.interview_template),
                joinedload(InterviewInvitation.sessions),
            )
            .where(InterviewInvitation.public_token == token)
        )
        .scalars()
        .first()
    )
    if invitation is None:
        raise HTTPException(status_code=404, detail="Interview invitation not found")
    return invitation


def _resolve_voice_provider(provider_name: str | None) -> VoiceProvider:
    try:
        return get_voice_provider(provider_name)
    except UnsupportedVoiceProviderError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _resolve_session_provider(provider_name: str | None, session_record: InterviewSession) -> VoiceProvider:
    if provider_name is None:
        return _resolve_voice_provider(session_record.provider)

    provider = _resolve_voice_provider(provider_name)
    if session_record.provider is not None and provider.name != session_record.provider:
        raise HTTPException(status_code=409, detail="Interview session provider does not match")
    return provider


def _ensure_invitation_can_start(invitation: InterviewInvitation) -> None:
    availability = _evaluate_invitation_start_availability(invitation, _get_active_session(invitation))
    if availability.can_start:
        return
    assert availability.status_code is not None
    raise HTTPException(status_code=availability.status_code, detail=availability.detail)


def _ensure_invitation_is_open(invitation: InterviewInvitation) -> None:
    _ensure_invitation_not_expired(invitation)
    if invitation.completed_at is not None or invitation.status == "completed":
        raise HTTPException(status_code=409, detail="Interview invitation is already completed")
    if invitation.status != "in_progress":
        raise HTTPException(status_code=410, detail="Interview invitation is not active")


def _ensure_invitation_not_expired(invitation: InterviewInvitation) -> None:
    if invitation.expires_at is None:
        return
    expires_at = invitation.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < _utc_now():
        raise HTTPException(status_code=410, detail="Interview invitation has expired")


def _get_active_session(invitation: InterviewInvitation) -> InterviewSession | None:
    active_sessions = [
        session_record
        for session_record in invitation.sessions
        if session_record.status in {"created", "in_progress"} and session_record.completed_at is None
    ]
    if not active_sessions:
        return None
    return max(active_sessions, key=lambda session_record: session_record.created_at or datetime.min.replace(tzinfo=timezone.utc))


def _evaluate_invitation_start_availability(
    invitation: InterviewInvitation,
    active_session: InterviewSession | None,
) -> InvitationStartAvailability:
    if _is_invitation_expired(invitation):
        return InvitationStartAvailability(
            can_start=False,
            reason="expired",
            detail="Interview invitation has expired",
            status_code=410,
        )
    if invitation.completed_at is not None or invitation.status == "completed":
        return InvitationStartAvailability(
            can_start=False,
            reason="completed",
            detail="Interview invitation is already completed",
            status_code=409,
        )
    if active_session is not None:
        return InvitationStartAvailability(
            can_start=False,
            reason="session_in_progress",
            detail="Interview session is already in progress",
            status_code=409,
        )
    if invitation.status not in STARTABLE_INVITATION_STATUSES:
        return InvitationStartAvailability(
            can_start=False,
            reason="inactive",
            detail="Interview invitation is not active",
            status_code=410,
        )
    if invitation.attempt_count >= invitation.max_attempts:
        return InvitationStartAvailability(
            can_start=False,
            reason="attempt_limit_reached",
            detail="Interview attempt limit has been reached",
            status_code=409,
        )
    return InvitationStartAvailability(
        can_start=True,
        reason="ready",
    )


def _is_invitation_expired(invitation: InterviewInvitation) -> bool:
    if invitation.expires_at is None:
        return False
    expires_at = invitation.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at < _utc_now()


def _append_transcript_turns_with_retry(
    db: Session,
    interview_session_id,
    normalized_events,
) -> None:
    for attempt in range(2):
        try:
            _lock_interview_session_for_update(db, interview_session_id)
            next_turn_index = _get_next_transcript_turn_index(db, interview_session_id)
            for normalized_event in normalized_events:
                transcript_payload = dict(normalized_event.payload or {})
                if normalized_event.question_key is not None:
                    transcript_payload["question_key"] = normalized_event.question_key
                db.add(
                    InterviewTranscriptTurn(
                        interview_session_id=interview_session_id,
                        speaker_role=normalized_event.speaker_role,
                        turn_index=next_turn_index + normalized_event.turn_index,
                        transcript_text=normalized_event.transcript_text,
                        time_offset_ms=normalized_event.time_offset_ms,
                        payload=transcript_payload or None,
                    )
                )
            db.commit()
            return
        except IntegrityError as exc:
            db.rollback()
            if attempt == 1 or not _is_turn_index_conflict(exc):
                raise
            logger.warning(
                "Retrying transcript ingest after turn_index conflict for interview session %s",
                interview_session_id,
            )


def _lock_interview_session_for_update(db: Session, interview_session_id) -> None:
    db.execute(
        select(InterviewSession.id)
        .where(InterviewSession.id == interview_session_id)
        .with_for_update()
    ).scalar_one()


def _get_next_transcript_turn_index(db: Session, interview_session_id) -> int:
    max_turn_index = db.execute(
        select(func.max(InterviewTranscriptTurn.turn_index)).where(
            InterviewTranscriptTurn.interview_session_id == interview_session_id
        )
    ).scalar_one()
    return (max_turn_index if max_turn_index is not None else -1) + 1


def _is_turn_index_conflict(exc: IntegrityError) -> bool:
    message = str(exc.orig) if getattr(exc, "orig", None) is not None else str(exc)
    return "uq_interview_transcript_turns_session_turn_index" in message or (
        "interview_transcript_turns.interview_session_id" in message and "turn_index" in message
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def enqueue_interview_report_generation(db: Session, interview_session_id) -> None:
    from src.services.interview_report_service import mark_interview_report_pending_in_db
    from worker.tasks import generate_interview_report

    task_result = generate_interview_report.delay(str(interview_session_id))
    mark_interview_report_pending_in_db(
        db,
        interview_session_id=interview_session_id,
        task_id=getattr(task_result, "id", None),
        retry_count=0,
        state="queued",
    )
