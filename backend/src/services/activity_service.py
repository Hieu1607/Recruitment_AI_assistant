from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload, selectinload

from src.models.candidate_profile import CandidateProfile
from src.models.enums import MatchRunStatus, SentStatus, UploadStatus
from src.models.interview_invitation import InterviewInvitation
from src.models.job import Job
from src.models.job_matching import JobDescription, MatchRun
from src.models.outreach import OutreachMessage
from src.models.query_shortlist import ShortlistItem
from src.models.resume_document import ResumeDocument


@dataclass(slots=True)
class ActivityEvent:
    id: str
    kind: str
    timestamp: datetime
    subject_name: str | None
    context_name: str | None
    status: str | None
    target_url: str | None
    metadata: dict[str, Any]


def _event_to_dict(event: ActivityEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "kind": event.kind,
        "timestamp": event.timestamp,
        "subject_name": event.subject_name,
        "context_name": event.context_name,
        "status": event.status,
        "target_url": event.target_url,
        "metadata": event.metadata,
    }


def _activity_query_limit(limit: int) -> int:
    return max(limit * 3, 20)


def _resume_events(*, db: Session, user_id: uuid.UUID, job_id: uuid.UUID | None, limit: int) -> list[ActivityEvent]:
    query = (
        db.query(ResumeDocument)
        .join(Job, Job.id == ResumeDocument.job_id)
        .options(
            joinedload(ResumeDocument.candidate_profile),
            selectinload(ResumeDocument.extraction_traces),
        )
        .filter(Job.owner_user_id == user_id)
        .order_by(ResumeDocument.uploaded_at.desc())
    )
    if job_id is not None:
        query = query.filter(ResumeDocument.job_id == job_id)

    events: list[ActivityEvent] = []
    for resume in query.limit(limit).all():
        subject_name = resume.original_file_name
        target_url = f"/candidates/{resume.id}"

        if resume.upload_status == UploadStatus.PROCESSED and resume.processed_at is not None:
            events.append(
                ActivityEvent(
                    id=f"resume-processed-{resume.id}",
                    kind="resume_processed",
                    timestamp=resume.processed_at,
                    subject_name=subject_name,
                    context_name=resume.candidate_profile.full_name if resume.candidate_profile else None,
                    status=UploadStatus.PROCESSED.value,
                    target_url=target_url,
                    metadata={
                        "resume_id": str(resume.id),
                        "candidate_profile_id": (
                            str(resume.candidate_profile.id) if resume.candidate_profile else None
                        ),
                    },
                )
            )
            continue

        if resume.upload_status == UploadStatus.FAILED:
            failed_trace = max(
                (trace for trace in resume.extraction_traces if trace.status == "failed"),
                key=lambda trace: trace.created_at,
                default=None,
            )
            timestamp = failed_trace.created_at if failed_trace is not None else resume.uploaded_at
            if timestamp is None:
                continue
            events.append(
                ActivityEvent(
                    id=f"resume-failed-{resume.id}",
                    kind="resume_failed",
                    timestamp=timestamp,
                    subject_name=subject_name,
                    context_name=resume.candidate_profile.full_name if resume.candidate_profile else None,
                    status=UploadStatus.FAILED.value,
                    target_url=target_url,
                    metadata={
                        "resume_id": str(resume.id),
                        "candidate_profile_id": (
                            str(resume.candidate_profile.id) if resume.candidate_profile else None
                        ),
                        "message": failed_trace.message if failed_trace is not None else None,
                    },
                )
            )
            continue

        if resume.uploaded_at is None:
            continue
        if resume.candidate_profile is not None:
            continue
        events.append(
            ActivityEvent(
                id=f"resume-uploaded-{resume.id}",
                kind="resume_uploaded",
                timestamp=resume.uploaded_at,
                subject_name=subject_name,
                context_name=resume.candidate_profile.full_name if resume.candidate_profile else None,
                status=str(resume.upload_status.value if hasattr(resume.upload_status, "value") else resume.upload_status),
                target_url=target_url,
                metadata={
                    "resume_id": str(resume.id),
                    "candidate_profile_id": (
                        str(resume.candidate_profile.id) if resume.candidate_profile else None
                    ),
                },
            )
        )
    return events


def _shortlist_events(*, db: Session, user_id: uuid.UUID, job_id: uuid.UUID | None, limit: int) -> list[ActivityEvent]:
    query = (
        db.query(ShortlistItem)
        .join(ShortlistItem.shortlist_collection)
        .join(ShortlistItem.candidate_profile)
        .join(CandidateProfile.resume_document)
        .join(ResumeDocument.job)
        .options(
            joinedload(ShortlistItem.shortlist_collection),
            joinedload(ShortlistItem.candidate_profile),
        )
        .filter(
            Job.owner_user_id == user_id,
            ShortlistItem.shortlist_collection.has(created_by_user_id=user_id),
        )
        .order_by(ShortlistItem.added_at.desc())
    )
    if job_id is not None:
        query = query.filter(ResumeDocument.job_id == job_id)

    events: list[ActivityEvent] = []
    for item in query.limit(limit).all():
        candidate = item.candidate_profile
        collection = item.shortlist_collection
        events.append(
            ActivityEvent(
                id=f"shortlist-added-{item.id}",
                kind="shortlist_added",
                timestamp=item.added_at,
                subject_name=candidate.full_name if candidate else None,
                context_name=collection.name if collection else None,
                status=None,
                target_url=f"/shortlists/collections/{collection.id}" if collection else "/shortlists",
                metadata={
                    "candidate_profile_id": str(candidate.id) if candidate else None,
                    "collection_id": str(collection.id) if collection else None,
                },
            )
        )
    return events


def _outreach_events(*, db: Session, user_id: uuid.UUID, job_id: uuid.UUID | None, limit: int) -> list[ActivityEvent]:
    query = (
        db.query(OutreachMessage)
        .join(OutreachMessage.candidate_profile)
        .join(CandidateProfile.resume_document)
        .join(ResumeDocument.job)
        .options(joinedload(OutreachMessage.candidate_profile))
        .filter(
            OutreachMessage.created_by_user_id == user_id,
            OutreachMessage.sent_status.in_([SentStatus.SENT, SentStatus.FAILED]),
            Job.owner_user_id == user_id,
        )
        .order_by(func.coalesce(OutreachMessage.sent_at, OutreachMessage.created_at).desc())
    )
    if job_id is not None:
        query = query.filter(ResumeDocument.job_id == job_id)

    events: list[ActivityEvent] = []
    for message in query.limit(limit).all():
        timestamp = message.sent_at or message.created_at
        if timestamp is None:
            continue
        kind = "outreach_sent" if message.sent_status == SentStatus.SENT else "outreach_failed"
        events.append(
            ActivityEvent(
                id=f"{kind}-{message.id}",
                kind=kind,
                timestamp=timestamp,
                subject_name=message.candidate_profile.full_name if message.candidate_profile else None,
                context_name=message.subject,
                status=message.sent_status.value,
                target_url="/outreach",
                metadata={
                    "message_id": str(message.id),
                    "candidate_profile_id": (
                        str(message.candidate_profile.id) if message.candidate_profile else None
                    ),
                },
            )
        )
    return events


def _interview_events(*, db: Session, user_id: uuid.UUID, job_id: uuid.UUID | None, limit: int) -> list[ActivityEvent]:
    query = (
        db.query(InterviewInvitation)
        .join(InterviewInvitation.job)
        .options(
            joinedload(InterviewInvitation.candidate_profile),
            joinedload(InterviewInvitation.interview_template),
        )
        .filter(Job.owner_user_id == user_id)
        .order_by(InterviewInvitation.updated_at.desc(), InterviewInvitation.created_at.desc())
    )
    if job_id is not None:
        query = query.filter(InterviewInvitation.job_id == job_id)

    events: list[ActivityEvent] = []
    for invitation in query.limit(limit).all():
        kind: str
        timestamp: datetime | None
        status = invitation.status
        if invitation.completed_at is not None:
            kind = "interview_completed"
            timestamp = invitation.completed_at
        elif invitation.cancelled_at is not None:
            kind = "interview_cancelled"
            timestamp = invitation.cancelled_at
        elif invitation.sent_at is not None:
            kind = "interview_invitation_sent"
            timestamp = invitation.sent_at
        else:
            kind = "interview_link_created"
            timestamp = invitation.created_at
        if timestamp is None:
            continue
        events.append(
            ActivityEvent(
                id=f"{kind}-{invitation.id}",
                kind=kind,
                timestamp=timestamp,
                subject_name=(
                    invitation.candidate_profile.full_name if invitation.candidate_profile else None
                ),
                context_name=(
                    invitation.interview_template.name if invitation.interview_template else None
                ),
                status=status,
                target_url="/interviews",
                metadata={
                    "invitation_id": str(invitation.id),
                    "candidate_profile_id": (
                        str(invitation.candidate_profile.id) if invitation.candidate_profile else None
                    ),
                    "interview_template_id": str(invitation.interview_template_id),
                },
            )
        )
    return events


def _scoring_events(*, db: Session, user_id: uuid.UUID, job_id: uuid.UUID | None, limit: int) -> list[ActivityEvent]:
    query = (
        db.query(MatchRun)
        .join(MatchRun.job_description)
        .join(JobDescription.job)
        .options(joinedload(MatchRun.job_description).joinedload(JobDescription.job))
        .filter(
            MatchRun.initiated_by_user_id == user_id,
            MatchRun.run_status == MatchRunStatus.COMPLETED,
            Job.owner_user_id == user_id,
        )
        .order_by(func.coalesce(MatchRun.completed_at, MatchRun.created_at).desc())
    )
    if job_id is not None:
        query = query.filter(JobDescription.job_id == job_id)

    events: list[ActivityEvent] = []
    for run in query.limit(limit).all():
        timestamp = run.completed_at or run.created_at
        if timestamp is None:
            continue
        job_description = run.job_description
        job = job_description.job if job_description else None
        events.append(
            ActivityEvent(
                id=f"scoring-completed-{run.id}",
                kind="scoring_completed",
                timestamp=timestamp,
                subject_name=job.title if job else None,
                context_name=job_description.title if job_description else None,
                status=run.run_status.value if hasattr(run.run_status, "value") else str(run.run_status),
                target_url=f"/scoring/{run.id}",
                metadata={
                    "match_run_id": str(run.id),
                    "job_description_id": str(job_description.id) if job_description else None,
                    "job_id": str(job.id) if job else None,
                },
            )
        )
    return events


def list_recent_activities(
    *,
    db: Session,
    user_id: uuid.UUID,
    job_id: uuid.UUID | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    fetch_limit = _activity_query_limit(limit)
    events: list[ActivityEvent] = []
    events.extend(_resume_events(db=db, user_id=user_id, job_id=job_id, limit=fetch_limit))
    events.extend(_shortlist_events(db=db, user_id=user_id, job_id=job_id, limit=fetch_limit))
    events.extend(_outreach_events(db=db, user_id=user_id, job_id=job_id, limit=fetch_limit))
    events.extend(_interview_events(db=db, user_id=user_id, job_id=job_id, limit=fetch_limit))
    events.extend(_scoring_events(db=db, user_id=user_id, job_id=job_id, limit=fetch_limit))
    events.sort(key=lambda event: event.timestamp, reverse=True)
    return [_event_to_dict(event) for event in events[:limit]]
