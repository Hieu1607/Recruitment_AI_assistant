from __future__ import annotations

import uuid

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models.interview_invitation import InterviewInvitation
from src.models.interview_template import InterviewTemplate
from src.models.job import Job
from src.models.outreach import OutreachMessage
from src.models.outreach_template import OutreachTemplate
from src.models.query_shortlist import QuerySession, QueryTurn, ShortlistCollection, ShortlistItem


def get_current_user_owned_query_session(
    db: Session,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
) -> QuerySession:
    session = db.execute(
        select(QuerySession).where(
            QuerySession.id == session_id,
            QuerySession.user_id == user_id,
        )
    ).scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session


def get_current_user_owned_query_turn(
    db: Session,
    user_id: uuid.UUID,
    turn_id: uuid.UUID,
) -> QueryTurn:
    turn = db.execute(
        select(QueryTurn)
        .join(QuerySession, QuerySession.id == QueryTurn.query_session_id)
        .where(
            QueryTurn.id == turn_id,
            QuerySession.user_id == user_id,
        )
    ).scalar_one_or_none()
    if turn is None:
        raise HTTPException(status_code=404, detail=f"Turn '{turn_id}' not found")
    return turn


def get_current_user_owned_shortlist_collection(
    db: Session,
    user_id: uuid.UUID,
    collection_id: uuid.UUID,
) -> ShortlistCollection:
    collection = db.execute(
        select(ShortlistCollection).where(
            ShortlistCollection.id == collection_id,
            ShortlistCollection.created_by_user_id == user_id,
        )
    ).scalar_one_or_none()
    if collection is None:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_id}' not found",
        )
    return collection


def get_current_user_owned_shortlist_item(
    db: Session,
    user_id: uuid.UUID,
    collection_id: uuid.UUID,
    candidate_id: uuid.UUID,
) -> ShortlistItem:
    item = db.execute(
        select(ShortlistItem)
        .join(
            ShortlistCollection,
            ShortlistCollection.id == ShortlistItem.shortlist_collection_id,
        )
        .where(
            ShortlistItem.shortlist_collection_id == collection_id,
            ShortlistItem.candidate_profile_id == candidate_id,
            ShortlistCollection.created_by_user_id == user_id,
        )
    ).scalar_one_or_none()
    if item is None:
        raise HTTPException(
            status_code=404,
            detail=f"Candidate '{candidate_id}' not found in collection '{collection_id}'",
        )
    return item


def get_current_user_owned_outreach_template(
    db: Session,
    user_id: uuid.UUID,
    template_id: uuid.UUID,
) -> OutreachTemplate:
    template = db.execute(
        select(OutreachTemplate).where(
            OutreachTemplate.id == template_id,
            OutreachTemplate.created_by_user_id == user_id,
        )
    ).scalar_one_or_none()
    if template is None:
        raise HTTPException(
            status_code=404,
            detail=f"OutreachTemplate '{template_id}' not found",
        )
    return template


def get_current_user_owned_active_interview_template(
    db: Session,
    user_id: uuid.UUID,
    job_id: uuid.UUID,
    template_id: uuid.UUID,
) -> InterviewTemplate:
    template = db.execute(
        select(InterviewTemplate)
        .join(Job, Job.id == InterviewTemplate.job_id)
        .where(
            InterviewTemplate.id == template_id,
            InterviewTemplate.job_id == job_id,
            InterviewTemplate.status == "active",
            Job.owner_user_id == user_id,
        )
    ).scalar_one_or_none()
    if template is None:
        raise HTTPException(status_code=404, detail="Active interview template not found")
    return template


def get_current_user_latest_outreach(
    db: Session,
    user_id: uuid.UUID,
    candidate_id: uuid.UUID,
) -> OutreachMessage | None:
    return (
        db.query(OutreachMessage)
        .filter(
            OutreachMessage.candidate_profile_id == candidate_id,
            OutreachMessage.created_by_user_id == user_id,
        )
        .order_by(OutreachMessage.created_at.desc())
        .first()
    )


def get_current_user_latest_interview(
    db: Session,
    user_id: uuid.UUID,
    candidate_id: uuid.UUID,
    job_id: uuid.UUID | None = None,
) -> InterviewInvitation | None:
    query = db.query(InterviewInvitation).filter(
        InterviewInvitation.candidate_profile_id == candidate_id,
        InterviewInvitation.sent_by_user_id == user_id,
    )
    if job_id is not None:
        query = query.filter(InterviewInvitation.job_id == job_id)
    return query.order_by(InterviewInvitation.created_at.desc()).first()
