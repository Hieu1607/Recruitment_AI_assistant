"""CRUD endpoints for QuerySession, QueryTurn, ShortlistCollection, ShortlistItem.

Route map
---------
Sessions
  POST   /shortlist/sessions/                          create session
  GET    /shortlist/sessions/                          list sessions for current user
  GET    /shortlist/sessions/{session_id}              get session + turn count
  PATCH  /shortlist/sessions/{session_id}              update title
  DELETE /shortlist/sessions/{session_id}              delete session (cascades turns)

Turns
  POST   /shortlist/sessions/{session_id}/turns        create turn
  GET    /shortlist/sessions/{session_id}/turns        list turns in session
  GET    /shortlist/turns/{turn_id}                    get single turn
  DELETE /shortlist/turns/{turn_id}                    delete turn

Collections
  POST   /shortlist/collections/                       create collection
  GET    /shortlist/collections/                       list collections for current user
  GET    /shortlist/collections/{collection_id}        get collection + items
  PATCH  /shortlist/collections/{collection_id}        rename collection
  DELETE /shortlist/collections/{collection_id}        delete collection (cascades items)

Items
  POST   /shortlist/collections/{collection_id}/items  add candidate to collection
  GET    /shortlist/collections/{collection_id}/items  list items in collection
  DELETE /shortlist/collections/{collection_id}/items/{candidate_id}  remove item
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload
from src.models.candidate_profile import CandidateProfile
from src.models.deps import get_current_user, get_db
from src.models.enums import ContentSource, SentStatus
from src.models.interview_invitation import InterviewInvitation
from src.models.interview_template import InterviewTemplate
from src.models.job import Job
from src.models.oauth_identity import GMAIL_SEND_SCOPE, OAuthIdentity
from src.models.outreach import OutreachMessage
from src.models.outreach_template import OutreachTemplate
from src.models.query_shortlist import (
    QuerySession,
    QueryTurn,
    ShortlistCollection,
    ShortlistItem,
)
from src.models.resume_document import ResumeDocument
from src.models.user_account import UserAccount
from src.services.interview_template_service import (
    get_job_scoped_interview_question_set,
    materialize_question_set_template,
)
from src.services.outreach_service import (
    build_render_variables,
    normalize_rich_message,
    render_template_string,
)
from src.services.shortlist_scope import (
    get_current_user_latest_interview,
    get_current_user_latest_outreach,
    get_current_user_owned_active_interview_template,
    get_current_user_owned_outreach_template,
    get_current_user_owned_query_session,
    get_current_user_owned_query_turn,
    get_current_user_owned_shortlist_collection,
    get_current_user_owned_shortlist_item,
)

router = APIRouter()


# ---------------------------------------------------------------------------
def _is_unique_violation(exc: Exception, *markers: str) -> bool:
    message = str(exc).lower()
    return any(marker.lower() in message for marker in markers)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

# --- QuerySession ---


class SessionCreateRequest(BaseModel):
    session_title: Optional[str] = Field(None, max_length=255)


class SessionUpdateRequest(BaseModel):
    session_title: Optional[str] = Field(None, max_length=255)


class SessionResponse(BaseModel):
    id: str
    user_id: str
    session_title: Optional[str]
    turn_count: int
    created_at: datetime
    updated_at: datetime


# --- QueryTurn ---


class TurnCreateRequest(BaseModel):
    user_question: str = Field(..., min_length=1)
    answer_text: str = Field(..., min_length=1)
    matched_candidate_ids: Optional[List[str]] = None
    matched_count: Optional[int] = Field(None, ge=0)
    tool_trace_masked: Optional[Dict[str, Any]] = None


class TurnResponse(BaseModel):
    id: str
    query_session_id: str
    user_question: str
    answer_text: str
    matched_candidate_ids: Optional[List[str]]
    matched_count: Optional[int]
    tool_trace_masked: Optional[Dict[str, Any]]
    created_at: datetime


# --- ShortlistCollection ---


class CollectionCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    source_query_turn_id: Optional[uuid.UUID] = None


class CollectionUpdateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)


class CollectionResponse(BaseModel):
    id: str
    name: str
    created_by_user_id: str
    source_query_turn_id: Optional[str]
    item_count: int
    created_at: datetime


class SessionListResponse(BaseModel):
    items: List[SessionResponse]
    total: int


class CollectionListResponse(BaseModel):
    items: List[CollectionResponse]
    total: int


# --- ShortlistItem ---


class ItemAddRequest(BaseModel):
    candidate_profile_id: uuid.UUID


class ItemResponse(BaseModel):
    id: str
    shortlist_collection_id: str
    candidate_profile_id: str
    added_at: datetime


class ItemListResponse(BaseModel):
    items: list["ItemResponse"]
    total: int


# --- Shortlist dispatch ---


class DispatchCollectionResponse(BaseModel):
    id: str
    name: str
    item_count: int


class DispatchJobResponse(BaseModel):
    id: str
    title: str


class DispatchOutreachStatus(BaseModel):
    latest_message_id: str
    status: str
    created_at: datetime
    sent_at: Optional[datetime]


class DispatchInterviewStatus(BaseModel):
    latest_invitation_id: str
    status: str
    interview_template_id: str
    template_name: Optional[str]
    sent_at: Optional[datetime]
    completed_at: Optional[datetime]


class DispatchCandidateResponse(BaseModel):
    candidate_profile_id: str
    full_name: str
    email: Optional[str]
    current_job_title: Optional[str]
    skills_text: Optional[str]
    contact_status: str
    outreach: Optional[DispatchOutreachStatus]
    interview: Optional[DispatchInterviewStatus]
    blockers: list[str]


class DispatchCapabilitiesResponse(BaseModel):
    gmail_connected: bool
    active_interview_templates_count: int


class DispatchSummaryResponse(BaseModel):
    collection: DispatchCollectionResponse
    job: Optional[DispatchJobResponse]
    candidates: list[DispatchCandidateResponse]
    capabilities: DispatchCapabilitiesResponse


class OutreachDraftBatchRequest(BaseModel):
    candidate_profile_ids: list[uuid.UUID] = Field(..., min_length=1)
    subject_template: str = Field(..., min_length=1, max_length=255)
    body_text_template: str | None = Field(None, min_length=1)
    body_html_template: str | None = Field(None, min_length=1)
    content_source: ContentSource = ContentSource.TEMPLATE
    template_id: uuid.UUID | None = None
    force_update: bool = False


class InterviewInvitationBatchRequest(BaseModel):
    candidate_profile_ids: list[uuid.UUID] = Field(..., min_length=1)
    job_id: uuid.UUID
    interview_template_id: uuid.UUID | None = None
    interview_question_set_id: uuid.UUID | None = None
    expires_in_hours: Optional[int] = Field(None, ge=1, le=24 * 30)
    send_email: bool = True

    @model_validator(mode="after")
    def validate_source(self):
        if bool(self.interview_template_id) == bool(self.interview_question_set_id):
            raise ValueError(
                "Provide exactly one of interview_template_id or interview_question_set_id."
            )
        return self


class BatchCandidateResult(BaseModel):
    candidate_profile_id: str
    full_name: Optional[str]
    status: str
    reason: Optional[str] = None
    record_id: Optional[str] = None


class BatchActionResponse(BaseModel):
    created_count: int
    skipped_count: int
    failed_count: int
    results: list[BatchCandidateResult]


# ---------------------------------------------------------------------------
# Serialisers
# ---------------------------------------------------------------------------


def _ser_session(s: QuerySession) -> SessionResponse:
    return SessionResponse(
        id=str(s.id),
        user_id=str(s.user_id),
        session_title=s.session_title,
        turn_count=len(s.turns) if s.turns is not None else 0,
        created_at=s.created_at,
        updated_at=s.updated_at,
    )


def _ser_turn(t: QueryTurn) -> TurnResponse:
    return TurnResponse(
        id=str(t.id),
        query_session_id=str(t.query_session_id),
        user_question=t.user_question,
        answer_text=t.answer_text,
        matched_candidate_ids=t.matched_candidate_ids,
        matched_count=t.matched_count,
        tool_trace_masked=t.tool_trace_masked,
        created_at=t.created_at,
    )


def _ser_collection(c: ShortlistCollection) -> CollectionResponse:
    return CollectionResponse(
        id=str(c.id),
        name=c.name,
        created_by_user_id=str(c.created_by_user_id),
        source_query_turn_id=(
            str(c.source_query_turn_id) if c.source_query_turn_id else None
        ),
        item_count=len(c.items) if c.items is not None else 0,
        created_at=c.created_at,
    )


def _ser_item(i: ShortlistItem) -> ItemResponse:
    return ItemResponse(
        id=str(i.id),
        shortlist_collection_id=str(i.shortlist_collection_id),
        candidate_profile_id=str(i.candidate_profile_id),
        added_at=i.added_at,
    )


def _render_candidate_template(
    template: str,
    *,
    candidate: CandidateProfile,
    job: Job | None,
    company_name: str | None = None,
) -> str:
    return render_template_string(
        template,
        build_render_variables(candidate, job, company_name),
    )


def _load_collection_candidates(
    db: Session,
    collection: ShortlistCollection,
) -> list[CandidateProfile]:
    rows = (
        db.query(ShortlistItem)
        .options(
            joinedload(ShortlistItem.candidate_profile)
            .joinedload(CandidateProfile.resume_document)
            .joinedload(ResumeDocument.job)
        )
        .filter(ShortlistItem.shortlist_collection_id == collection.id)
        .order_by(ShortlistItem.added_at.asc())
        .all()
    )
    return [row.candidate_profile for row in rows if row.candidate_profile is not None]


def _candidate_job(candidate: CandidateProfile | None) -> Job | None:
    if candidate is None or candidate.resume_document is None:
        return None
    return candidate.resume_document.job


def _collection_job(candidates: list[CandidateProfile]) -> Job | None:
    for candidate in candidates:
        job = _candidate_job(candidate)
        if job is not None:
            return job
    return None


def _active_interview_template_count(db, job_id: uuid.UUID | None) -> int:
    if job_id is None:
        return 0
    return (
        db.query(InterviewTemplate)
        .filter(InterviewTemplate.job_id == job_id, InterviewTemplate.status == "active")
        .count()
    )


def _gmail_connected(db, user_id: uuid.UUID) -> bool:
    identity = (
        db.query(OAuthIdentity)
        .filter(
            OAuthIdentity.user_id == user_id,
            OAuthIdentity.provider == "google",
        )
        .first()
    )
    return bool(
        identity
        and identity.refresh_token_encrypted
        and identity.has_scope(GMAIL_SEND_SCOPE)
    )


def _serialize_dispatch_candidate(
    db,
    candidate: CandidateProfile,
    *,
    current_user_id: uuid.UUID,
    job_id: uuid.UUID | None,
    gmail_connected: bool,
    active_template_count: int,
) -> DispatchCandidateResponse:
    outreach = get_current_user_latest_outreach(db, current_user_id, candidate.id)
    interview = get_current_user_latest_interview(db, current_user_id, candidate.id, job_id)
    blockers: list[str] = []
    if not candidate.email:
        blockers.append("missing_email")
    if not gmail_connected:
        blockers.append("gmail_not_connected")
    if active_template_count == 0:
        blockers.append("no_active_template")

    return DispatchCandidateResponse(
        candidate_profile_id=str(candidate.id),
        full_name=candidate.full_name,
        email=candidate.email,
        current_job_title=candidate.current_job_title,
        skills_text=candidate.skills_text,
        contact_status="ready" if candidate.email else "missing_email",
        outreach=(
            DispatchOutreachStatus(
                latest_message_id=str(outreach.id),
                status=outreach.sent_status.value,
                created_at=outreach.created_at,
                sent_at=outreach.sent_at,
            )
            if outreach is not None
            else None
        ),
        interview=(
            DispatchInterviewStatus(
                latest_invitation_id=str(interview.id),
                status=interview.status,
                interview_template_id=str(interview.interview_template_id),
                template_name=interview.interview_template.name
                if interview.interview_template is not None
                else None,
                sent_at=interview.sent_at,
                completed_at=interview.completed_at,
            )
            if interview is not None
            else None
        ),
        blockers=blockers,
    )


def _selected_collection_candidates(
    candidates: list[CandidateProfile], selected_ids: list[uuid.UUID]
) -> tuple[list[CandidateProfile], set[uuid.UUID]]:
    by_id = {candidate.id: candidate for candidate in candidates}
    selected = [by_id[candidate_id] for candidate_id in selected_ids if candidate_id in by_id]
    missing = {candidate_id for candidate_id in selected_ids if candidate_id not in by_id}
    return selected, missing


def _batch_response(results: list[BatchCandidateResult]) -> BatchActionResponse:
    return BatchActionResponse(
        created_count=sum(1 for result in results if result.status == "created"),
        skipped_count=sum(1 for result in results if result.status.startswith("skipped")),
        failed_count=sum(1 for result in results if result.status == "failed"),
        results=results,
    )


# ---------------------------------------------------------------------------
# QuerySession endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/sessions/", response_model=SessionResponse, status_code=201, tags=["sessions"]
)
def create_session(
    body: SessionCreateRequest,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    session = QuerySession(user_id=current_user.id, session_title=body.session_title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return _ser_session(session)


@router.get("/sessions/", response_model=SessionListResponse, tags=["sessions"])
def list_sessions(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    query = db.query(QuerySession).filter(QuerySession.user_id == current_user.id)
    total = query.count()
    rows = (
        query.order_by(QuerySession.updated_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return SessionListResponse(items=[_ser_session(s) for s in rows], total=total)


@router.get("/sessions/{session_id}", response_model=SessionResponse, tags=["sessions"])
def get_session(
    session_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    session = get_current_user_owned_query_session(db, current_user.id, session_id)
    return _ser_session(session)


@router.patch(
    "/sessions/{session_id}", response_model=SessionResponse, tags=["sessions"]
)
def update_session(
    session_id: uuid.UUID,
    body: SessionUpdateRequest,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    session = get_current_user_owned_query_session(db, current_user.id, session_id)
    if body.session_title is not None:
        session.session_title = body.session_title
    db.commit()
    db.refresh(session)
    return _ser_session(session)


@router.delete("/sessions/{session_id}", status_code=204, tags=["sessions"])
def delete_session(
    session_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    session = get_current_user_owned_query_session(db, current_user.id, session_id)
    db.delete(session)
    db.commit()


# ---------------------------------------------------------------------------
# QueryTurn endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/sessions/{session_id}/turns",
    response_model=TurnResponse,
    status_code=201,
    tags=["turns"],
)
def create_turn(
    session_id: uuid.UUID,
    body: TurnCreateRequest,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    session = get_current_user_owned_query_session(db, current_user.id, session_id)
    turn = QueryTurn(
        query_session_id=session.id,
        user_question=body.user_question,
        answer_text=body.answer_text,
        matched_candidate_ids=body.matched_candidate_ids,
        matched_count=body.matched_count,
        tool_trace_masked=body.tool_trace_masked,
    )
    db.add(turn)
    db.commit()
    db.refresh(turn)
    return _ser_turn(turn)


@router.get(
    "/sessions/{session_id}/turns",
    response_model=List[TurnResponse],
    tags=["turns"],
)
def list_turns(
    session_id: uuid.UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    session = get_current_user_owned_query_session(db, current_user.id, session_id)
    rows = (
        db.query(QueryTurn)
        .filter(QueryTurn.query_session_id == session.id)
        .order_by(QueryTurn.created_at.asc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [_ser_turn(t) for t in rows]


@router.get("/turns/{turn_id}", response_model=TurnResponse, tags=["turns"])
def get_turn(
    turn_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    turn = get_current_user_owned_query_turn(db, current_user.id, turn_id)
    return _ser_turn(turn)


@router.delete("/turns/{turn_id}", status_code=204, tags=["turns"])
def delete_turn(
    turn_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    turn = get_current_user_owned_query_turn(db, current_user.id, turn_id)
    db.delete(turn)
    db.commit()


# ---------------------------------------------------------------------------
# ShortlistCollection endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/collections/",
    response_model=CollectionResponse,
    status_code=201,
    tags=["collections"],
)
def create_collection(
    body: CollectionCreateRequest,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    try:
        if body.source_query_turn_id is not None:
            get_current_user_owned_query_turn(
                db,
                current_user.id,
                body.source_query_turn_id,
            )
        collection = ShortlistCollection(
            name=body.name,
            created_by_user_id=current_user.id,
            source_query_turn_id=body.source_query_turn_id,
        )
        db.add(collection)
        db.commit()
        db.refresh(collection)
        return _ser_collection(collection)
    except Exception as exc:
        db.rollback()
        if isinstance(exc, IntegrityError) and _is_unique_violation(
            exc,
            "uq_shortlist_creator_name",
            "unique constraint failed: shortlist_collections.created_by_user_id, shortlist_collections.name",
        ):
            raise HTTPException(
                status_code=409,
                detail=f"Collection named '{body.name}' already exists for this user",
            ) from exc
        raise


@router.get(
    "/collections/",
    response_model=CollectionListResponse,
    tags=["collections"],
)
def list_collections(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    query = db.query(ShortlistCollection).filter(
        ShortlistCollection.created_by_user_id == current_user.id
    )
    total = query.count()
    rows = (
        query.order_by(ShortlistCollection.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return CollectionListResponse(
        items=[_ser_collection(c) for c in rows], total=total
    )


@router.get(
    "/collections/{collection_id}",
    response_model=CollectionResponse,
    tags=["collections"],
)
def get_collection(
    collection_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    collection = get_current_user_owned_shortlist_collection(
        db,
        current_user.id,
        collection_id,
    )
    return _ser_collection(collection)


@router.get(
    "/collections/{collection_id}/dispatch-summary",
    response_model=DispatchSummaryResponse,
    tags=["collections"],
)
def get_dispatch_summary(
    collection_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    collection = get_current_user_owned_shortlist_collection(
        db,
        current_user.id,
        collection_id,
    )
    candidates = _load_collection_candidates(db, collection)
    job = _collection_job(candidates)
    job_id = job.id if job is not None else None
    active_template_count = _active_interview_template_count(db, job_id)
    gmail_connected = _gmail_connected(db, current_user.id)

    return DispatchSummaryResponse(
        collection=DispatchCollectionResponse(
            id=str(collection.id),
            name=collection.name,
            item_count=len(candidates),
        ),
        job=(
            DispatchJobResponse(id=str(job.id), title=job.title)
            if job is not None
            else None
        ),
        candidates=[
            _serialize_dispatch_candidate(
                db,
                candidate,
                current_user_id=current_user.id,
                job_id=job_id,
                gmail_connected=gmail_connected,
                active_template_count=active_template_count,
            )
            for candidate in candidates
        ],
        capabilities=DispatchCapabilitiesResponse(
            gmail_connected=gmail_connected,
            active_interview_templates_count=active_template_count,
        ),
    )


@router.post(
    "/collections/{collection_id}/outreach-drafts",
    response_model=BatchActionResponse,
    status_code=201,
    tags=["collections"],
)
def create_collection_outreach_drafts(
    collection_id: uuid.UUID,
    body: OutreachDraftBatchRequest,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    try:
        collection = get_current_user_owned_shortlist_collection(
            db,
            current_user.id,
            collection_id,
        )
        candidates = _load_collection_candidates(db, collection)
        job = _collection_job(candidates)
        template = None
        if body.template_id is not None:
            template = get_current_user_owned_outreach_template(
                db,
                current_user.id,
                body.template_id,
            )
        source_subject = template.subject_template if template is not None else body.subject_template
        source_text = (
            template.body_text_template
            if template is not None
            else (body.body_text_template or body.body_html_template or "")
        )
        source_html = (
            template.body_html_template
            if template is not None
            else (body.body_html_template or body.body_text_template or "")
        )
        selected, missing_ids = _selected_collection_candidates(
            candidates, body.candidate_profile_ids
        )
        results: list[BatchCandidateResult] = [
            BatchCandidateResult(
                candidate_profile_id=str(candidate_id),
                full_name=None,
                status="skipped_not_in_collection",
                reason="Candidate is not in this shortlist collection.",
            )
            for candidate_id in missing_ids
        ]

        for candidate in selected:
            if not candidate.email:
                results.append(
                    BatchCandidateResult(
                        candidate_profile_id=str(candidate.id),
                        full_name=candidate.full_name,
                        status="skipped_missing_email",
                        reason="Candidate has no email address.",
                    )
                )
                continue

            existing = get_current_user_latest_outreach(
                db,
                current_user.id,
                candidate.id,
            )
            if (
                existing is not None
                and existing.sent_status != SentStatus.FAILED
                and not body.force_update
            ):
                results.append(
                    BatchCandidateResult(
                        candidate_profile_id=str(candidate.id),
                        full_name=candidate.full_name,
                        status="skipped_duplicate",
                        reason="Candidate already has an outreach message.",
                        record_id=str(existing.id),
                    )
                )
                continue

            message = OutreachMessage(
                candidate_profile_id=candidate.id,
                created_by_user_id=collection.created_by_user_id,
                content_source=body.content_source,
                subject=_render_candidate_template(
                    source_subject,
                    candidate=candidate,
                    job=job,
                ).strip(),
                body_text=normalize_rich_message(
                    body_text=_render_candidate_template(
                        source_text,
                        candidate=candidate,
                        job=job,
                    ).strip(),
                    body_html=_render_candidate_template(
                        source_html,
                        candidate=candidate,
                        job=job,
                    ).strip(),
                )[0],
                body_html=normalize_rich_message(
                    body_text=_render_candidate_template(
                        source_text,
                        candidate=candidate,
                        job=job,
                    ).strip(),
                    body_html=_render_candidate_template(
                        source_html,
                        candidate=candidate,
                        job=job,
                    ).strip(),
                )[1],
                template_id=template.id if template is not None else None,
                render_variables=build_render_variables(candidate, job),
                sent_status=SentStatus.NOT_SENT,
            )
            db.add(message)
            db.flush()
            results.append(
                BatchCandidateResult(
                    candidate_profile_id=str(candidate.id),
                    full_name=candidate.full_name,
                    status="created",
                    record_id=str(message.id),
                )
            )

        db.commit()
        return _batch_response(results)
    except Exception:
        db.rollback()
        raise


@router.post(
    "/collections/{collection_id}/interview-invitations",
    response_model=BatchActionResponse,
    status_code=201,
    tags=["collections"],
)
def create_collection_interview_invitations(
    collection_id: uuid.UUID,
    body: InterviewInvitationBatchRequest,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    try:
        collection = get_current_user_owned_shortlist_collection(
            db,
            current_user.id,
            collection_id,
        )
        candidates = _load_collection_candidates(db, collection)
        if body.interview_template_id is not None:
            template = get_current_user_owned_active_interview_template(
                db,
                current_user.id,
                body.job_id,
                body.interview_template_id,
            )
        else:
            question_set = get_job_scoped_interview_question_set(
                db,
                user_id=current_user.id,
                job_id=body.job_id,
                question_set_id=body.interview_question_set_id,
            )
            template = materialize_question_set_template(db, job_id=body.job_id, question_set=question_set)

        selected, missing_ids = _selected_collection_candidates(
            candidates, body.candidate_profile_ids
        )
        expires_at = (
            datetime.now(timezone.utc) + timedelta(hours=body.expires_in_hours)
            if body.expires_in_hours is not None
            else None
        )
        results: list[BatchCandidateResult] = [
            BatchCandidateResult(
                candidate_profile_id=str(candidate_id),
                full_name=None,
                status="skipped_not_in_collection",
                reason="Candidate is not in this shortlist collection.",
            )
            for candidate_id in missing_ids
        ]

        for candidate in selected:
            candidate_job = _candidate_job(candidate)
            if candidate_job is None or candidate_job.id != body.job_id:
                results.append(
                    BatchCandidateResult(
                        candidate_profile_id=str(candidate.id),
                        full_name=candidate.full_name,
                        status="skipped_job_mismatch",
                        reason="Candidate does not belong to the selected job.",
                    )
                )
                continue
            if body.send_email and not candidate.email:
                results.append(
                    BatchCandidateResult(
                        candidate_profile_id=str(candidate.id),
                        full_name=candidate.full_name,
                        status="skipped_missing_email",
                        reason="Candidate has no email address.",
                    )
                )
                continue

            duplicate = (
                db.query(InterviewInvitation)
                .filter(
                    InterviewInvitation.candidate_profile_id == candidate.id,
                    InterviewInvitation.job_id == body.job_id,
                    InterviewInvitation.interview_template_id == template.id,
                    InterviewInvitation.status.in_(["pending", "in_progress", "completed"]),
                )
                .order_by(InterviewInvitation.created_at.desc())
                .first()
            )
            if duplicate is not None:
                results.append(
                    BatchCandidateResult(
                        candidate_profile_id=str(candidate.id),
                        full_name=candidate.full_name,
                        status="skipped_duplicate",
                        reason="Candidate already has an active interview invitation.",
                        record_id=str(duplicate.id),
                    )
                )
                continue

            invitation = InterviewInvitation(
                job_id=body.job_id,
                candidate_profile_id=candidate.id,
                interview_template_id=template.id,
                expires_at=expires_at,
                sent_by_user_id=collection.created_by_user_id,
            )
            db.add(invitation)
            db.flush()
            if body.send_email:
                from worker.tasks import send_interview_invitation_email

                delay = getattr(send_interview_invitation_email, "delay", None)
                if callable(delay):
                    delay(str(invitation.id))
            results.append(
                BatchCandidateResult(
                    candidate_profile_id=str(candidate.id),
                    full_name=candidate.full_name,
                    status="created",
                    record_id=str(invitation.id),
                )
            )

        db.commit()
        return _batch_response(results)
    except Exception:
        db.rollback()
        raise


@router.patch(
    "/collections/{collection_id}",
    response_model=CollectionResponse,
    tags=["collections"],
)
def update_collection(
    collection_id: uuid.UUID,
    body: CollectionUpdateRequest,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    try:
        collection = get_current_user_owned_shortlist_collection(
            db,
            current_user.id,
            collection_id,
        )
        collection.name = body.name
        db.commit()
        db.refresh(collection)
        return _ser_collection(collection)
    except Exception as exc:
        db.rollback()
        if isinstance(exc, IntegrityError) and _is_unique_violation(
            exc,
            "uq_shortlist_creator_name",
            "unique constraint failed: shortlist_collections.created_by_user_id, shortlist_collections.name",
        ):
            raise HTTPException(
                status_code=409,
                detail=f"Collection named '{body.name}' already exists for this user",
            ) from exc
        raise


@router.delete("/collections/{collection_id}", status_code=204, tags=["collections"])
def delete_collection(
    collection_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    collection = get_current_user_owned_shortlist_collection(
        db,
        current_user.id,
        collection_id,
    )
    db.delete(collection)
    db.commit()


# ---------------------------------------------------------------------------
# ShortlistItem endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/collections/{collection_id}/items",
    response_model=ItemResponse,
    status_code=201,
    tags=["items"],
)
def add_item(
    collection_id: uuid.UUID,
    body: ItemAddRequest,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    try:
        collection = get_current_user_owned_shortlist_collection(
            db,
            current_user.id,
            collection_id,
        )
        item = ShortlistItem(
            shortlist_collection_id=collection.id,
            candidate_profile_id=body.candidate_profile_id,
        )
        db.add(item)
        db.commit()
        db.refresh(item)
        return _ser_item(item)
    except Exception as exc:
        db.rollback()
        if isinstance(exc, IntegrityError) and _is_unique_violation(
            exc,
            "uq_shortlist_item_unique",
            "unique constraint failed: shortlist_items.shortlist_collection_id, shortlist_items.candidate_profile_id",
        ):
            raise HTTPException(
                status_code=409,
                detail=f"Candidate '{body.candidate_profile_id}' is already in this collection",
            ) from exc
        raise


@router.get(
    "/collections/{collection_id}/items",
    response_model=ItemListResponse,
    tags=["items"],
)
def list_items(
    collection_id: uuid.UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    collection = get_current_user_owned_shortlist_collection(
        db,
        current_user.id,
        collection_id,
    )
    total = (
        db.query(ShortlistItem)
        .filter(ShortlistItem.shortlist_collection_id == collection.id)
        .count()
    )
    rows = (
        db.query(ShortlistItem)
        .filter(ShortlistItem.shortlist_collection_id == collection.id)
        .order_by(ShortlistItem.added_at.asc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return ItemListResponse(items=[_ser_item(i) for i in rows], total=total)


@router.delete(
    "/collections/{collection_id}/items/{candidate_id}",
    status_code=204,
    tags=["items"],
)
def remove_item(
    collection_id: uuid.UUID,
    candidate_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    item = get_current_user_owned_shortlist_item(
        db,
        current_user.id,
        collection_id,
        candidate_id,
    )
    db.delete(item)
    db.commit()
