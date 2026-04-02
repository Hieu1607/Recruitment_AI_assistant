"""CRUD endpoints for QuerySession, QueryTurn, ShortlistCollection, ShortlistItem.

Route map
---------
Sessions
  POST   /shortlist/sessions/                          create session
  GET    /shortlist/sessions/                          list sessions for a user
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
  GET    /shortlist/collections/                       list collections for a user
  GET    /shortlist/collections/{collection_id}        get collection + items
  PATCH  /shortlist/collections/{collection_id}        rename collection
  DELETE /shortlist/collections/{collection_id}        delete collection (cascades items)

Items
  POST   /shortlist/collections/{collection_id}/items  add candidate to collection
  GET    /shortlist/collections/{collection_id}/items  list items in collection
  DELETE /shortlist/collections/{collection_id}/items/{candidate_id}  remove item
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.models.query_shortlist import (
    QuerySession,
    QueryTurn,
    ShortlistCollection,
    ShortlistItem,
)
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

# --- QuerySession ---

class SessionCreateRequest(BaseModel):
    user_id: uuid.UUID
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
    created_by_user_id: uuid.UUID
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


# --- ShortlistItem ---

class ItemAddRequest(BaseModel):
    candidate_profile_id: uuid.UUID


class ItemResponse(BaseModel):
    id: str
    shortlist_collection_id: str
    candidate_profile_id: str
    added_at: datetime


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
        source_query_turn_id=str(c.source_query_turn_id) if c.source_query_turn_id else None,
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


# ---------------------------------------------------------------------------
# QuerySession endpoints
# ---------------------------------------------------------------------------

@router.post("/sessions/", response_model=SessionResponse, status_code=201, tags=["sessions"])
def create_session(body: SessionCreateRequest):
    db = SessionLocal()
    try:
        session = QuerySession(user_id=body.user_id, session_title=body.session_title)
        db.add(session)
        db.commit()
        db.refresh(session)
        return _ser_session(session)
    finally:
        db.close()


@router.get("/sessions/", response_model=List[SessionResponse], tags=["sessions"])
def list_sessions(
    user_id: uuid.UUID = Query(..., description="Filter sessions by user UUID"),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    db = SessionLocal()
    try:
        rows = (
            db.query(QuerySession)
            .filter(QuerySession.user_id == user_id)
            .order_by(QuerySession.updated_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [_ser_session(s) for s in rows]
    finally:
        db.close()


@router.get("/sessions/{session_id}", response_model=SessionResponse, tags=["sessions"])
def get_session(session_id: uuid.UUID):
    db = SessionLocal()
    try:
        session = _get_or_404(db, QuerySession, session_id, "Session")
        return _ser_session(session)
    finally:
        db.close()


@router.patch("/sessions/{session_id}", response_model=SessionResponse, tags=["sessions"])
def update_session(session_id: uuid.UUID, body: SessionUpdateRequest):
    db = SessionLocal()
    try:
        session = _get_or_404(db, QuerySession, session_id, "Session")
        if body.session_title is not None:
            session.session_title = body.session_title
        db.commit()
        db.refresh(session)
        return _ser_session(session)
    finally:
        db.close()


@router.delete("/sessions/{session_id}", status_code=204, tags=["sessions"])
def delete_session(session_id: uuid.UUID):
    db = SessionLocal()
    try:
        session = _get_or_404(db, QuerySession, session_id, "Session")
        db.delete(session)
        db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# QueryTurn endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/sessions/{session_id}/turns",
    response_model=TurnResponse,
    status_code=201,
    tags=["turns"],
)
def create_turn(session_id: uuid.UUID, body: TurnCreateRequest):
    db = SessionLocal()
    try:
        _get_or_404(db, QuerySession, session_id, "Session")
        turn = QueryTurn(
            query_session_id=session_id,
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
    finally:
        db.close()


@router.get(
    "/sessions/{session_id}/turns",
    response_model=List[TurnResponse],
    tags=["turns"],
)
def list_turns(
    session_id: uuid.UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    db = SessionLocal()
    try:
        _get_or_404(db, QuerySession, session_id, "Session")
        rows = (
            db.query(QueryTurn)
            .filter(QueryTurn.query_session_id == session_id)
            .order_by(QueryTurn.created_at.asc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [_ser_turn(t) for t in rows]
    finally:
        db.close()


@router.get("/turns/{turn_id}", response_model=TurnResponse, tags=["turns"])
def get_turn(turn_id: uuid.UUID):
    db = SessionLocal()
    try:
        turn = _get_or_404(db, QueryTurn, turn_id, "Turn")
        return _ser_turn(turn)
    finally:
        db.close()


@router.delete("/turns/{turn_id}", status_code=204, tags=["turns"])
def delete_turn(turn_id: uuid.UUID):
    db = SessionLocal()
    try:
        turn = _get_or_404(db, QueryTurn, turn_id, "Turn")
        db.delete(turn)
        db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# ShortlistCollection endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/collections/",
    response_model=CollectionResponse,
    status_code=201,
    tags=["collections"],
)
def create_collection(body: CollectionCreateRequest):
    db = SessionLocal()
    try:
        collection = ShortlistCollection(
            name=body.name,
            created_by_user_id=body.created_by_user_id,
            source_query_turn_id=body.source_query_turn_id,
        )
        db.add(collection)
        db.commit()
        db.refresh(collection)
        return _ser_collection(collection)
    except Exception as exc:
        db.rollback()
        if "uq_shortlist_creator_name" in str(exc):
            raise HTTPException(
                status_code=409,
                detail=f"Collection named '{body.name}' already exists for this user",
            ) from exc
        raise
    finally:
        db.close()


@router.get(
    "/collections/",
    response_model=List[CollectionResponse],
    tags=["collections"],
)
def list_collections(
    user_id: uuid.UUID = Query(..., description="Filter collections by creator user UUID"),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    db = SessionLocal()
    try:
        rows = (
            db.query(ShortlistCollection)
            .filter(ShortlistCollection.created_by_user_id == user_id)
            .order_by(ShortlistCollection.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [_ser_collection(c) for c in rows]
    finally:
        db.close()


@router.get(
    "/collections/{collection_id}",
    response_model=CollectionResponse,
    tags=["collections"],
)
def get_collection(collection_id: uuid.UUID):
    db = SessionLocal()
    try:
        collection = _get_or_404(db, ShortlistCollection, collection_id, "Collection")
        return _ser_collection(collection)
    finally:
        db.close()


@router.patch(
    "/collections/{collection_id}",
    response_model=CollectionResponse,
    tags=["collections"],
)
def update_collection(collection_id: uuid.UUID, body: CollectionUpdateRequest):
    db = SessionLocal()
    try:
        collection = _get_or_404(db, ShortlistCollection, collection_id, "Collection")
        collection.name = body.name
        db.commit()
        db.refresh(collection)
        return _ser_collection(collection)
    except Exception as exc:
        db.rollback()
        if "uq_shortlist_creator_name" in str(exc):
            raise HTTPException(
                status_code=409,
                detail=f"Collection named '{body.name}' already exists for this user",
            ) from exc
        raise
    finally:
        db.close()


@router.delete("/collections/{collection_id}", status_code=204, tags=["collections"])
def delete_collection(collection_id: uuid.UUID):
    db = SessionLocal()
    try:
        collection = _get_or_404(db, ShortlistCollection, collection_id, "Collection")
        db.delete(collection)
        db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# ShortlistItem endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/collections/{collection_id}/items",
    response_model=ItemResponse,
    status_code=201,
    tags=["items"],
)
def add_item(collection_id: uuid.UUID, body: ItemAddRequest):
    db = SessionLocal()
    try:
        _get_or_404(db, ShortlistCollection, collection_id, "Collection")
        item = ShortlistItem(
            shortlist_collection_id=collection_id,
            candidate_profile_id=body.candidate_profile_id,
        )
        db.add(item)
        db.commit()
        db.refresh(item)
        return _ser_item(item)
    except Exception as exc:
        db.rollback()
        if "uq_shortlist_item_unique" in str(exc):
            raise HTTPException(
                status_code=409,
                detail=f"Candidate '{body.candidate_profile_id}' is already in this collection",
            ) from exc
        raise
    finally:
        db.close()


@router.get(
    "/collections/{collection_id}/items",
    response_model=List[ItemResponse],
    tags=["items"],
)
def list_items(
    collection_id: uuid.UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
):
    db = SessionLocal()
    try:
        _get_or_404(db, ShortlistCollection, collection_id, "Collection")
        rows = (
            db.query(ShortlistItem)
            .filter(ShortlistItem.shortlist_collection_id == collection_id)
            .order_by(ShortlistItem.added_at.asc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [_ser_item(i) for i in rows]
    finally:
        db.close()


@router.delete(
    "/collections/{collection_id}/items/{candidate_id}",
    status_code=204,
    tags=["items"],
)
def remove_item(collection_id: uuid.UUID, candidate_id: uuid.UUID):
    db = SessionLocal()
    try:
        item = (
            db.query(ShortlistItem)
            .filter(
                ShortlistItem.shortlist_collection_id == collection_id,
                ShortlistItem.candidate_profile_id == candidate_id,
            )
            .first()
        )
        if item is None:
            raise HTTPException(
                status_code=404,
                detail=f"Candidate '{candidate_id}' not found in collection '{collection_id}'",
            )
        db.delete(item)
        db.commit()
    finally:
        db.close()
