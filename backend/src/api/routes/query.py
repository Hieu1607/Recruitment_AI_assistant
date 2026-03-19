from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.agents.router.query_router import build_routing_plan
from src.agents.tools.llm_semantic_tool import run_llm_semantic_search
from src.agents.tools.sql_search_tool import run_sql_search
from src.agents.verifier.query_verifier import QueryExecutionContext, verify_query_result
from src.api.dependencies.auth import CurrentUser, require_roles
from src.api.errors import AppError
from src.repositories.db import get_session
from src.services.query.query_history_service import query_history_service

router = APIRouter(prefix="/v1/query-sessions", tags=["query"])


class QuerySessionCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=255)


class QuerySessionResponse(BaseModel):
    id: str
    title: str | None


class AskRequest(BaseModel):
    question: str = Field(min_length=3)


class AskResponse(BaseModel):
    answer: str
    matchedCount: int
    matchedCandidateIds: list[str]
    routingStrategy: str
    queryTurnId: str


@router.post("", response_model=QuerySessionResponse)
def create_query_session(
    payload: QuerySessionCreateRequest,
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[CurrentUser, Depends(require_roles("admin", "recruiter", "viewer"))],
) -> QuerySessionResponse:
    user_id = uuid.UUID(current_user.user_id)
    query_session = query_history_service.ensure_session(
        session,
        user_id=user_id,
        session_title=payload.title,
    )
    session.commit()
    return QuerySessionResponse(id=str(query_session.id), title=query_session.session_title)


@router.get("", response_model=list[QuerySessionResponse])
def list_query_sessions(
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[CurrentUser, Depends(require_roles("admin", "recruiter", "viewer"))],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[QuerySessionResponse]:
    user_id = uuid.UUID(current_user.user_id)
    rows = query_history_service.list_sessions_for_user(session, user_id=user_id, limit=limit)
    return [QuerySessionResponse(id=str(row.id), title=row.session_title) for row in rows]


@router.post("/{session_id}/ask", response_model=AskResponse)
def ask_query(
    session_id: str,
    payload: AskRequest,
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[CurrentUser, Depends(require_roles("admin", "recruiter", "viewer"))],
) -> AskResponse:
    user_id = uuid.UUID(current_user.user_id)
    try:
        parsed_session_id = uuid.UUID(session_id)
    except ValueError as exc:
        raise AppError(code="invalid_session", message="sessionId must be a UUID", status_code=422) from exc

    query_session = query_history_service.ensure_session(
        session,
        user_id=user_id,
        session_id=parsed_session_id,
    )

    plan = build_routing_plan(payload.question)
    sql_result_ids: list[str] = []
    llm_result_ids: list[str] = []
    traces: dict[str, dict] = {}

    if plan.run_sql:
        sql_result = run_sql_search(session, payload.question)
        sql_result_ids = sql_result.candidate_ids
        traces["sql"] = sql_result.trace

    if plan.run_llm:
        llm_result = run_llm_semantic_search(session, payload.question)
        llm_result_ids = llm_result.candidate_ids
        traces["llm"] = llm_result.trace

    verified = verify_query_result(
        QueryExecutionContext(
            question=payload.question,
            strategy=plan.strategy,
            sql_candidate_ids=sql_result_ids,
            llm_candidate_ids=llm_result_ids,
        )
    )

    persisted = query_history_service.persist_turn(
        session,
        query_session_id=query_session.id,
        question=payload.question,
        routing_strategy=verified.strategy,
        answer_text=verified.answer_text,
        matched_candidate_ids=verified.matched_candidate_ids,
        matched_count=verified.matched_count,
        tool_trace_masked={"router": plan.strategy.value, "tools": traces, "verifier": verified.tool_trace},
    )

    session.commit()
    return AskResponse(
        answer=verified.answer_text,
        matchedCount=verified.matched_count,
        matchedCandidateIds=verified.matched_candidate_ids,
        routingStrategy=verified.strategy.value,
        queryTurnId=persisted.query_turn_id,
    )
