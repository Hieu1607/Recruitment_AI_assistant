from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models.matching import QuerySession, QueryTurn, RoutingStrategy
from src.services.observability.audit_logger import audit_log


@dataclass
class PersistedQueryTurn:
    query_turn_id: str
    query_session_id: str


class QueryHistoryService:
    def ensure_session(
        self,
        session: Session,
        user_id: uuid.UUID,
        session_id: uuid.UUID | None = None,
        session_title: str | None = None,
    ) -> QuerySession:
        if session_id:
            existing = session.get(QuerySession, session_id)
            if existing:
                return existing

        query_session = QuerySession(user_id=user_id, session_title=session_title)
        session.add(query_session)
        session.flush()
        return query_session

    def list_sessions_for_user(self, session: Session, user_id: uuid.UUID, limit: int = 50) -> list[QuerySession]:
        stmt = (
            select(QuerySession)
            .where(QuerySession.user_id == user_id)
            .order_by(QuerySession.updated_at.desc())
            .limit(max(1, min(limit, 200)))
        )
        return list(session.scalars(stmt))

    def persist_turn(
        self,
        session: Session,
        query_session_id: uuid.UUID,
        question: str,
        routing_strategy: RoutingStrategy,
        answer_text: str,
        matched_candidate_ids: list[str],
        matched_count: int,
        tool_trace_masked: dict,
    ) -> PersistedQueryTurn:
        query_turn = QueryTurn(
            query_session_id=query_session_id,
            user_question=question,
            routing_strategy=routing_strategy,
            answer_text=answer_text,
            matched_candidate_ids=matched_candidate_ids,
            matched_count=matched_count,
            tool_trace_masked=tool_trace_masked,
        )
        session.add(query_turn)
        session.flush()

        audit_log(
            "query_turn_persisted",
            {
                "query_session_id": str(query_session_id),
                "query_turn_id": str(query_turn.id),
                "routing_strategy": routing_strategy.value,
                "matched_count": matched_count,
                "question": question,
            },
        )

        return PersistedQueryTurn(query_turn_id=str(query_turn.id), query_session_id=str(query_session_id))


query_history_service = QueryHistoryService()
