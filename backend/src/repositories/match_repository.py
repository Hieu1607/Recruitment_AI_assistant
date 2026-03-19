from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models.matching import MatchResult


class MatchRepository:
    def list_results(
        self,
        session: Session,
        match_run_id: uuid.UUID,
        threshold: float | None = None,
        limit: int = 200,
    ) -> list[MatchResult]:
        stmt = select(MatchResult).where(MatchResult.match_run_id == match_run_id)
        if threshold is not None:
            stmt = stmt.where(MatchResult.total_score >= threshold)
        stmt = stmt.order_by(MatchResult.total_score.desc(), MatchResult.score_list_index.asc())
        stmt = stmt.limit(max(1, min(limit, 1000)))
        return list(session.scalars(stmt))


match_repository = MatchRepository()
