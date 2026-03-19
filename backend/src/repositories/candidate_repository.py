from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import Select, or_, select
from sqlalchemy.orm import Session

from src.models.candidate import CandidateProfile, ExtractionTrace


class CandidateRepository:
    def list_candidates(self, session: Session, query: str | None = None, limit: int = 50) -> list[CandidateProfile]:
        stmt: Select[tuple[CandidateProfile]] = select(CandidateProfile).order_by(
            CandidateProfile.updated_at.desc()
        )
        if query:
            pattern = f"%{query.strip()}%"
            stmt = stmt.where(
                or_(
                    CandidateProfile.full_name.ilike(pattern),
                    CandidateProfile.email.ilike(pattern),
                    CandidateProfile.current_job_title.ilike(pattern),
                    CandidateProfile.skills_text.ilike(pattern),
                )
            )
        stmt = stmt.limit(max(1, min(limit, 200)))
        return list(session.scalars(stmt))

    def get_candidate(self, session: Session, candidate_id: uuid.UUID) -> CandidateProfile | None:
        return session.get(CandidateProfile, candidate_id)

    def get_candidates_by_ids(
        self, session: Session, candidate_ids: list[uuid.UUID]
    ) -> list[CandidateProfile]:
        if not candidate_ids:
            return []
        stmt = (
            select(CandidateProfile)
            .where(CandidateProfile.id.in_(candidate_ids))
            .order_by(CandidateProfile.updated_at.desc())
        )
        return list(session.scalars(stmt))

    def update_candidate(
        self,
        session: Session,
        candidate_id: uuid.UUID,
        patch: dict[str, Any],
    ) -> CandidateProfile | None:
        candidate = self.get_candidate(session, candidate_id)
        if not candidate:
            return None
        for field, value in patch.items():
            if hasattr(candidate, field):
                setattr(candidate, field, value)
        session.add(candidate)
        session.flush()
        return candidate

    def get_traces_for_candidate(
        self, session: Session, candidate_id: uuid.UUID, limit: int = 100
    ) -> list[ExtractionTrace]:
        stmt = (
            select(ExtractionTrace)
            .where(ExtractionTrace.candidate_profile_id == candidate_id)
            .order_by(ExtractionTrace.created_at.desc())
            .limit(max(1, min(limit, 500)))
        )
        return list(session.scalars(stmt))


candidate_repository = CandidateRepository()
