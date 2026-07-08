from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Enum as SqlEnum, ForeignKey, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.enums import CandidateEvaluationStatus

_ENUM_VALUES = lambda enum_cls: [item.value for item in enum_cls]

if TYPE_CHECKING:
    from src.models.candidate_profile import CandidateProfile
    from src.models.job import Job
    from src.models.job_matching import JobDescription, MatchRun
    from src.models.user_account import UserAccount


class CandidateEvaluation(Base):
    __tablename__ = "candidate_evaluations"
    __table_args__ = (
        UniqueConstraint(
            "job_description_id",
            "candidate_profile_id",
            "scoring_signature",
            name="uq_candidate_evaluations_jd_candidate_signature",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    job_description_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("job_descriptions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    candidate_profile_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("candidate_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    scoring_signature: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    rubric_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    raw_component_scores: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    rationale_summary: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    status: Mapped[CandidateEvaluationStatus] = mapped_column(
        SqlEnum(
            CandidateEvaluationStatus,
            name="candidate_evaluation_status_enum",
            values_callable=_ENUM_VALUES,
        ),
        nullable=False,
        default=CandidateEvaluationStatus.PENDING,
        server_default=CandidateEvaluationStatus.PENDING.value,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_match_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("match_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    scored_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    job: Mapped["Job"] = relationship()
    job_description: Mapped["JobDescription"] = relationship()
    candidate_profile: Mapped["CandidateProfile"] = relationship()
    source_match_run: Mapped["MatchRun | None"] = relationship()


class JobScoringPreference(Base):
    __tablename__ = "job_scoring_preferences"

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    section_weights: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    score_threshold: Mapped[Decimal] = mapped_column(
        Numeric(5, 2),
        nullable=False,
        default=Decimal("50.00"),
        server_default="50.00",
    )
    updated_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("user_accounts.id", ondelete="SET NULL"),
        nullable=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    job: Mapped["Job"] = relationship()
    updated_by_user: Mapped["UserAccount | None"] = relationship()
