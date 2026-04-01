from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SqlEnum,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.enums import MatchRunStatus


_ENUM_VALUES = lambda enum_cls: [item.value for item in enum_cls]

if TYPE_CHECKING:
    from src.models.candidate_profile import CandidateProfile


class JobDescription(Base):
    __tablename__ = "job_descriptions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    jd_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_by_user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")

    match_runs: Mapped[list["MatchRun"]] = relationship(back_populates="job_description", cascade="all, delete-orphan")
    interview_question_sets: Mapped[list["InterviewQuestionSet"]] = relationship(back_populates="job_description", cascade="all, delete-orphan")


class MatchRun(Base):
    __tablename__ = "match_runs"
    __table_args__ = (
        CheckConstraint("score_threshold >= 0 AND score_threshold <= 100", name="score_threshold_range"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_description_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("job_descriptions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    score_threshold: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    initiated_by_user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    run_status: Mapped[MatchRunStatus] = mapped_column(
        SqlEnum(MatchRunStatus, name="match_run_status_enum", values_callable=_ENUM_VALUES),
        nullable=False,
        default=MatchRunStatus.RUNNING,
        server_default=MatchRunStatus.RUNNING.value,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    job_description: Mapped[JobDescription] = relationship(back_populates="match_runs")
    match_results: Mapped[list["MatchResult"]] = relationship(back_populates="match_run", cascade="all, delete-orphan")


class MatchResult(Base):
    __tablename__ = "match_results"
    __table_args__ = (
        CheckConstraint("total_score >= 0 AND total_score <= 100", name="total_score_range"),
        UniqueConstraint("match_run_id", "candidate_profile_id", name="uq_match_results_run_candidate"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    match_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("match_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    candidate_profile_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("candidate_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    score_list_index: Mapped[int] = mapped_column(Integer, nullable=False)
    total_score: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    passed_threshold: Mapped[bool] = mapped_column(Boolean, nullable=False)
    rationale_summary: Mapped[str] = mapped_column(Text, nullable=False)
    component_scores: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    match_run: Mapped[MatchRun] = relationship(back_populates="match_results")
    candidate_profile: Mapped["CandidateProfile"] = relationship(back_populates="match_results")


class InterviewQuestionSet(Base):
    __tablename__ = "interview_question_sets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_profile_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("candidate_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    job_description_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("job_descriptions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    generated_by_user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    question_payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    candidate_profile: Mapped["CandidateProfile"] = relationship(back_populates="interview_question_sets")
    job_description: Mapped[JobDescription] = relationship(back_populates="interview_question_sets")
