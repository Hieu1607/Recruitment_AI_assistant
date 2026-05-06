from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Enum as SqlEnum, ForeignKey, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.enums import ProfileStatus


_ENUM_VALUES = lambda enum_cls: [item.value for item in enum_cls]

if TYPE_CHECKING:
    from src.models.job_matching import InterviewQuestionSet, MatchResult
    from src.models.outreach import OutreachMessage
    from src.models.query_shortlist import ShortlistItem
    from src.models.resume_document import ResumeDocument


class CandidateProfile(Base):
    __tablename__ = "candidate_profiles"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    resume_document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resume_documents.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    email: Mapped[str | None] = mapped_column(String(320), nullable=True, index=True)
    location_normalized: Mapped[str | None] = mapped_column(String(255), nullable=True)
    contact: Mapped[str | None] = mapped_column(String(255), nullable=True)
    current_job_title: Mapped[str | None] = mapped_column(String(255), nullable=True)

    educated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    ever_studied_abroad: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    major: Mapped[str | None] = mapped_column(String(255), nullable=True)
    cpa: Mapped[str | None] = mapped_column(String(100), nullable=True)

    education_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    experience_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    experience_years: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    skills_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    languages_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    projects_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    achievements_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    publications_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    certifications_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    references_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    other_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    profile_status: Mapped[ProfileStatus] = mapped_column(
        SqlEnum(ProfileStatus, name="profile_status_enum", values_callable=_ENUM_VALUES),
        nullable=False,
        default=ProfileStatus.DRAFT,
        server_default=ProfileStatus.DRAFT.value,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    resume_document: Mapped["ResumeDocument"] = relationship(back_populates="candidate_profile")
    match_results: Mapped[list["MatchResult"]] = relationship(back_populates="candidate_profile", cascade="all, delete-orphan")
    outreach_messages: Mapped[list["OutreachMessage"]] = relationship(back_populates="candidate_profile", cascade="all, delete-orphan")
    interview_question_sets: Mapped[list["InterviewQuestionSet"]] = relationship(back_populates="candidate_profile", cascade="all, delete-orphan")
    shortlist_items: Mapped[list["ShortlistItem"]] = relationship(back_populates="candidate_profile", cascade="all, delete-orphan")
