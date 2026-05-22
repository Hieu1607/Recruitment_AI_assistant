from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base

if TYPE_CHECKING:
    from src.models.interview_invitation import InterviewInvitation
    from src.models.interview_template import InterviewTemplate


class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    interview_invitation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("interview_invitations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    provider: Mapped[str | None] = mapped_column(String(100), nullable=True)
    provider_session_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="created", server_default="created")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    device_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    browser_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    connection_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    invitation: Mapped["InterviewInvitation"] = relationship(back_populates="sessions")
    response_items: Mapped[list["InterviewResponseItem"]] = relationship(
        back_populates="interview_session",
        cascade="all, delete-orphan",
    )
    transcript_turns: Mapped[list["InterviewTranscriptTurn"]] = relationship(
        back_populates="interview_session",
        cascade="all, delete-orphan",
    )
    report: Mapped["InterviewReport | None"] = relationship(
        back_populates="interview_session",
        uselist=False,
        cascade="all, delete-orphan",
    )


class InterviewResponseItem(Base):
    __tablename__ = "interview_response_items"
    __table_args__ = (
        UniqueConstraint("interview_session_id", "question_key", name="uq_interview_response_items_session_question"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    interview_session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_key: Mapped[str] = mapped_column(String(255), nullable=False)
    question_order: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prompt_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    interview_session: Mapped["InterviewSession"] = relationship(back_populates="response_items")
    transcript_turns: Mapped[list["InterviewTranscriptTurn"]] = relationship(
        back_populates="response_item",
        passive_deletes=True,
    )


class InterviewTranscriptTurn(Base):
    __tablename__ = "interview_transcript_turns"
    __table_args__ = (
        UniqueConstraint(
            "interview_session_id",
            "turn_index",
            name="uq_interview_transcript_turns_session_turn_index",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    interview_session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    response_item_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("interview_response_items.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    speaker_role: Mapped[str] = mapped_column(String(50), nullable=False)
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    transcript_text: Mapped[str] = mapped_column(Text, nullable=False)
    time_offset_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    interview_session: Mapped["InterviewSession"] = relationship(back_populates="transcript_turns")
    response_item: Mapped["InterviewResponseItem | None"] = relationship(back_populates="transcript_turns")


class InterviewReport(Base):
    __tablename__ = "interview_reports"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    interview_session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    interview_template_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("interview_templates.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    summary_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_payload: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    interview_session: Mapped["InterviewSession"] = relationship(back_populates="report")
    interview_template: Mapped["InterviewTemplate | None"] = relationship(back_populates="reports")
