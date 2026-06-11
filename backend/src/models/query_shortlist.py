from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base

if TYPE_CHECKING:
    from src.models.candidate_profile import CandidateProfile
    from src.models.job import Job


class QuerySession(Base):
    __tablename__ = "query_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    session_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    turns: Mapped[list["QueryTurn"]] = relationship(back_populates="query_session", cascade="all, delete-orphan")
    job: Mapped["Job | None"] = relationship()


class QueryTurn(Base):
    __tablename__ = "query_turns"
    __table_args__ = (
        CheckConstraint("matched_count IS NULL OR matched_count >= 0", name="matched_count_non_negative"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("query_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_question: Mapped[str] = mapped_column(Text, nullable=False)
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    matched_candidate_ids: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    matched_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tool_trace_masked: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    query_session: Mapped[QuerySession] = relationship(back_populates="turns")
    shortlist_collections: Mapped[list["ShortlistCollection"]] = relationship(back_populates="source_query_turn")


class ShortlistCollection(Base):
    __tablename__ = "shortlist_collections"
    __table_args__ = (
        UniqueConstraint("created_by_user_id", "name", name="uq_shortlist_creator_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_by_user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    source_query_turn_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("query_turns.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    source_query_turn: Mapped[QueryTurn | None] = relationship(back_populates="shortlist_collections")
    items: Mapped[list["ShortlistItem"]] = relationship(back_populates="shortlist_collection", cascade="all, delete-orphan")


class ShortlistItem(Base):
    __tablename__ = "shortlist_items"
    __table_args__ = (
        UniqueConstraint("shortlist_collection_id", "candidate_profile_id", name="uq_shortlist_item_unique"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    shortlist_collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("shortlist_collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    candidate_profile_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("candidate_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    shortlist_collection: Mapped[ShortlistCollection] = relationship(back_populates="items")
    candidate_profile: Mapped["CandidateProfile"] = relationship(back_populates="shortlist_items")
