from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum as SqlEnum, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.enums import ContentSource, SentStatus


_ENUM_VALUES = lambda enum_cls: [item.value for item in enum_cls]

if TYPE_CHECKING:
    from src.models.candidate_profile import CandidateProfile


class OutreachMessage(Base):
    __tablename__ = "outreach_messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_profile_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("candidate_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_by_user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    content_source: Mapped[ContentSource] = mapped_column(
        SqlEnum(ContentSource, name="content_source_enum", values_callable=_ENUM_VALUES),
        nullable=False,
    )
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    sent_status: Mapped[SentStatus] = mapped_column(
        SqlEnum(SentStatus, name="sent_status_enum", values_callable=_ENUM_VALUES),
        nullable=False,
        default=SentStatus.NOT_SENT,
        server_default=SentStatus.NOT_SENT.value,
    )
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    candidate_profile: Mapped["CandidateProfile"] = relationship(back_populates="outreach_messages")
