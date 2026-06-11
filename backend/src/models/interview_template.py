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
    from src.models.interview_session import InterviewReport
    from src.models.job import Job


class InterviewTemplate(Base):
    __tablename__ = "interview_templates"
    __table_args__ = (
        UniqueConstraint("id", "job_id", name="uq_interview_templates_id_job_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    language_code: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="vi-VN",
        server_default="vi-VN",
    )
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="draft", server_default="draft")
    intro_script: Mapped[str | None] = mapped_column(Text, nullable=True)
    closing_script: Mapped[str | None] = mapped_column(Text, nullable=True)
    question_payload: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'"),
    )
    report_rubric: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'"),
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    job: Mapped["Job"] = relationship(back_populates="interview_templates")
    invitations: Mapped[list["InterviewInvitation"]] = relationship(
        back_populates="interview_template",
        cascade="all, delete-orphan",
        overlaps="job,interview_invitations",
    )
    reports: Mapped[list["InterviewReport"]] = relationship(back_populates="interview_template")
