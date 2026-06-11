from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.services.public_job_service import generate_public_apply_token

if TYPE_CHECKING:
    from src.models.interview_invitation import InterviewInvitation
    from src.models.interview_template import InterviewTemplate
    from src.models.job_matching import JobDescription
    from src.models.resume_document import ResumeDocument
    from src.models.user_account import UserAccount


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("user_accounts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="active", server_default="active")
    public_apply_token: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
        default=generate_public_apply_token,
    )
    public_apply_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
    )
    candidate_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    public_apply_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    public_apply_disabled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    owner: Mapped["UserAccount"] = relationship(back_populates="jobs")
    job_descriptions: Mapped[list["JobDescription"]] = relationship(back_populates="job")
    interview_templates: Mapped[list["InterviewTemplate"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
    )
    interview_invitations: Mapped[list["InterviewInvitation"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        overlaps="interview_template,invitations",
    )
    resumes: Mapped[list["ResumeDocument"]] = relationship(back_populates="job")
