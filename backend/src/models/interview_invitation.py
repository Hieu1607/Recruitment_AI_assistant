from __future__ import annotations

import secrets
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, ForeignKeyConstraint, Integer, String, event, func, select
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship

from src.models.base import Base

if TYPE_CHECKING:
    from src.models.candidate_profile import CandidateProfile
    from src.models.interview_session import InterviewSession
    from src.models.interview_template import InterviewTemplate
    from src.models.job import Job


def generate_interview_public_token() -> str:
    return secrets.token_urlsafe(32)


class InterviewInvitation(Base):
    __tablename__ = "interview_invitations"
    __table_args__ = (
        CheckConstraint("max_attempts > 0", name="ck_interview_invitations_max_attempts_positive"),
        CheckConstraint("attempt_count >= 0", name="ck_interview_invitations_attempt_count_non_negative"),
        CheckConstraint("attempt_count <= max_attempts", name="ck_interview_invitations_attempt_count_within_max"),
        ForeignKeyConstraint(
            ["interview_template_id", "job_id"],
            ["interview_templates.id", "interview_templates.job_id"],
            ondelete="CASCADE",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    candidate_profile_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("candidate_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    interview_template_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    public_token: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
        index=True,
        default=generate_interview_public_token,
    )
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending", server_default="pending")
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    sent_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("user_accounts.id", ondelete="SET NULL"),
        nullable=True,
    )
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    job: Mapped["Job"] = relationship(
        back_populates="interview_invitations",
        overlaps="interview_template,invitations",
    )
    candidate_profile: Mapped["CandidateProfile"] = relationship(back_populates="interview_invitations")
    interview_template: Mapped["InterviewTemplate"] = relationship(
        back_populates="invitations",
        overlaps="job,interview_invitations",
    )
    sessions: Mapped[list["InterviewSession"]] = relationship(
        back_populates="invitation",
        cascade="all, delete-orphan",
    )


@event.listens_for(Session, "before_flush")
def validate_interview_invitation_job_consistency(session: Session, _flush_context, _instances) -> None:
    from src.models.candidate_profile import CandidateProfile

    for invitation in session.new.union(session.dirty):
        if not isinstance(invitation, InterviewInvitation) or invitation in session.deleted:
            continue
        if invitation.job_id is None or invitation.candidate_profile_id is None:
            continue

        candidate_profile = session.execute(
            select(CandidateProfile)
            .join(CandidateProfile.resume_document)
            .where(CandidateProfile.id == invitation.candidate_profile_id)
        ).scalar_one_or_none()

        if candidate_profile is None:
            continue

        if candidate_profile.resume_document.job_id != invitation.job_id:
            raise ValueError(
                "Interview invitation job_id must match the candidate profile resume_document.job_id."
            )
