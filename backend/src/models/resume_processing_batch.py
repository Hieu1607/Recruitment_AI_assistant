from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, Enum as SqlEnum, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.enums import ResumeProcessingBatchStatus

_ENUM_VALUES = lambda enum_cls: [item.value for item in enum_cls]

if TYPE_CHECKING:
    from src.models.resume_document import ResumeDocument


class ResumeProcessingBatch(Base):
    __tablename__ = "resume_processing_batches"
    __table_args__ = (
        CheckConstraint("total_count >= 1", name="ck_resume_batches_total_positive"),
        CheckConstraint("terminal_count >= 0", name="ck_resume_batches_terminal_nonnegative"),
        CheckConstraint("processed_count >= 0", name="ck_resume_batches_processed_nonnegative"),
        CheckConstraint("failed_count >= 0", name="ck_resume_batches_failed_nonnegative"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    total_count: Mapped[int] = mapped_column(Integer, nullable=False)
    terminal_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    processed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    failed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    status: Mapped[ResumeProcessingBatchStatus] = mapped_column(
        SqlEnum(
            ResumeProcessingBatchStatus,
            name="resume_processing_batch_status_enum",
            values_callable=_ENUM_VALUES,
        ),
        nullable=False,
        default=ResumeProcessingBatchStatus.PARSING,
        server_default=ResumeProcessingBatchStatus.PARSING.value,
    )
    evaluation_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    evaluation_dispatch_attempted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    resume_documents: Mapped[list["ResumeDocument"]] = relationship(back_populates="processing_batch")
