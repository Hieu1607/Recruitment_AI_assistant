from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, Text, Enum as SqlEnum, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.enums import UploadStatus


_ENUM_VALUES = lambda enum_cls: [item.value for item in enum_cls]

if TYPE_CHECKING:
    from src.models.candidate_profile import CandidateProfile
    from src.models.job import Job


class ResumeDocument(Base):
    __tablename__ = "resume_documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    upload_status: Mapped[UploadStatus] = mapped_column(
        SqlEnum(UploadStatus, name="upload_status_enum", values_callable=_ENUM_VALUES),
        nullable=False,
        default=UploadStatus.UPLOADED,
        server_default=UploadStatus.UPLOADED.value,
    )
    duplicate_group_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    uploaded_by_user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retention_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    candidate_profile: Mapped["CandidateProfile"] = relationship(
        back_populates="resume_document",
        uselist=False,
        cascade="all, delete-orphan",
    )
    job: Mapped["Job"] = relationship(back_populates="resumes")
    extraction_traces: Mapped[list["ExtractionTrace"]] = relationship(
        back_populates="resume_document",
        cascade="all, delete-orphan",
    )


class ExtractionTrace(Base):
    __tablename__ = "extraction_traces"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    resume_document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resume_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stage: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    resume_document: Mapped[ResumeDocument] = relationship(back_populates="extraction_traces")
