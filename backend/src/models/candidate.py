from __future__ import annotations

import enum
import uuid
from datetime import datetime, timedelta

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.repositories.db import Base


def _enum_values(enum_cls: type[enum.Enum]) -> list[str]:
    return [str(member.value) for member in enum_cls]


class UploadStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class ParseStatus(str, enum.Enum):
    NOT_STARTED = "not_started"
    TEXT_EXTRACTED = "text_extracted"
    NORMALIZED = "normalized"
    FAILED = "failed"


class ProfileStatus(str, enum.Enum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    ARCHIVED = "archived"


class ResumeDocument(Base):
    __tablename__ = "resume_documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_provider: Mapped[str] = mapped_column(String(32), nullable=False, default="minio")
    minio_bucket: Mapped[str] = mapped_column(String(128), nullable=False)
    minio_object_key: Mapped[str] = mapped_column(String(512), nullable=False)
    storage_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False, default="application/pdf")
    upload_status: Mapped[UploadStatus] = mapped_column(
        Enum(UploadStatus, name="upload_status", values_callable=_enum_values),
        nullable=False,
        default=UploadStatus.UPLOADED,
    )
    parse_status: Mapped[ParseStatus] = mapped_column(
        Enum(ParseStatus, name="parse_status", values_callable=_enum_values),
        nullable=False,
        default=ParseStatus.NOT_STARTED,
    )
    language_detected: Mapped[str | None] = mapped_column(String(10), nullable=True)
    duplicate_group_key: Mapped[str | None] = mapped_column(String(120), nullable=True)
    uploaded_by_user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retention_expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.utcnow() + timedelta(days=365)
    )


class CandidateProfile(Base):
    __tablename__ = "candidate_profiles"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    resume_document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("resume_documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    email: Mapped[str | None] = mapped_column(String(320), nullable=True, index=True)
    location_normalized: Mapped[str | None] = mapped_column(String(255), nullable=True)
    contact: Mapped[str | None] = mapped_column(String(255), nullable=True)
    current_job_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    educated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    ever_studied_abroad: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    major: Mapped[str | None] = mapped_column(String(255), nullable=True)
    cpa: Mapped[str | None] = mapped_column(String(255), nullable=True)
    education_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    experience_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    experience_years: Mapped[float | None] = mapped_column(Numeric(4, 1), nullable=True)
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
        Enum(ProfileStatus, name="profile_status", values_callable=_enum_values),
        nullable=False,
        default=ProfileStatus.DRAFT,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )


class ExtractionTrace(Base):
    __tablename__ = "extraction_traces"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    resume_document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("resume_documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    candidate_profile_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("candidate_profiles.id", ondelete="SET NULL"), nullable=True, index=True
    )
    source_page: Mapped[int] = mapped_column(Integer, nullable=False)
    source_bbox: Mapped[dict] = mapped_column(JSONB, nullable=False)
    source_text_snippet: Mapped[str] = mapped_column(Text, nullable=False)
    mapped_field_name: Mapped[str] = mapped_column(String(120), nullable=False)
    confidence_score: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
