from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Enum as SqlEnum, ForeignKey, JSON, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.enums import ContentSource


_ENUM_VALUES = lambda enum_cls: [item.value for item in enum_cls]

if TYPE_CHECKING:
    from src.models.job import Job
    from src.models.outreach import OutreachMessage


class OutreachTemplate(Base):
    __tablename__ = "outreach_templates"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_by_user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    content_source: Mapped[ContentSource] = mapped_column(
        SqlEnum(ContentSource, name="content_source_enum", values_callable=_ENUM_VALUES),
        nullable=False,
    )
    subject_template: Mapped[str] = mapped_column(String(255), nullable=False)
    body_text_template: Mapped[str] = mapped_column(Text, nullable=False)
    body_html_template: Mapped[str] = mapped_column(Text, nullable=False)
    editor_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    variables_used: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    job: Mapped["Job | None"] = relationship()
    outreach_messages: Mapped[list["OutreachMessage"]] = relationship(back_populates="template")
