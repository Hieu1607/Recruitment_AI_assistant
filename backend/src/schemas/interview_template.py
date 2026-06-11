from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class InterviewTemplateCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    language_code: str = Field(default="vi-VN", min_length=1, max_length=16)
    status: str = Field(default="draft", min_length=1, max_length=50)
    intro_script: str | None = None
    closing_script: str | None = None
    question_payload: dict[str, Any] = Field(default_factory=dict)
    report_rubric: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "language_code", "status")
    @classmethod
    def validate_non_blank_trimmed_fields(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("intro_script", "closing_script")
    @classmethod
    def validate_optional_non_blank_scripts(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value


class InterviewTemplateUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    language_code: str | None = Field(default=None, min_length=1, max_length=16)
    status: str | None = Field(default=None, min_length=1, max_length=50)
    intro_script: str | None = None
    closing_script: str | None = None
    question_payload: dict[str, Any] | None = None
    report_rubric: dict[str, Any] | None = None

    @field_validator("name", "language_code", "status")
    @classmethod
    def validate_optional_non_blank_trimmed_fields(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("intro_script", "closing_script")
    @classmethod
    def validate_optional_non_blank_scripts(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value


class InterviewTemplateResponse(BaseModel):
    id: str
    job_id: str
    name: str
    language_code: str
    status: str
    intro_script: str | None
    closing_script: str | None
    question_payload: dict[str, Any]
    report_rubric: dict[str, Any]
    version: int
    created_at: datetime
    updated_at: datetime


class InterviewTemplateListResponse(BaseModel):
    items: list[InterviewTemplateResponse]
    total: int
