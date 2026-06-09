from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class PublicInterviewStartRequest(BaseModel):
    provider: str | None = None
    provider_session_id: str | None = Field(default=None, max_length=255)
    device_metadata: dict[str, Any] | None = None
    browser_metadata: dict[str, Any] | None = None
    connection_metadata: dict[str, Any] | None = None

    @field_validator("provider", "provider_session_id")
    @classmethod
    def validate_optional_trimmed_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        candidate = value.strip()
        if not candidate:
            raise ValueError("must not be blank")
        return candidate


class PublicInterviewEventIngestItem(BaseModel):
    speaker: str = Field(..., min_length=1, max_length=50)
    text: str = Field(..., min_length=1)
    offset_ms: int | str | None = None
    question_key: str | None = Field(default=None, max_length=255)
    payload: dict[str, Any] | None = None

    @field_validator("speaker", "text")
    @classmethod
    def validate_required_trimmed_string(cls, value: str) -> str:
        candidate = value.strip()
        if not candidate:
            raise ValueError("must not be blank")
        return candidate

    @field_validator("question_key")
    @classmethod
    def validate_optional_question_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        candidate = value.strip()
        if not candidate:
            raise ValueError("must not be blank")
        return candidate


class PublicInterviewEventsRequest(BaseModel):
    provider: str | None = None
    events: list[PublicInterviewEventIngestItem] = Field(..., min_length=1)

    @field_validator("provider")
    @classmethod
    def validate_optional_provider(cls, value: str | None) -> str | None:
        if value is None:
            return None
        candidate = value.strip()
        if not candidate:
            raise ValueError("must not be blank")
        return candidate


class PublicInterviewCompleteRequest(BaseModel):
    provider: str | None = None

    @field_validator("provider")
    @classmethod
    def validate_optional_provider(cls, value: str | None) -> str | None:
        if value is None:
            return None
        candidate = value.strip()
        if not candidate:
            raise ValueError("must not be blank")
        return candidate


class PublicInterviewTTSRequest(BaseModel):
    text: str = Field(..., min_length=1)

    @field_validator("text")
    @classmethod
    def validate_required_trimmed_text(cls, value: str) -> str:
        candidate = value.strip()
        if not candidate:
            raise ValueError("must not be blank")
        return candidate


class PublicInterviewInvitationPayload(BaseModel):
    id: str
    public_token: str
    status: str
    expires_at: datetime | None
    max_attempts: int
    attempt_count: int
    candidate_full_name: str | None
    completed_at: datetime | None


class PublicInterviewSessionPayload(BaseModel):
    id: str
    provider: str | None
    provider_session_id: str | None
    status: str
    started_at: datetime | None
    completed_at: datetime | None


class PublicInterviewTemplatePayload(BaseModel):
    id: str
    name: str
    language_code: str
    intro_script: str | None
    closing_script: str | None
    question_payload: dict[str, Any]


class PublicInterviewAvailabilityPayload(BaseModel):
    can_start: bool
    reason: str
    detail: str | None


class PublicInterviewStatusResponse(BaseModel):
    invitation: PublicInterviewInvitationPayload
    template: PublicInterviewTemplatePayload
    availability: PublicInterviewAvailabilityPayload


class PublicInterviewStartResponse(BaseModel):
    invitation: PublicInterviewInvitationPayload
    session: PublicInterviewSessionPayload
    template: PublicInterviewTemplatePayload


class PublicInterviewEventsResponse(BaseModel):
    accepted: bool
    stored_turns: int


class PublicInterviewCompleteResponse(BaseModel):
    invitation: PublicInterviewInvitationPayload
    session: PublicInterviewSessionPayload
