from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class InterviewInvitationCreateRequest(BaseModel):
    job_id: uuid.UUID
    candidate_profile_id: uuid.UUID
    interview_template_id: uuid.UUID | None = None
    interview_question_set_id: uuid.UUID | None = None
    expires_in_hours: int | None = Field(default=None, ge=1, le=24 * 30)
    send_email: bool = True

    @model_validator(mode="after")
    def validate_source(self):
        if bool(self.interview_template_id) == bool(self.interview_question_set_id):
            raise ValueError(
                "Provide exactly one of interview_template_id or interview_question_set_id."
            )
        return self


class InterviewInvitationResponse(BaseModel):
    id: str
    job_id: str
    candidate_profile_id: str
    candidate_full_name: str | None
    interview_template_id: str
    interview_template_name: str | None
    public_token: str
    public_url: str
    status: str
    expires_at: datetime | None
    max_attempts: int
    attempt_count: int
    latest_interview_session_id: str | None
    sent_by_user_id: str | None
    sent_at: datetime | None
    opened_at: datetime | None
    completed_at: datetime | None
    cancelled_at: datetime | None
    created_at: datetime
    updated_at: datetime


class InterviewInvitationListResponse(BaseModel):
    items: list[InterviewInvitationResponse]
    total: int
