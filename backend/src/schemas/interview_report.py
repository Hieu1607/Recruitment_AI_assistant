from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class InterviewReportQuestionItem(BaseModel):
    question_key: str | None = Field(default=None, min_length=1)
    question_text: str = Field(..., min_length=1)
    question_transcript_turn_id: str = Field(..., min_length=1)
    question_turn_index: int = Field(..., ge=0)
    answer_text: str = Field(..., min_length=1)
    answer_transcript_turn_id: str = Field(..., min_length=1)
    answer_turn_index: int = Field(..., ge=0)
    evaluation: str = Field(..., min_length=1)


class InterviewReportSummary(BaseModel):
    candidate_overview: str = Field(..., min_length=1)
    questions: list[InterviewReportQuestionItem] = Field(..., min_length=1)
    overall_summary: str = Field(..., min_length=1)


class InterviewReportFailure(BaseModel):
    stage: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    retryable: bool


class InterviewReportTaskState(BaseModel):
    state: str = Field(..., min_length=1)
    task_id: str | None = None
    retry_count: int = Field(default=0, ge=0)


class InterviewReportPayload(BaseModel):
    status: Literal["pending", "completed", "failed"]
    summary: InterviewReportSummary | None = None
    failure: InterviewReportFailure | None = None
    task: InterviewReportTaskState | None = None

    @model_validator(mode="after")
    def validate_state_shape(self) -> "InterviewReportPayload":
        if self.status == "completed":
            if self.summary is None:
                raise ValueError("completed reports require summary")
            if self.failure is not None:
                raise ValueError("completed reports must not include failure")
        elif self.status == "failed":
            if self.failure is None:
                raise ValueError("failed reports require failure details")
        elif self.status == "pending":
            if self.task is None:
                raise ValueError("pending reports require task state")
        return self

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump()


class InterviewReportResponse(BaseModel):
    id: str
    interview_session_id: str
    interview_template_id: str | None
    summary_text: str | None
    report_payload: dict[str, Any]
    created_at: datetime
    updated_at: datetime
