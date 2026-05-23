from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class InterviewReportEvidenceItem(BaseModel):
    transcript_turn_id: str = Field(..., min_length=1)
    turn_index: int = Field(..., ge=0)
    question_key: str | None = Field(default=None, min_length=1)
    speaker_role: str = Field(..., min_length=1)
    transcript_text: str = Field(..., min_length=1)


class InterviewReportCompetency(BaseModel):
    name: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    evidence: list[InterviewReportEvidenceItem] = Field(..., min_length=1)


class InterviewReportSummary(BaseModel):
    candidate_overview: str = Field(..., min_length=1)
    competencies: list[InterviewReportCompetency] = Field(..., min_length=1)
    communication_summary: str = Field(..., min_length=1)
    follow_up_topics: list[str] = Field(default_factory=list)
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
