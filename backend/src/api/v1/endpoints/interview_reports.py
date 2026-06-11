from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models.deps import get_current_user, get_db
from src.models.interview_invitation import InterviewInvitation
from src.models.interview_session import InterviewReport, InterviewSession
from src.models.job import Job
from src.models.user_account import UserAccount
from src.schemas.interview_report import InterviewReportResponse

router = APIRouter()


@router.get("/interview-reports/{interview_session_id}", response_model=InterviewReportResponse)
def get_interview_report(
    interview_session_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    stmt = (
        select(InterviewReport)
        .join(InterviewSession, InterviewSession.id == InterviewReport.interview_session_id)
        .join(InterviewInvitation, InterviewInvitation.id == InterviewSession.interview_invitation_id)
        .join(Job, Job.id == InterviewInvitation.job_id)
        .where(
            InterviewReport.interview_session_id == interview_session_id,
            Job.owner_user_id == current_user.id,
        )
    )
    report = db.execute(stmt).scalar_one_or_none()
    if report is None:
        raise HTTPException(status_code=404, detail="Interview report not found")

    return InterviewReportResponse(
        id=str(report.id),
        interview_session_id=str(report.interview_session_id),
        interview_template_id=str(report.interview_template_id) if report.interview_template_id else None,
        summary_text=report.summary_text,
        report_payload=report.report_payload or {},
        created_at=report.created_at,
        updated_at=report.updated_at,
    )
