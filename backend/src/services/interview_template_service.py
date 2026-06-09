from __future__ import annotations

import uuid

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models.interview_template import InterviewTemplate
from src.models.job_matching import InterviewQuestionSet, JobDescription
from src.schemas.interview_template import (
    InterviewTemplateCreateRequest,
    InterviewTemplateResponse,
    InterviewTemplateUpdateRequest,
)
from src.services.job_scope import get_current_user_owned_job, get_user_owned_interview_template


def _normalize_optional_script(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def materialize_question_set_template(
    db: Session,
    *,
    job_id: uuid.UUID,
    question_set: InterviewQuestionSet,
) -> InterviewTemplate:
    jd_title = question_set.job_description.title if question_set.job_description is not None else "Interview"
    candidate_name = question_set.candidate_profile.full_name if question_set.candidate_profile is not None else None
    name_parts = [jd_title.strip() if jd_title else "Interview"]
    if candidate_name:
        name_parts.append(candidate_name.strip())
    template = InterviewTemplate(
        job_id=job_id,
        name="Question Set · " + " · ".join(part for part in name_parts if part),
        language_code="vi-VN",
        status="active",
        question_payload=question_set.question_payload or {},
        report_rubric={},
    )
    db.add(template)
    db.flush()
    return template


def get_job_scoped_interview_question_set(
    db: Session,
    *,
    user_id: uuid.UUID,
    job_id: uuid.UUID,
    question_set_id: uuid.UUID,
) -> InterviewQuestionSet:
    get_current_user_owned_job(db, user_id, job_id)
    question_set = (
        db.execute(
            select(InterviewQuestionSet)
            .join(JobDescription, InterviewQuestionSet.job_description_id == JobDescription.id)
            .where(
                InterviewQuestionSet.id == question_set_id,
                JobDescription.job_id == job_id,
            )
        )
        .scalars()
        .first()
    )
    if question_set is None:
        raise HTTPException(status_code=404, detail="Interview question set not found for this job")
    return question_set


def serialize_interview_template(template: InterviewTemplate) -> InterviewTemplateResponse:
    return InterviewTemplateResponse(
        id=str(template.id),
        job_id=str(template.job_id),
        name=template.name,
        language_code=template.language_code,
        status=template.status,
        intro_script=template.intro_script,
        closing_script=template.closing_script,
        question_payload=template.question_payload or {},
        report_rubric=template.report_rubric or {},
        version=template.version,
        created_at=template.created_at,
        updated_at=template.updated_at,
    )


def create_interview_template(
    db: Session,
    *,
    user_id: uuid.UUID,
    job_id: uuid.UUID,
    body: InterviewTemplateCreateRequest,
) -> InterviewTemplate:
    get_current_user_owned_job(db, user_id, job_id)
    template = InterviewTemplate(
        job_id=job_id,
        name=body.name.strip(),
        language_code=body.language_code.strip(),
        status=body.status.strip(),
        intro_script=_normalize_optional_script(body.intro_script),
        closing_script=_normalize_optional_script(body.closing_script),
        question_payload=body.question_payload,
        report_rubric=body.report_rubric,
    )
    db.add(template)
    db.commit()
    db.refresh(template)
    return template


def list_interview_templates(db: Session, *, user_id: uuid.UUID, job_id: uuid.UUID) -> list[InterviewTemplate]:
    get_current_user_owned_job(db, user_id, job_id)
    return (
        db.execute(
            select(InterviewTemplate)
            .where(InterviewTemplate.job_id == job_id)
            .order_by(InterviewTemplate.updated_at.desc(), InterviewTemplate.created_at.desc())
        )
        .scalars()
        .all()
    )


def get_interview_template(db: Session, *, user_id: uuid.UUID, template_id: uuid.UUID) -> InterviewTemplate:
    return get_user_owned_interview_template(db, user_id, template_id)


def delete_interview_template(db: Session, *, user_id: uuid.UUID, template_id: uuid.UUID) -> None:
    template = get_user_owned_interview_template(db, user_id, template_id)
    if template.invitations:
        raise HTTPException(status_code=409, detail="Interview template is already in use")
    db.delete(template)
    db.commit()


def update_interview_template(
    db: Session,
    *,
    user_id: uuid.UUID,
    template_id: uuid.UUID,
    body: InterviewTemplateUpdateRequest,
) -> InterviewTemplate:
    template = get_user_owned_interview_template(db, user_id, template_id)
    should_increment_version = False

    if body.name is not None:
        template.name = body.name.strip()
    if body.language_code is not None:
        template.language_code = body.language_code.strip()
    if body.status is not None:
        template.status = body.status.strip()
    normalized_intro_script = _normalize_optional_script(body.intro_script)
    normalized_closing_script = _normalize_optional_script(body.closing_script)
    if "intro_script" in body.model_fields_set and normalized_intro_script != template.intro_script:
        template.intro_script = normalized_intro_script
        should_increment_version = True
    if "closing_script" in body.model_fields_set and normalized_closing_script != template.closing_script:
        template.closing_script = normalized_closing_script
        should_increment_version = True
    if body.question_payload is not None and body.question_payload != template.question_payload:
        template.question_payload = body.question_payload
        should_increment_version = True
    if body.report_rubric is not None and body.report_rubric != template.report_rubric:
        template.report_rubric = body.report_rubric
        should_increment_version = True

    if should_increment_version:
        template.version += 1

    db.commit()
    db.refresh(template)
    return template
