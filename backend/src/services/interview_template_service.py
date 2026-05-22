from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models.interview_template import InterviewTemplate
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
