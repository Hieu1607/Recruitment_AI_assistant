"""CRUD endpoints for OutreachMessage.

Route map
---------
  POST   /outreach/              create a message
  GET    /outreach/              list (filter by user / candidate / sent_status)
  POST   /outreach/bulk-send     queue up to 50 owned messages
  GET    /outreach/{id}         get single message
  PATCH  /outreach/{id}         update subject, body, or sent_status
  DELETE /outreach/{id}         delete
"""

import importlib
import json
import sys
import uuid
from datetime import datetime, timezone
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select

from src.models.candidate_profile import CandidateProfile
from src.models.deps import get_current_user, get_db
from src.models.enums import ContentSource, SentStatus
from src.models.job import Job
from src.models.oauth_identity import GMAIL_SEND_SCOPE, OAuthIdentity
from src.models.outreach import OutreachMessage
from src.models.outreach_template import OutreachTemplate
from src.models.session import SessionLocal
from src.models.user_account import UserAccount
from src.services.outreach_service import normalize_rich_message

router = APIRouter()


# ---------------------------------------------------------------------------
# Template variable rules
#
# Templates may declare a fixed set of "default variables" the recruiter fills
# in once (job_title, company_name). candidate_name / candidate_email are
# intentionally excluded here — they always auto-resolve from the candidate
# selected in New message, never from a template default.
# ---------------------------------------------------------------------------

TEMPLATE_DEFAULT_VARIABLE_KEYS = {"job_title", "company_name"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_or_404(db, model, record_id: uuid.UUID, label: str):
    obj = db.get(model, record_id)
    if obj is None:
        raise HTTPException(status_code=404, detail=f"{label} '{record_id}' not found")
    return obj


def _get_gmail_capable_identity(db, user_id: uuid.UUID) -> OAuthIdentity | None:
    identity = (
        db.execute(
            select(OAuthIdentity).where(
                OAuthIdentity.user_id == user_id,
                OAuthIdentity.provider == "google",
            )
        )
        .scalar_one_or_none()
    )
    if identity is None:
        return None
    if not identity.refresh_token_encrypted:
        return None
    if not identity.has_scope(GMAIL_SEND_SCOPE):
        return None
    return identity


def _generate_outreach_template_draft(
    *,
    brief: str,
    job: Job,
    variables_allowed: list[str],
) -> dict:
    from src.prompts.build_prompts import build_prompts
    from src.services.llm_service import LLMProvider, LLMProviderError

    prompt = build_prompts.build_outreach_template_draft_prompt(
        brief=brief,
        job_title=job.title,
        company_name=None,
        variables_allowed=variables_allowed,
    )

    try:
        response = LLMProvider().generate(prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.rsplit("```", 1)[0].strip()
        return json.loads(text)
    except (LLMProviderError, json.JSONDecodeError, Exception) as exc:
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {exc}") from exc


def _missing_template_default_variables(template: OutreachTemplate) -> list[str]:
    """Return which of the template's declared variables (job_title/company_name)
    are used in its content but have no configured default value yet."""
    used = set(template.variables_used or [])
    defaults = template.default_variables or {}
    return sorted(
        key
        for key in TEMPLATE_DEFAULT_VARIABLE_KEYS
        if key in used and not (defaults.get(key) or "").strip()
    )


def _resolve_template_render_variables(
    candidate: CandidateProfile,
    template: OutreachTemplate,
) -> dict[str, str]:
    """Auto-resolve candidate_name/candidate_email from the candidate and merge
    in the template's configured job_title/company_name defaults."""
    defaults = template.default_variables or {}
    return {
        "candidate_name": candidate.full_name or "",
        "candidate_email": candidate.email or "",
        "job_title": defaults.get("job_title", ""),
        "company_name": defaults.get("company_name", ""),
    }


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class OutreachCreateRequest(BaseModel):
    candidate_profile_id: uuid.UUID
    created_by_user_id: uuid.UUID
    content_source: ContentSource
    subject: str = Field(..., min_length=1, max_length=255)
    body_text: str = Field(..., min_length=1)
    body_html: str = Field(..., min_length=1)
    template_id: Optional[uuid.UUID] = None
    render_variables: Optional[dict] = None


class OutreachUpdateRequest(BaseModel):
    subject: Optional[str] = Field(None, min_length=1, max_length=255)
    body_text: Optional[str] = Field(None, min_length=1)
    body_html: Optional[str] = Field(None, min_length=1)
    sent_status: Optional[SentStatus] = None


class OutreachResponse(BaseModel):
    id: str
    candidate_profile_id: str
    candidate_full_name: Optional[str]
    created_by_user_id: str
    content_source: str
    subject: str
    body_text: str
    body_html: str
    template_id: Optional[str]
    render_variables: Optional[dict]
    sent_status: str
    sent_at: Optional[datetime]
    created_at: datetime


class OutreachListResponse(BaseModel):
    total: int
    items: List[OutreachResponse]


class OutreachBulkSendRequest(BaseModel):
    message_ids: list[uuid.UUID] = Field(..., min_length=1, max_length=50)


class OutreachBulkSendResult(BaseModel):
    message_id: str
    status: Literal["queued", "skipped", "failed"]
    reason: Optional[str] = None


class OutreachBulkSendResponse(BaseModel):
    queued_count: int
    skipped_count: int
    failed_count: int
    results: list[OutreachBulkSendResult]


def _validate_default_variables(value: Optional[dict]) -> Optional[dict[str, str]]:
    if value is None:
        return None
    invalid_keys = sorted(set(value.keys()) - TEMPLATE_DEFAULT_VARIABLE_KEYS)
    if invalid_keys:
        raise ValueError(
            f"default_variables only supports {sorted(TEMPLATE_DEFAULT_VARIABLE_KEYS)}; "
            f"unsupported keys: {invalid_keys}"
        )
    return {str(k): str(v) for k, v in value.items()}


class OutreachTemplateCreateRequest(BaseModel):
    created_by_user_id: uuid.UUID
    job_id: Optional[uuid.UUID] = None
    name: str = Field(..., min_length=1, max_length=255)
    content_source: ContentSource = ContentSource.TEMPLATE
    subject_template: str = Field(..., min_length=1, max_length=255)
    body_text_template: str = Field(..., min_length=1)
    body_html_template: str = Field(..., min_length=1)
    editor_json: Optional[dict] = None
    variables_used: Optional[list[str]] = None
    default_variables: Optional[dict[str, str]] = None

    @field_validator("default_variables")
    @classmethod
    def _check_default_variables(cls, value: Optional[dict]) -> Optional[dict[str, str]]:
        return _validate_default_variables(value)


class OutreachTemplateUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    subject_template: Optional[str] = Field(None, min_length=1, max_length=255)
    body_text_template: Optional[str] = Field(None, min_length=1)
    body_html_template: Optional[str] = Field(None, min_length=1)
    editor_json: Optional[dict] = None
    variables_used: Optional[list[str]] = None
    default_variables: Optional[dict[str, str]] = None

    @field_validator("default_variables")
    @classmethod
    def _check_default_variables(cls, value: Optional[dict]) -> Optional[dict[str, str]]:
        return _validate_default_variables(value)


class OutreachTemplateGenerateRequest(BaseModel):
    job_id: uuid.UUID
    brief: str = Field(..., min_length=1)
    variables_allowed: list[str] = Field(default_factory=list)


class OutreachTemplateResponse(BaseModel):
    id: str
    created_by_user_id: str
    job_id: Optional[str]
    name: str
    content_source: str
    subject_template: str
    body_text_template: str
    body_html_template: str
    editor_json: Optional[dict]
    variables_used: list[str]
    default_variables: dict[str, str]
    created_at: datetime
    updated_at: datetime


class OutreachTemplateListResponse(BaseModel):
    total: int
    items: list[OutreachTemplateResponse]


class OutreachTemplateGenerateResponse(BaseModel):
    subject: str
    body_text: str
    body_html: str
    variables_used: list[str]


# ---------------------------------------------------------------------------
# Serialiser
# ---------------------------------------------------------------------------

def _ser(m: OutreachMessage) -> OutreachResponse:
    return OutreachResponse(
        id=str(m.id),
        candidate_profile_id=str(m.candidate_profile_id),
        candidate_full_name=m.candidate_profile.full_name if m.candidate_profile else None,
        created_by_user_id=str(m.created_by_user_id),
        content_source=m.content_source.value,
        subject=m.subject,
        body_text=m.body_text,
        body_html=m.body_html,
        template_id=str(m.template_id) if m.template_id else None,
        render_variables=m.render_variables,
        sent_status=m.sent_status.value,
        sent_at=m.sent_at,
        created_at=m.created_at,
    )


def _ser_template(template: OutreachTemplate) -> OutreachTemplateResponse:
    return OutreachTemplateResponse(
        id=str(template.id),
        created_by_user_id=str(template.created_by_user_id),
        job_id=str(template.job_id) if template.job_id else None,
        name=template.name,
        content_source=template.content_source.value,
        subject_template=template.subject_template,
        body_text_template=template.body_text_template,
        body_html_template=template.body_html_template,
        editor_json=template.editor_json,
        variables_used=template.variables_used or [],
        default_variables=template.default_variables or {},
        created_at=template.created_at,
        updated_at=template.updated_at,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/", response_model=OutreachResponse, status_code=201)
def create_message(body: OutreachCreateRequest):
    db = SessionLocal()
    try:
        candidate = _get_or_404(db, CandidateProfile, body.candidate_profile_id, "CandidateProfile")

        # When composed from a template, candidate_name/candidate_email always
        # auto-resolve from the selected candidate, and job_title/company_name
        # come from the template's configured defaults — the server is
        # authoritative here regardless of what the client sends.
        render_variables = body.render_variables
        if body.template_id is not None:
            template = _get_or_404(db, OutreachTemplate, body.template_id, "OutreachTemplate")
            missing = _missing_template_default_variables(template)
            if missing:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "missing_default_variables",
                        "missing": missing,
                        "message": (
                            "Template uses "
                            + ", ".join(f"{{{{{key}}}}}" for key in missing)
                            + " but no default value is configured yet. "
                            "Configure it in the Templates workspace first."
                        ),
                    },
                )
            render_variables = _resolve_template_render_variables(candidate, template)

        msg = OutreachMessage(
            candidate_profile_id=body.candidate_profile_id,
            created_by_user_id=body.created_by_user_id,
            content_source=body.content_source,
            subject=body.subject,
            body_text=normalize_rich_message(body_text=body.body_text, body_html=body.body_html)[0],
            body_html=normalize_rich_message(body_text=body.body_text, body_html=body.body_html)[1],
            template_id=body.template_id,
            render_variables=render_variables,
            sent_status=SentStatus.NOT_SENT,
        )
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return _ser(msg)
    finally:
        db.close()


@router.get("/", response_model=OutreachListResponse)
def list_messages(
    created_by_user_id: Optional[uuid.UUID] = Query(None, description="Filter by creator user"),
    candidate_profile_id: Optional[uuid.UUID] = Query(None, description="Filter by candidate"),
    sent_status: Optional[SentStatus] = Query(None, description="Filter by sent status"),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    db = SessionLocal()
    try:
        query = db.query(OutreachMessage)

        if created_by_user_id is not None:
            query = query.filter(OutreachMessage.created_by_user_id == created_by_user_id)
        if candidate_profile_id is not None:
            query = query.filter(OutreachMessage.candidate_profile_id == candidate_profile_id)
        if sent_status is not None:
            query = query.filter(OutreachMessage.sent_status == sent_status)

        total = query.count()
        rows = query.order_by(OutreachMessage.created_at.desc()).offset(offset).limit(limit).all()

        return OutreachListResponse(total=total, items=[_ser(r) for r in rows])
    finally:
        db.close()


@router.post("/templates", response_model=OutreachTemplateResponse, status_code=201)
def create_template(body: OutreachTemplateCreateRequest):
    db = SessionLocal()
    try:
        body_text, body_html = normalize_rich_message(
            body_text=body.body_text_template,
            body_html=body.body_html_template,
        )
        template = OutreachTemplate(
            created_by_user_id=body.created_by_user_id,
            job_id=body.job_id,
            name=body.name,
            content_source=body.content_source,
            subject_template=body.subject_template.strip(),
            body_text_template=body_text,
            body_html_template=body_html,
            editor_json=body.editor_json,
            variables_used=body.variables_used or [],
            default_variables=body.default_variables or {},
        )
        db.add(template)
        db.commit()
        db.refresh(template)
        return _ser_template(template)
    finally:
        db.close()


@router.post("/templates/generate-draft", response_model=OutreachTemplateGenerateResponse)
def generate_template_draft(body: OutreachTemplateGenerateRequest):
    if not body.brief.strip():
        raise HTTPException(status_code=422, detail="brief must not be empty")

    db = SessionLocal()
    try:
        job = _get_or_404(db, Job, body.job_id, "Job")
        payload = _generate_outreach_template_draft(
            brief=body.brief,
            job=job,
            variables_allowed=body.variables_allowed,
        )
        body_text, body_html = normalize_rich_message(
            body_text=(payload.get("body_text") or "").strip(),
            body_html=(payload.get("body_html") or "").strip(),
        )
        return OutreachTemplateGenerateResponse(
            subject=(payload.get("subject") or "").strip(),
            body_text=body_text,
            body_html=body_html,
            variables_used=list(payload.get("variables_used") or []),
        )
    finally:
        db.close()


@router.get("/templates", response_model=OutreachTemplateListResponse)
def list_templates(
    created_by_user_id: Optional[uuid.UUID] = Query(None),
    job_id: Optional[uuid.UUID] = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    db = SessionLocal()
    try:
        query = db.query(OutreachTemplate)
        if created_by_user_id is not None:
            query = query.filter(OutreachTemplate.created_by_user_id == created_by_user_id)
        if job_id is not None:
            query = query.filter(OutreachTemplate.job_id == job_id)
        total = query.count()
        rows = query.order_by(OutreachTemplate.updated_at.desc()).offset(offset).limit(limit).all()
        return OutreachTemplateListResponse(total=total, items=[_ser_template(row) for row in rows])
    finally:
        db.close()


@router.get("/templates/{template_id}", response_model=OutreachTemplateResponse)
def get_template(template_id: uuid.UUID):
    db = SessionLocal()
    try:
        template = _get_or_404(db, OutreachTemplate, template_id, "OutreachTemplate")
        return _ser_template(template)
    finally:
        db.close()


@router.patch("/templates/{template_id}", response_model=OutreachTemplateResponse)
def update_template(template_id: uuid.UUID, body: OutreachTemplateUpdateRequest):
    db = SessionLocal()
    try:
        template = _get_or_404(db, OutreachTemplate, template_id, "OutreachTemplate")
        if body.name is not None:
            template.name = body.name.strip()
        if body.subject_template is not None:
            template.subject_template = body.subject_template.strip()
        if body.body_text_template is not None or body.body_html_template is not None:
            normalized_text, normalized_html = normalize_rich_message(
                body_text=body.body_text_template if body.body_text_template is not None else template.body_text_template,
                body_html=body.body_html_template if body.body_html_template is not None else template.body_html_template,
            )
            template.body_text_template = normalized_text
            template.body_html_template = normalized_html
        if body.editor_json is not None:
            template.editor_json = body.editor_json
        if body.variables_used is not None:
            template.variables_used = body.variables_used
        if body.default_variables is not None:
            template.default_variables = body.default_variables
        db.commit()
        db.refresh(template)
        return _ser_template(template)
    finally:
        db.close()


@router.post("/bulk-send", response_model=OutreachBulkSendResponse, status_code=status.HTTP_202_ACCEPTED)
def bulk_send_messages(
    body: OutreachBulkSendRequest,
    db=Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    if _get_gmail_capable_identity(db, current_user.id) is None:
        raise HTTPException(status_code=409, detail="gmail_not_connected")

    worker_package = sys.modules.get("worker")
    tasks_module = getattr(worker_package, "tasks", None) if worker_package is not None else None
    if tasks_module is None:
        tasks_module = importlib.import_module("worker.tasks")

    requested_ids = list(dict.fromkeys(body.message_ids))
    owned_messages = (
        db.execute(
            select(OutreachMessage).where(
                OutreachMessage.id.in_(requested_ids),
                OutreachMessage.created_by_user_id == current_user.id,
            )
        )
        .scalars()
        .all()
    )
    messages_by_id = {message.id: message for message in owned_messages}
    results: list[OutreachBulkSendResult] = []

    for message_id in requested_ids:
        message = messages_by_id.get(message_id)
        if message is None:
            results.append(
                OutreachBulkSendResult(
                    message_id=str(message_id),
                    status="skipped",
                    reason="Message was not found.",
                )
            )
            continue
        if message.sent_status == SentStatus.SENT:
            results.append(
                OutreachBulkSendResult(
                    message_id=str(message_id),
                    status="skipped",
                    reason="Message has already been sent.",
                )
            )
            continue
        if not message.candidate_profile or not message.candidate_profile.email:
            message.sent_status = SentStatus.FAILED
            results.append(
                OutreachBulkSendResult(
                    message_id=str(message_id),
                    status="failed",
                    reason="Candidate has no email address.",
                )
            )
            continue

        try:
            if message.sent_status == SentStatus.FAILED:
                message.sent_status = SentStatus.NOT_SENT
            tasks_module.send_outreach_email.delay(str(message.id))
            results.append(OutreachBulkSendResult(message_id=str(message_id), status="queued"))
        except Exception:
            message.sent_status = SentStatus.FAILED
            results.append(
                OutreachBulkSendResult(
                    message_id=str(message_id),
                    status="failed",
                    reason="Email could not be queued.",
                )
            )

    db.commit()
    return OutreachBulkSendResponse(
        queued_count=sum(result.status == "queued" for result in results),
        skipped_count=sum(result.status == "skipped" for result in results),
        failed_count=sum(result.status == "failed" for result in results),
        results=results,
    )


@router.get("/{message_id}", response_model=OutreachResponse)
def get_message(message_id: uuid.UUID):
    db = SessionLocal()
    try:
        msg = _get_or_404(db, OutreachMessage, message_id, "OutreachMessage")
        return _ser(msg)
    finally:
        db.close()


@router.patch("/{message_id}", response_model=OutreachResponse)
def update_message(message_id: uuid.UUID, body: OutreachUpdateRequest):
    db = SessionLocal()
    try:
        msg = _get_or_404(db, OutreachMessage, message_id, "OutreachMessage")

        if body.subject is not None:
            msg.subject = body.subject
        if body.body_text is not None or body.body_html is not None:
            body_text, body_html = normalize_rich_message(
                body_text=body.body_text if body.body_text is not None else msg.body_text,
                body_html=body.body_html if body.body_html is not None else msg.body_html,
            )
            msg.body_text = body_text
            msg.body_html = body_html
        if body.sent_status is not None:
            msg.sent_status = body.sent_status
            if body.sent_status == SentStatus.SENT and msg.sent_at is None:
                msg.sent_at = datetime.now(timezone.utc)

        db.commit()
        db.refresh(msg)
        return _ser(msg)
    finally:
        db.close()


@router.post("/{message_id}/send", response_model=OutreachResponse, status_code=status.HTTP_202_ACCEPTED)
def send_message(
    message_id: uuid.UUID,
    db=Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
):
    msg = _get_or_404(db, OutreachMessage, message_id, "OutreachMessage")
    if msg.created_by_user_id != current_user.id:
        raise HTTPException(status_code=404, detail=f"OutreachMessage '{message_id}' not found")
    if msg.sent_status == SentStatus.SENT:
        return _ser(msg)
    if _get_gmail_capable_identity(db, current_user.id) is None:
        raise HTTPException(status_code=409, detail="gmail_not_connected")
    if not msg.candidate_profile or not msg.candidate_profile.email:
        msg.sent_status = SentStatus.FAILED
        db.commit()
        db.refresh(msg)
        return _ser(msg)

    worker_package = sys.modules.get("worker")
    tasks_module = getattr(worker_package, "tasks", None) if worker_package is not None else None
    if tasks_module is None:
        tasks_module = importlib.import_module("worker.tasks")
    tasks_module.send_outreach_email.delay(str(msg.id))
    return _ser(msg)


@router.delete("/{message_id}", status_code=204)
def delete_message(message_id: uuid.UUID):
    db = SessionLocal()
    try:
        msg = _get_or_404(db, OutreachMessage, message_id, "OutreachMessage")
        db.delete(msg)
        db.commit()
    finally:
        db.close()
