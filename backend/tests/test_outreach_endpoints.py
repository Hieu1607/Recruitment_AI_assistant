import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
import uuid

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.api.v1.endpoints import outreach as outreach_module  # noqa: E402
from src.api.v1.endpoints.outreach import (  # noqa: E402
    OutreachCreateRequest,
    OutreachTemplateCreateRequest,
    OutreachTemplateGenerateRequest,
    OutreachTemplateUpdateRequest,
    OutreachUpdateRequest,
    create_message,
    create_template,
    delete_message,
    generate_template_draft,
    get_template,
    get_message,
    list_messages,
    list_templates,
    send_message,
    update_template,
    update_message,
)
from src.models.base import Base  # noqa: E402
from src.models.candidate_profile import CandidateProfile  # noqa: E402
from src.models.enums import ContentSource, ProfileStatus, SentStatus, UploadStatus, UserStatus  # noqa: E402
from src.models.job import Job  # noqa: E402
from src.models.oauth_identity import OAuthIdentity  # noqa: E402
from src.models.resume_document import ResumeDocument  # noqa: E402
from src.models.user_account import UserAccount  # noqa: E402


def _load_worker_tasks_module():
    worker_package = sys.modules.get("worker")
    if worker_package is None:
        worker_package = ModuleType("worker")
        sys.modules["worker"] = worker_package
    worker_package.__path__ = [str(BACKEND_ROOT / "worker")]

    if "worker.celery_app" not in sys.modules:
        celery_app_module = ModuleType("worker.celery_app")

        class _CeleryAppStub:
            @staticmethod
            def task(*args, **kwargs):
                def decorator(fn):
                    if kwargs.get("bind"):
                        fn.run = lambda *run_args, **run_kwargs: fn(
                            SimpleNamespace(
                                request=SimpleNamespace(retries=0, id="test-task-id"),
                                max_retries=0,
                                retry=lambda exc=None: (_ for _ in ()).throw(exc),
                            ),
                            *run_args,
                            **run_kwargs,
                        )
                    else:
                        fn.run = fn
                    return fn

                return decorator

        celery_app_module.celery_app = _CeleryAppStub()
        sys.modules["worker.celery_app"] = celery_app_module

    module_name = "worker.tasks"
    module_path = BACKEND_ROOT / "worker" / "tasks.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _create_test_tables(engine):
    tables = [
        Base.metadata.tables["user_accounts"],
        Base.metadata.tables["jobs"],
        Base.metadata.tables["resume_documents"],
        Base.metadata.tables["candidate_profiles"],
        Base.metadata.tables["oauth_identities"],
        Base.metadata.tables["outreach_messages"],
        Base.metadata.tables["outreach_templates"],
    ]
    Base.metadata.create_all(engine, tables=tables)


@pytest.fixture()
def db_session_factory(monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    _create_test_tables(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    monkeypatch.setattr(outreach_module, "SessionLocal", factory)
    return factory


@pytest.fixture()
def seeded_data(db_session_factory):
    db: Session = db_session_factory()
    try:
        user = UserAccount(
            email="owner@example.com",
            display_name="Owner",
            password_hash=None,
            status=UserStatus.ACTIVE,
        )
        db.add(user)
        db.flush()

        job = Job(owner_user_id=user.id, title="Platform Engineer", status="active")
        db.add(job)
        db.flush()

        resume = ResumeDocument(
            original_file_name="candidate.pdf",
            storage_uri="s3://bucket/resumes/candidate.pdf",
            upload_status=UploadStatus.PROCESSED,
            job_id=job.id,
            uploaded_by_user_id=user.id,
            retention_expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
        )
        db.add(resume)
        db.flush()

        candidate = CandidateProfile(
            resume_document_id=resume.id,
            full_name="Candidate One",
            email="candidate@example.com",
            profile_status=ProfileStatus.REVIEWED,
        )
        db.add(candidate)
        db.commit()
        db.refresh(user)
        db.refresh(job)
        db.refresh(candidate)
        return {"user": user, "job": job, "candidate": candidate}
    finally:
        db.close()


def test_outreach_create_list_update_and_delete(db_session_factory, seeded_data):
    user = seeded_data["user"]
    candidate = seeded_data["candidate"]

    created = create_message(
        OutreachCreateRequest(
            candidate_profile_id=candidate.id,
            created_by_user_id=user.id,
            content_source=ContentSource.AI_DRAFT,
            subject="Intro call",
            body_text="Would you be open to a short intro call?",
            body_html="<p>Would you be open to a short intro call?</p>",
        )
    )

    listed = list_messages(
        created_by_user_id=user.id,
        candidate_profile_id=candidate.id,
        sent_status=SentStatus.NOT_SENT,
        offset=0,
        limit=50,
    )
    fetched = get_message(uuid.UUID(created.id))
    updated = update_message(
        uuid.UUID(created.id),
        OutreachUpdateRequest(sent_status=SentStatus.SENT),
    )

    assert created.sent_status == SentStatus.NOT_SENT.value
    assert created.candidate_full_name == "Candidate One"
    assert listed.total == 1
    assert listed.items[0].id == created.id
    assert fetched.subject == "Intro call"
    assert fetched.body_text == "Would you be open to a short intro call?"
    assert fetched.body_html == "<p>Would you be open to a short intro call?</p>"
    assert updated.sent_status == SentStatus.SENT.value
    assert updated.sent_at is not None

    delete_message(uuid.UUID(created.id))

    after_delete = list_messages(
        created_by_user_id=user.id,
        candidate_profile_id=None,
        sent_status=None,
        offset=0,
        limit=50,
    )
    assert after_delete.total == 0


def test_send_outreach_returns_gmail_not_connected_when_google_capability_missing(
    db_session_factory,
    seeded_data,
):
    db: Session = db_session_factory()
    try:
        message = outreach_module.OutreachMessage(
            candidate_profile_id=seeded_data["candidate"].id,
            created_by_user_id=seeded_data["user"].id,
            content_source=ContentSource.AI_DRAFT,
            subject="Intro call",
            body_text="Would you be open to a short intro call?",
            body_html="<p>Would you be open to a short intro call?</p>",
            sent_status=SentStatus.NOT_SENT,
        )
        db.add(message)
        db.commit()
        db.refresh(message)
        with pytest.raises(HTTPException) as exc_info:
            send_message(message.id, db=db, current_user=seeded_data["user"])

        assert exc_info.value.status_code == 409
        assert exc_info.value.detail == "gmail_not_connected"
    finally:
        db.close()


def test_send_outreach_task_returns_gmail_not_connected_when_google_capability_missing(
    db_session_factory,
    seeded_data,
    monkeypatch,
):
    tasks_module = _load_worker_tasks_module()
    import src.models.session as session_module

    monkeypatch.setattr(session_module, "SessionLocal", db_session_factory)

    db: Session = db_session_factory()
    try:
        message = outreach_module.OutreachMessage(
            candidate_profile_id=seeded_data["candidate"].id,
            created_by_user_id=seeded_data["user"].id,
            content_source=ContentSource.AI_DRAFT,
            subject="Intro call",
            body_text="Would you be open to a short intro call?",
            body_html="<p>Would you be open to a short intro call?</p>",
            sent_status=SentStatus.NOT_SENT,
        )
        db.add(message)
        db.add(
            OAuthIdentity(
                user_id=seeded_data["user"].id,
                provider="google",
                provider_subject="google-subject",
                email=seeded_data["user"].email,
                refresh_token_encrypted="encrypted-refresh",
                scope="openid email profile",
            )
        )
        db.commit()
        db.refresh(message)
    finally:
        db.close()


def test_outreach_template_crud(db_session_factory, seeded_data):
    user = seeded_data["user"]

    created = create_template(
        OutreachTemplateCreateRequest(
            created_by_user_id=user.id,
            name="Warm intro",
            content_source=ContentSource.TEMPLATE,
            subject_template="Opportunity at {{company_name}}",
            body_text_template="Hi {{candidate_name}}, let's discuss {{job_title}}.",
            body_html_template="<p>Hi <strong>{{candidate_name}}</strong>, let's discuss {{job_title}}.</p>",
            editor_json={"type": "doc", "content": []},
            variables_used=["candidate_name", "company_name", "job_title"],
        )
    )

    listed = list_templates(created_by_user_id=user.id, job_id=None, offset=0, limit=50)
    fetched = get_template(uuid.UUID(created.id))
    updated = update_template(
        uuid.UUID(created.id),
        OutreachTemplateUpdateRequest(name="Warm intro v2"),
    )

    assert created.name == "Warm intro"
    assert created.body_html_template.startswith("<p>")
    assert listed.total == 1
    assert listed.items[0].id == created.id
    assert fetched.subject_template == "Opportunity at {{company_name}}"
    assert updated.name == "Warm intro v2"


def test_create_message_allows_manual_content_source(db_session_factory, seeded_data):
    user = seeded_data["user"]
    candidate = seeded_data["candidate"]

    created = create_message(
        OutreachCreateRequest(
            candidate_profile_id=candidate.id,
            created_by_user_id=user.id,
            content_source="manual",
            subject="Checking in",
            body_text="Hello there",
            body_html="<p>Hello there</p>",
        )
    )

    assert created.content_source == "manual"


def test_generate_template_draft_returns_subject_and_body(db_session_factory, seeded_data, monkeypatch):
    monkeypatch.setattr(
        outreach_module,
        "_generate_outreach_template_draft",
        lambda **kwargs: {
            "subject": "Opportunity at {{company_name}}",
            "body_text": "Hi {{candidate_name}}",
            "body_html": "<p>Hi {{candidate_name}}</p>",
            "variables_used": ["candidate_name", "company_name"],
        },
    )

    generated = generate_template_draft(
        OutreachTemplateGenerateRequest(
            job_id=seeded_data["job"].id,
            brief="Write a short recruiter intro email",
            variables_allowed=["candidate_name", "company_name", "job_title"],
        )
    )

    assert generated.subject == "Opportunity at {{company_name}}"
    assert generated.body_text == "Hi {{candidate_name}}"
    assert generated.body_html == "<p>Hi {{candidate_name}}</p>"
    assert generated.variables_used == ["candidate_name", "company_name"]


def test_generate_template_draft_rejects_blank_brief(db_session_factory, seeded_data):
    with pytest.raises(HTTPException) as exc_info:
        generate_template_draft(
            OutreachTemplateGenerateRequest(
                job_id=seeded_data["job"].id,
                brief="   ",
                variables_allowed=["candidate_name"],
            )
        )

    assert exc_info.value.status_code == 422


def test_generate_template_draft_returns_404_for_missing_job(db_session_factory):
    with pytest.raises(HTTPException) as exc_info:
        generate_template_draft(
            OutreachTemplateGenerateRequest(
                job_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                brief="Need an email",
                variables_allowed=["candidate_name"],
            )
        )

    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Template default variables (Configure Variables)
# ---------------------------------------------------------------------------

def test_template_save_and_load_roundtrips_default_variables(db_session_factory, seeded_data):
    user = seeded_data["user"]

    created = create_template(
        OutreachTemplateCreateRequest(
            created_by_user_id=user.id,
            name="Warm intro",
            content_source=ContentSource.TEMPLATE,
            subject_template="Opportunity at {{company_name}}",
            body_text_template="Hi {{candidate_name}}, let's discuss {{job_title}} at {{company_name}}.",
            body_html_template="<p>Hi {{candidate_name}}, let's discuss {{job_title}} at {{company_name}}.</p>",
            variables_used=["candidate_name", "company_name", "job_title"],
            default_variables={"job_title": "Backend Engineer", "company_name": "Acme Corp"},
        )
    )

    assert created.default_variables == {"job_title": "Backend Engineer", "company_name": "Acme Corp"}

    fetched = get_template(uuid.UUID(created.id))
    assert fetched.default_variables == {"job_title": "Backend Engineer", "company_name": "Acme Corp"}

    updated = update_template(
        uuid.UUID(created.id),
        OutreachTemplateUpdateRequest(default_variables={"job_title": "Staff Engineer"}),
    )
    assert updated.default_variables == {"job_title": "Staff Engineer"}

    refetched = get_template(uuid.UUID(created.id))
    assert refetched.default_variables == {"job_title": "Staff Engineer"}


def test_template_defaults_to_empty_dict_when_not_provided(db_session_factory, seeded_data):
    user = seeded_data["user"]

    created = create_template(
        OutreachTemplateCreateRequest(
            created_by_user_id=user.id,
            name="No defaults yet",
            content_source=ContentSource.TEMPLATE,
            subject_template="Hi {{candidate_name}}",
            body_text_template="Hi {{candidate_name}}",
            body_html_template="<p>Hi {{candidate_name}}</p>",
            variables_used=["candidate_name"],
        )
    )

    assert created.default_variables == {}


@pytest.mark.parametrize("disallowed_key", ["candidate_name", "candidate_email", "something_else"])
def test_template_rejects_disallowed_default_variable_keys(disallowed_key):
    with pytest.raises(ValidationError):
        OutreachTemplateCreateRequest(
            created_by_user_id=uuid.uuid4(),
            name="Bad template",
            subject_template="Hi",
            body_text_template="Hi",
            body_html_template="<p>Hi</p>",
            default_variables={disallowed_key: "value"},
        )

    with pytest.raises(ValidationError):
        OutreachTemplateUpdateRequest(default_variables={disallowed_key: "value"})


# ---------------------------------------------------------------------------
# Message creation from template: auto-resolve + merge render_variables
# ---------------------------------------------------------------------------

def _create_configured_template(user_id, *, default_variables):
    return create_template(
        OutreachTemplateCreateRequest(
            created_by_user_id=user_id,
            name="Warm intro",
            content_source=ContentSource.TEMPLATE,
            subject_template="Opportunity at {{company_name}}",
            body_text_template="Hi {{candidate_name}}, let's discuss the {{job_title}} role at {{company_name}}.",
            body_html_template=(
                "<p>Hi {{candidate_name}}, let's discuss the {{job_title}} role at {{company_name}}.</p>"
            ),
            variables_used=["candidate_name", "company_name", "job_title"],
            default_variables=default_variables,
        )
    )


def test_create_message_from_template_merges_render_variables(db_session_factory, seeded_data):
    user = seeded_data["user"]
    candidate = seeded_data["candidate"]

    template = _create_configured_template(
        user.id,
        default_variables={"job_title": "Backend Engineer", "company_name": "Acme Corp"},
    )

    created = create_message(
        OutreachCreateRequest(
            candidate_profile_id=candidate.id,
            created_by_user_id=user.id,
            content_source=ContentSource.TEMPLATE,
            subject="Opportunity at Acme Corp",
            body_text="Hi Candidate One, let's discuss the Backend Engineer role at Acme Corp.",
            body_html="<p>Hi Candidate One, let's discuss the Backend Engineer role at Acme Corp.</p>",
            template_id=uuid.UUID(template.id),
        )
    )

    assert created.render_variables == {
        "candidate_name": "Candidate One",
        "candidate_email": "candidate@example.com",
        "job_title": "Backend Engineer",
        "company_name": "Acme Corp",
    }


def test_create_message_from_template_server_overrides_client_render_variables(db_session_factory, seeded_data):
    """Server is authoritative: candidate_name/candidate_email always come from
    the selected candidate, job_title/company_name always come from the
    template defaults — never from whatever the client happens to send."""
    user = seeded_data["user"]
    candidate = seeded_data["candidate"]

    template = _create_configured_template(
        user.id,
        default_variables={"job_title": "Backend Engineer", "company_name": "Acme Corp"},
    )

    created = create_message(
        OutreachCreateRequest(
            candidate_profile_id=candidate.id,
            created_by_user_id=user.id,
            content_source=ContentSource.TEMPLATE,
            subject="Opportunity at Acme Corp",
            body_text="Hi there",
            body_html="<p>Hi there</p>",
            template_id=uuid.UUID(template.id),
            render_variables={
                "candidate_name": "Someone Else",
                "job_title": "Wrong Title",
                "company_name": "Wrong Co",
            },
        )
    )

    assert created.render_variables == {
        "candidate_name": "Candidate One",
        "candidate_email": "candidate@example.com",
        "job_title": "Backend Engineer",
        "company_name": "Acme Corp",
    }


def test_create_message_from_template_blocks_when_default_variable_missing(db_session_factory, seeded_data):
    user = seeded_data["user"]
    candidate = seeded_data["candidate"]

    template = _create_configured_template(
        user.id,
        default_variables={"company_name": "Acme Corp"},  # job_title left unconfigured
    )

    with pytest.raises(HTTPException) as exc_info:
        create_message(
            OutreachCreateRequest(
                candidate_profile_id=candidate.id,
                created_by_user_id=user.id,
                content_source=ContentSource.TEMPLATE,
                subject="Opportunity at Acme Corp",
                body_text="Hi Candidate One",
                body_html="<p>Hi Candidate One</p>",
                template_id=uuid.UUID(template.id),
            )
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail["missing"] == ["job_title"]

    # No message should have been persisted.
    listed = list_messages(
        created_by_user_id=user.id,
        candidate_profile_id=candidate.id,
        sent_status=None,
        offset=0,
        limit=50,
    )
    assert listed.total == 0


def test_create_message_from_template_allows_when_no_defaults_are_required(db_session_factory, seeded_data):
    user = seeded_data["user"]
    candidate = seeded_data["candidate"]

    template = create_template(
        OutreachTemplateCreateRequest(
            created_by_user_id=user.id,
            name="Candidate-only template",
            content_source=ContentSource.TEMPLATE,
            subject_template="Hi {{candidate_name}}",
            body_text_template="Hi {{candidate_name}}",
            body_html_template="<p>Hi {{candidate_name}}</p>",
            variables_used=["candidate_name"],
        )
    )

    created = create_message(
        OutreachCreateRequest(
            candidate_profile_id=candidate.id,
            created_by_user_id=user.id,
            content_source=ContentSource.TEMPLATE,
            subject="Hi Candidate One",
            body_text="Hi Candidate One",
            body_html="<p>Hi Candidate One</p>",
            template_id=uuid.UUID(template.id),
        )
    )

    assert created.render_variables == {
        "candidate_name": "Candidate One",
        "candidate_email": "candidate@example.com",
        "job_title": "",
        "company_name": "",
    }
