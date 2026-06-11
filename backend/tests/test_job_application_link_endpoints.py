import sys
import types
from pathlib import Path
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pydantic_settings  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("pydantic_settings")

    class BaseSettings:
        pass

    stub.BaseSettings = BaseSettings
    sys.modules["pydantic_settings"] = stub

if "jose" not in sys.modules:
    jose_stub = types.ModuleType("jose")

    class JWTError(Exception):
        pass

    jose_stub.JWTError = JWTError
    jose_stub.jwt = types.SimpleNamespace(decode=lambda *args, **kwargs: {})
    sys.modules["jose"] = jose_stub

if "langchain_core.messages" not in sys.modules:
    langchain_core = types.ModuleType("langchain_core")
    messages = types.ModuleType("langchain_core.messages")

    class HumanMessage:
        def __init__(self, content):
            self.content = content

    messages.HumanMessage = HumanMessage
    sys.modules["langchain_core"] = langchain_core
    sys.modules["langchain_core.messages"] = messages

if "multipart" not in sys.modules:
    multipart_stub = types.ModuleType("multipart")
    multipart_stub.__version__ = "0.0-test"
    multipart_multipart_stub = types.ModuleType("multipart.multipart")
    multipart_multipart_stub.parse_options_header = lambda value: ("", {})
    sys.modules["multipart"] = multipart_stub
    sys.modules["multipart.multipart"] = multipart_multipart_stub

if "src.services.ai_agent.graph" not in sys.modules:
    graph_stub = types.ModuleType("src.services.ai_agent.graph")
    graph_stub.get_graph = lambda: types.SimpleNamespace(invoke=lambda payload: payload)
    sys.modules["src.services.ai_agent.graph"] = graph_stub

if "src.services.job_description_service" not in sys.modules:
    jd_stub = types.ModuleType("src.services.job_description_service")
    jd_stub._jd_to_dict = lambda jd: {}
    sys.modules["src.services.job_description_service"] = jd_stub

if "src.services.resume_service" not in sys.modules:
    resume_stub = types.ModuleType("src.services.resume_service")
    resume_stub._resume_to_dict = lambda resume: {}
    resume_stub.create_resume_document = lambda **kwargs: types.SimpleNamespace(
        id=uuid.uuid4()
    )
    resume_stub.parse_pdf_to_sections = lambda **kwargs: []
    sys.modules["src.services.resume_service"] = resume_stub

if "src.services.score_candidate" not in sys.modules:
    score_stub = types.ModuleType("src.services.score_candidate")
    score_stub.score_candidates = lambda **kwargs: {}
    sys.modules["src.services.score_candidate"] = score_stub

from src.api.v1.endpoints.jobs import (  # noqa: E402
    JobCreateRequest,
    JobUpdateRequest,
    create_job,
    get_job,
    get_job_application_link,
    list_jobs,
    rotate_job_application_link,
    update_job,
)
from src.core.config import settings  # noqa: E402
from src.models.base import Base  # noqa: E402
from src.models.job import Job  # noqa: E402
from src.models.user_account import UserAccount  # noqa: E402
from src.models.enums import UserStatus  # noqa: E402


def _create_test_tables(engine):
    tables = [
        Base.metadata.tables["user_accounts"],
        Base.metadata.tables["jobs"],
    ]
    Base.metadata.create_all(engine, tables=tables)


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    _create_test_tables(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture()
def owner(db):
    user = UserAccount(
        email="owner@example.com",
        display_name="Owner",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture()
def outsider(db):
    user = UserAccount(
        email="outsider@example.com",
        display_name="Outsider",
        password_hash=None,
        status=UserStatus.ACTIVE,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def test_create_job_exposes_application_settings(db, owner, monkeypatch):
    monkeypatch.setattr(settings, "FRONTEND_BASE_URL", "https://frontend.example.com")

    response = create_job(
        JobCreateRequest(
            title="Platform Engineer",
            status="active",
            candidate_message=" Apply here ",
            public_apply_enabled=True,
        ),
        db=db,
        current_user=owner,
    )

    assert response.title == "Platform Engineer"
    assert response.candidate_message == "Apply here"
    assert response.public_apply_enabled is True
    assert response.public_apply_url.startswith("https://frontend.example.com/apply/")


def test_list_and_get_job_include_application_settings(db, owner, monkeypatch):
    monkeypatch.setattr(settings, "FRONTEND_BASE_URL", "https://frontend.example.com/")

    created = create_job(
        JobCreateRequest(
            title="Backend Engineer",
            status="active",
            candidate_message="Resume required",
            public_apply_enabled=True,
        ),
        db=db,
        current_user=owner,
    )

    listing = list_jobs(db=db, current_user=owner)
    fetched = get_job(job_id=uuid.UUID(created.id), db=db, current_user=owner)

    assert listing.total == 1
    assert listing.items[0].candidate_message == "Resume required"
    assert listing.items[0].public_apply_url == created.public_apply_url
    assert fetched.id == created.id
    assert fetched.public_apply_enabled is True


def test_update_job_can_change_candidate_message_and_public_apply_enabled(db, owner, monkeypatch):
    monkeypatch.setattr(settings, "FRONTEND_BASE_URL", "https://frontend.example.com")

    created = create_job(
        JobCreateRequest(title="QA Engineer", status="active"),
        db=db,
        current_user=owner,
    )

    updated = update_job(
        job_id=uuid.UUID(created.id),
        body=JobUpdateRequest(
            candidate_message=" Upload your latest CV ",
            public_apply_enabled=False,
        ),
        db=db,
        current_user=owner,
    )

    persisted = db.get(Job, uuid.UUID(created.id))

    assert updated.candidate_message == "Upload your latest CV"
    assert updated.public_apply_enabled is False
    assert persisted.public_apply_disabled_at is not None

    reenabled = update_job(
        job_id=persisted.id,
        body=JobUpdateRequest(public_apply_enabled=True),
        db=db,
        current_user=owner,
    )
    db.refresh(persisted)

    assert reenabled.public_apply_enabled is True
    assert persisted.public_apply_disabled_at is None


def test_get_application_link_returns_dedicated_payload(db, owner, monkeypatch):
    monkeypatch.setattr(settings, "FRONTEND_BASE_URL", "https://frontend.example.com")

    created = create_job(
        JobCreateRequest(
            title="Data Engineer",
            status="active",
            candidate_message="Please upload PDF only",
        ),
        db=db,
        current_user=owner,
    )

    payload = get_job_application_link(
        job_id=uuid.UUID(created.id),
        db=db,
        current_user=owner,
    )

    assert payload.candidate_message == "Please upload PDF only"
    assert payload.public_apply_enabled is True
    assert payload.public_apply_url == created.public_apply_url


def test_rotate_application_link_changes_job_token(db, owner, monkeypatch):
    monkeypatch.setattr(settings, "FRONTEND_BASE_URL", "https://frontend.example.com")

    create_job(
        JobCreateRequest(title="DevOps Engineer", status="active"),
        db=db,
        current_user=owner,
    )
    job = db.query(Job).filter(Job.owner_user_id == owner.id).one()
    original_token = job.public_apply_token

    payload = rotate_job_application_link(job_id=job.id, db=db, current_user=owner)
    db.refresh(job)

    assert job.public_apply_token != original_token
    assert payload.public_apply_url.endswith(job.public_apply_token)
    assert payload.public_apply_enabled is True


def test_application_links_are_unique_per_job_for_same_owner(db, owner, monkeypatch):
    monkeypatch.setattr(settings, "FRONTEND_BASE_URL", "https://frontend.example.com")

    first = create_job(
        JobCreateRequest(title="Frontend Engineer", status="active"),
        db=db,
        current_user=owner,
    )
    second = create_job(
        JobCreateRequest(title="ML Engineer", status="active"),
        db=db,
        current_user=owner,
    )

    listing = list_jobs(db=db, current_user=owner)
    urls_by_id = {item.id: item.public_apply_url for item in listing.items}
    first_original_url = urls_by_id[first.id]
    second_original_url = urls_by_id[second.id]

    assert listing.total == 2
    assert first_original_url
    assert second_original_url
    assert first_original_url != second_original_url

    rotated = rotate_job_application_link(job_id=uuid.UUID(first.id), db=db, current_user=owner)
    second_after_rotate = get_job(job_id=uuid.UUID(second.id), db=db, current_user=owner)

    assert rotated.public_apply_url != first_original_url
    assert second_after_rotate.public_apply_url == second_original_url


def test_non_owner_cannot_access_application_settings(db, owner, outsider):
    create_job(
        JobCreateRequest(title="Security Engineer", status="active"),
        db=db,
        current_user=owner,
    )
    job = db.query(Job).filter(Job.owner_user_id == owner.id).one()

    with pytest.raises(HTTPException) as exc_info:
        get_job_application_link(job_id=job.id, db=db, current_user=outsider)

    assert exc_info.value.status_code == 404
