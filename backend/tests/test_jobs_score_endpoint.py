import sys
import types
import uuid
from pathlib import Path

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

    class AIMessage:
        def __init__(self, content):
            self.content = content

    messages.HumanMessage = HumanMessage
    messages.AIMessage = AIMessage
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

from src.api.v1.endpoints.jobs import ScoreRequest, score_job_candidates  # noqa: E402
from src.models.base import Base  # noqa: E402
from src.models.enums import UserStatus  # noqa: E402
from src.models.user_account import UserAccount  # noqa: E402
from src.services.scoring_errors import ScoringProviderLimitError  # noqa: E402


def _create_test_tables(engine):
    tables = [
        Base.metadata.tables["user_accounts"],
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


def test_job_score_endpoint_returns_429_when_scoring_provider_limit_is_hit(
    monkeypatch, db, owner
):
    job_id = uuid.uuid4()
    jd = types.SimpleNamespace(id=uuid.uuid4())

    monkeypatch.setattr(
        "src.api.v1.endpoints.jobs.require_job_scoped_jd",
        lambda *args, **kwargs: jd,
    )
    monkeypatch.setattr(
        "src.api.v1.endpoints.jobs.score_candidates",
        lambda **kwargs: (_ for _ in ()).throw(
            ScoringProviderLimitError(
                "Scoring is temporarily unavailable because the configured LLM quota has been exhausted. Please retry later."
            )
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        score_job_candidates(
            job_id=job_id,
            body=ScoreRequest(),
            db=db,
            current_user=owner,
        )

    assert exc_info.value.status_code == 429
    assert "quota" in exc_info.value.detail.lower()
