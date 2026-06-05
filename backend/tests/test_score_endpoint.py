import sys
import uuid
from pathlib import Path
from decimal import Decimal

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.v1.endpoints.score import (  # noqa: E402
    ScoreRequest,
    SectionWeights,
    score_candidates_endpoint,
)
from src.models.base import Base  # noqa: E402
from src.models.enums import UserStatus  # noqa: E402
from src.models.job import Job  # noqa: E402
from src.models.job_matching import JobDescription  # noqa: E402
from src.services.job_description_service import (  # noqa: E402
    _jd_to_dict,
    create_job_description,
    get_job_description,
    update_job_description,
)
from src.models.user_account import UserAccount  # noqa: E402
from src.services.scoring_errors import ScoringProviderLimitError  # noqa: E402


def _create_test_tables(engine):
    tables = [
        Base.metadata.tables["user_accounts"],
        Base.metadata.tables["jobs"],
        Base.metadata.tables["job_descriptions"],
    ]
    Base.metadata.create_all(engine, tables=tables)


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
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


@pytest.fixture()
def job_description(db, owner):
    job = Job(owner_user_id=owner.id, title="Backend Engineer", status="active")
    db.add(job)
    db.flush()

    jd = JobDescription(
        job_id=job.id,
        title="Backend Engineer",
        jd_text="Python, FastAPI, PostgreSQL",
        created_by_user_id=owner.id,
        is_active=True,
    )
    db.add(jd)
    db.commit()
    db.refresh(jd)
    return jd


def test_score_endpoint_rejects_all_zero_weights(db, owner, job_description):
    with pytest.raises(HTTPException) as exc_info:
        score_candidates_endpoint(
            ScoreRequest(
                job_description_id=job_description.id,
                section_weights=SectionWeights(skills=0, experience=0),
            ),
            db=db,
            current_user=owner,
        )

    assert exc_info.value.status_code == 422
    assert "At least one section must have a positive weight" in exc_info.value.detail


def test_score_endpoint_rejects_job_description_not_owned(db, outsider, job_description):
    with pytest.raises(HTTPException) as exc_info:
        score_candidates_endpoint(
            ScoreRequest(job_description_id=job_description.id),
            db=db,
            current_user=outsider,
        )

    assert exc_info.value.status_code == 404
    assert str(job_description.id) in exc_info.value.detail


def test_score_endpoint_passes_explicit_weights_and_filters(monkeypatch, db, owner, job_description):
    candidate_ids = [uuid.uuid4(), uuid.uuid4()]
    captured = {}

    def fake_score_candidates(**kwargs):
        captured.update(kwargs)
        return {
            "match_run_id": "run-123",
            "job_description_id": str(kwargs["job_description_id"]),
            "total_candidates": 2,
            "total_passed_candidates": 1,
            "batches": 1,
            "scores": [],
        }

    monkeypatch.setattr(
        "src.api.v1.endpoints.score.score_candidates",
        fake_score_candidates,
    )

    response = score_candidates_endpoint(
        ScoreRequest(
            job_description_id=job_description.id,
            score_threshold=77.5,
            candidate_profile_ids=candidate_ids,
            section_weights=SectionWeights(skills=40, experience=60, education=None),
            batch_size=7,
        ),
        db=db,
        current_user=owner,
    )

    assert response["match_run_id"] == "run-123"
    assert captured["job_description_id"] == job_description.id
    assert captured["initiated_by_user_id"] == owner.id
    assert captured["score_threshold"] == Decimal("77.5")
    assert captured["candidate_profile_ids"] == candidate_ids
    assert captured["section_weights"] == {"skills": 40.0, "experience": 60.0}
    assert captured["batch_size"] == 7


def test_score_endpoint_returns_429_when_scoring_provider_limit_is_hit(
    monkeypatch, db, owner, job_description
):
    monkeypatch.setattr(
        "src.api.v1.endpoints.score.score_candidates",
        lambda **kwargs: (_ for _ in ()).throw(
            ScoringProviderLimitError(
                "Scoring is temporarily unavailable because the configured LLM quota has been exhausted. Please retry later."
            )
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        score_candidates_endpoint(
            ScoreRequest(job_description_id=job_description.id),
            db=db,
            current_user=owner,
        )

    assert exc_info.value.status_code == 429
    assert "quota" in exc_info.value.detail.lower()


def test_job_description_serialization_includes_hidden_text(job_description):
    job_description.hidden_text = "Internal scoring criteria"

    payload = _jd_to_dict(job_description)

    assert payload["hidden_text"] == "Internal scoring criteria"


def test_job_description_crud_round_trips_hidden_text(db, owner):
    job = Job(owner_user_id=owner.id, title="Data Engineer", status="active")
    db.add(job)
    db.commit()
    db.refresh(job)

    created = create_job_description(
        db=db,
        job_id=job.id,
        title="Data Engineer",
        jd_text="Public JD",
        hidden_text="Internal scoring criteria",
        created_by_user_id=owner.id,
    )
    assert created["hidden_text"] == "Internal scoring criteria"

    updated = update_job_description(
        db=db,
        jd_id=uuid.UUID(created["id"]),
        hidden_text="Updated internal criteria",
    )
    assert updated is not None
    assert updated["hidden_text"] == "Updated internal criteria"

    fetched = get_job_description(db=db, jd_id=uuid.UUID(created["id"]))
    assert fetched is not None
    assert fetched["hidden_text"] == "Updated internal criteria"
