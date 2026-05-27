import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.v1.endpoints import interview_questions as interview_module  # noqa: E402
from src.api.v1.endpoints.interview_questions import (  # noqa: E402
    GenerateQuestionsRequest,
    QuestionSetCreateRequest,
    QuestionSetUpdateRequest,
    create_question_set,
    delete_question_set,
    generate_question_set,
    get_question_set,
    list_question_sets,
    update_question_set,
)
from src.models.base import Base  # noqa: E402
from src.models.candidate_profile import CandidateProfile  # noqa: E402
from src.models.enums import ProfileStatus, UploadStatus, UserStatus  # noqa: E402
from src.models.job import Job  # noqa: E402
from src.models.job_matching import InterviewQuestionSet, JobDescription  # noqa: E402
from src.models.resume_document import ResumeDocument  # noqa: E402
from src.models.user_account import UserAccount  # noqa: E402


def _create_test_tables(engine):
    tables = [
        Base.metadata.tables["user_accounts"],
        Base.metadata.tables["jobs"],
        Base.metadata.tables["resume_documents"],
        Base.metadata.tables["candidate_profiles"],
        Base.metadata.tables["job_descriptions"],
        Base.metadata.tables["interview_question_sets"],
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
    monkeypatch.setattr(interview_module, "SessionLocal", factory)
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
            current_job_title="Backend Engineer",
            skills_text="Python, FastAPI",
            experience_text="5 years backend engineering",
            profile_status=ProfileStatus.REVIEWED,
        )
        db.add(candidate)
        db.flush()

        jd = JobDescription(
            job_id=job.id,
            title="Platform Engineer",
            jd_text="Need Python, FastAPI and systems design.",
            created_by_user_id=user.id,
            is_active=True,
        )
        db.add(jd)
        db.commit()
        db.refresh(user)
        db.refresh(candidate)
        db.refresh(jd)
        return {"user": user, "candidate": candidate, "jd": jd}
    finally:
        db.close()


def test_interview_question_set_crud(db_session_factory, seeded_data):
    user = seeded_data["user"]
    candidate = seeded_data["candidate"]
    jd = seeded_data["jd"]

    created = create_question_set(
        QuestionSetCreateRequest(
            candidate_profile_id=candidate.id,
            job_description_id=jd.id,
            generated_by_user_id=user.id,
            question_payload={"categories": [{"name": "Core", "questions": ["Q1"]}]},
        )
    )

    listed = list_question_sets(
        generated_by_user_id=user.id,
        candidate_profile_id=candidate.id,
        job_description_id=jd.id,
        offset=0,
        limit=50,
    )
    fetched = get_question_set(uuid.UUID(created.id))
    updated = update_question_set(
        uuid.UUID(created.id),
        QuestionSetUpdateRequest(
            question_payload={"categories": [{"name": "Updated", "questions": ["Q1", "Q2"]}]}
        ),
    )

    assert created.candidate_full_name == "Candidate One"
    assert listed.total == 1
    assert fetched.id == created.id
    assert updated.question_payload["categories"][0]["name"] == "Updated"

    delete_question_set(uuid.UUID(created.id))

    with db_session_factory() as db:
        assert db.get(InterviewQuestionSet, uuid.UUID(created.id)) is None


def test_generate_question_set_uses_llm_payload(monkeypatch, seeded_data):
    user = seeded_data["user"]
    candidate = seeded_data["candidate"]
    jd = seeded_data["jd"]

    class FakePrompts:
        @staticmethod
        def build_interview_questions_prompt(candidate_data, job_description_text):
            assert candidate_data["full_name"] == "Candidate One"
            assert "FastAPI" in job_description_text
            return "prompt"

    class FakeLLM:
        def generate(self, prompt):
            assert prompt == "prompt"
            return type(
                "Resp",
                (),
                {
                    "text": json.dumps(
                        {
                            "categories": [
                                {"name": "Technical", "questions": ["Explain FastAPI dependency injection."]}
                            ]
                        }
                    )
                },
            )()

    monkeypatch.setattr(
        "src.prompts.build_prompts.build_prompts",
        FakePrompts,
    )
    monkeypatch.setattr(
        "src.services.llm_service.LLMProvider",
        lambda: FakeLLM(),
    )

    result = generate_question_set(
        GenerateQuestionsRequest(
            candidate_profile_id=candidate.id,
            job_description_id=jd.id,
        ),
        current_user=user,
    )

    assert result.generated_by_user_id == str(user.id)
    assert result.question_payload["categories"][0]["name"] == "Technical"


def test_generate_question_set_returns_502_for_invalid_llm_json(monkeypatch, seeded_data):
    user = seeded_data["user"]
    candidate = seeded_data["candidate"]
    jd = seeded_data["jd"]

    class FakePrompts:
        @staticmethod
        def build_interview_questions_prompt(candidate_data, job_description_text):
            return "prompt"

    class BadLLM:
        def generate(self, prompt):
            return type("Resp", (), {"text": "not-json"})()

    monkeypatch.setattr(
        "src.prompts.build_prompts.build_prompts",
        FakePrompts,
    )
    monkeypatch.setattr(
        "src.services.llm_service.LLMProvider",
        lambda: BadLLM(),
    )

    with pytest.raises(HTTPException) as exc_info:
        generate_question_set(
            GenerateQuestionsRequest(
                candidate_profile_id=candidate.id,
                job_description_id=jd.id,
            ),
            current_user=user,
        )

    assert exc_info.value.status_code == 502
