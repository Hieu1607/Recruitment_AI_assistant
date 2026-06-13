import json
import sys
import types
import uuid
from pathlib import Path

import pytest
from sqlalchemy import Text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pydantic_settings  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("pydantic_settings")

    class BaseSettings:
        pass

    stub.BaseSettings = BaseSettings
    sys.modules["pydantic_settings"] = stub

from src.models.candidate_profile import CandidateProfile  # noqa: E402
from src.models.resume_document import ExtractionTrace, ResumeDocument  # noqa: E402
from src.services import resume_service  # noqa: E402


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Sinh viên năm cuối Đại học Bách Khoa", "final_year"),
        ("Đang học năm cuối ngành Kế toán", "final_year"),
        ("Dự kiến tốt nghiệp năm 2026", "final_year"),
        ("Sắp tốt nghiệp chương trình cử nhân", "final_year"),
        ("Đang học tại Đại học Kinh tế", "studying"),
        ("Chưa tốt nghiệp, đang là sinh viên năm ba", "studying"),
    ],
)
def test_infer_graduation_status_supports_common_vietnamese_phrases(text, expected):
    assert resume_service._infer_graduation_status_from_text(text) == expected


class FakeSession:
    def __init__(self):
        self.objects: dict[type, dict[uuid.UUID, object]] = {}
        self.added: list[object] = []
        self.rollback_calls = 0

    def add(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = uuid.uuid4()
        self.objects.setdefault(type(obj), {})[obj.id] = obj
        self.added.append(obj)

    def commit(self):
        return None

    def refresh(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = uuid.uuid4()
        self.objects.setdefault(type(obj), {})[obj.id] = obj

    def rollback(self):
        self.rollback_calls += 1

    def get(self, model_cls, obj_id):
        return self.objects.get(model_cls, {}).get(obj_id)


def _stored_one(db: FakeSession, model_cls):
    values = list(db.objects.get(model_cls, {}).values())
    assert len(values) == 1
    return values[0]


def test_build_profile_prefers_usable_parsed_name_and_email():
    profile = resume_service._build_profile_from_parsed(
        uuid.uuid4(),
        {
            "name": " Parsed Candidate ",
            "email": " parsed@example.com ",
            "phone": " 123 ",
            "structured_profile": {
                "summary": {
                    "text": "Profile overview",
                    "links": ["https://example.com/portfolio"],
                },
                "projects": {
                    "entries": [
                        {
                            "title": "Project A",
                            "links": [{"url": "https://github.com/example/project-a"}],
                            "bullets": ["Built feature A"],
                        }
                    ]
                },
            },
        },
        submitted_full_name="Submitted Candidate",
        submitted_email="submitted@example.com",
    )

    assert profile.full_name == "Parsed Candidate"
    assert profile.email == "parsed@example.com"
    assert profile.phone == "123"
    assert profile.submitted_full_name == "Submitted Candidate"
    assert profile.submitted_email == "submitted@example.com"
    assert profile.structured_profile["summary"]["links"][0]["url"] == "https://example.com/portfolio"
    assert profile.structured_profile["projects"]["entries"][0]["title"] == "Project A"


def test_build_profile_strips_missing_placeholders_from_structured_entries():
    profile = resume_service._build_profile_from_parsed(
        uuid.uuid4(),
        {
            "name": "Teacher Candidate",
            "skills": "Curriculum Development",
            "structured_profile": {
                "skills": {
                    "entries": [
                        {
                            "title": "Curriculum Development",
                            "subtitle": "N/A",
                            "role": "N/A",
                            "location": "N/A",
                            "dateRange": "N/A",
                            "description": "N/A",
                            "metadata": ["N/A", "Curriculum Development"],
                            "bullets": ["N/A", "Built lesson plans"],
                        }
                    ],
                    "rawText": "N/A",
                }
            },
        },
    )

    skill_entry = profile.structured_profile["skills"]["entries"][0]

    assert skill_entry["title"] == "Curriculum Development"
    assert skill_entry["subtitle"] is None
    assert skill_entry["role"] is None
    assert skill_entry["location"] is None
    assert skill_entry["dateRange"] is None
    assert skill_entry["description"] is None
    assert skill_entry["metadata"] == ["Curriculum Development"]
    assert skill_entry["bullets"] == ["Built lesson plans"]
    assert profile.structured_profile["skills"]["rawText"] == "Curriculum Development"


def test_build_profile_falls_back_to_submitted_name_and_email_when_parse_is_sparse():
    profile = resume_service._build_profile_from_parsed(
        uuid.uuid4(),
        {
            "name": "   ",
            "email": None,
        },
        submitted_full_name=" Submitted Candidate ",
        submitted_email=" submitted@example.com ",
    )

    assert profile.full_name == "Submitted Candidate"
    assert profile.email == "submitted@example.com"
    assert profile.submitted_full_name == "Submitted Candidate"
    assert profile.submitted_email == "submitted@example.com"


def test_candidate_profile_cpa_column_supports_unbounded_text():
    cpa_column = CandidateProfile.__table__.columns["cpa"]

    assert isinstance(cpa_column.type, Text)


def test_parse_pdf_to_sections_persists_submitted_values_and_uses_fallbacks(
    monkeypatch,
):
    db = FakeSession()
    job_id = uuid.uuid4()
    owner_user_id = uuid.uuid4()

    monkeypatch.setattr(
        resume_service, "extract_text_from_pdf", lambda filepath: "cv text"
    )
    monkeypatch.setattr(
        resume_service.build_prompts,
        "build_cv_parsing_prompt",
        lambda cv_text: "prompt",
    )

    class FakeProvider:
        def __init__(self):
            self.provider = "groq"
            self.model_name = "fake-model"

        def clone_with_model(self, *, provider=None, model_name=None, allow_fallback=None):
            self.provider = provider or self.provider
            self.model_name = model_name or self.model_name
            return self

        def generate(self, prompt):
            return types.SimpleNamespace(
                text=json.dumps({"name": "", "email": None, "skills": "Python"}),
                provider=self.provider,
                model=self.model_name,
            )

    monkeypatch.setattr(resume_service, "LLMProvider", lambda **kwargs: FakeProvider())

    result = resume_service.parse_pdf_to_sections(
        filepaths=["resume.pdf"],
        db=db,
        job_id=job_id,
        uploaded_by_user_id=owner_user_id,
        original_filenames=["resume.pdf"],
        submitted_full_names=["Submitted Name"],
        submitted_emails=["submitted@example.com"],
    )

    assert result[0]["status"] == "processed"

    profile = _stored_one(db, CandidateProfile)
    trace = [item for item in db.added if isinstance(item, ExtractionTrace)][-1]

    assert profile.full_name == "Submitted Name"
    assert profile.email == "submitted@example.com"
    assert profile.submitted_full_name == "Submitted Name"
    assert profile.submitted_email == "submitted@example.com"
    assert profile.skills_text == "Python"
    assert trace.payload["submittedFullName"] == "Submitted Name"
    assert trace.payload["submittedEmail"] == "submitted@example.com"
    assert trace.payload["parsedName"] is None
    assert trace.payload["parsedEmail"] is None
    assert trace.payload["usedSubmittedFullName"] is True
    assert trace.payload["usedSubmittedEmail"] is True


def test_parse_pdf_to_sections_creates_minimal_profile_on_parse_failure(monkeypatch):
    db = FakeSession()
    job_id = uuid.uuid4()

    monkeypatch.setattr(
        resume_service, "extract_text_from_pdf", lambda filepath: "cv text"
    )
    monkeypatch.setattr(
        resume_service.build_prompts,
        "build_cv_parsing_prompt",
        lambda cv_text: "prompt",
    )

    class ExplodingProvider:
        def generate(self, prompt):
            raise RuntimeError("parse failed")

    monkeypatch.setattr(resume_service, "LLMProvider", lambda **kwargs: ExplodingProvider())

    result = resume_service.parse_pdf_to_sections(
        filepaths=["resume.pdf"],
        db=db,
        job_id=job_id,
        original_filenames=["resume.pdf"],
        submitted_full_names=["Fallback Name"],
        submitted_emails=["fallback@example.com"],
    )

    profile = _stored_one(db, CandidateProfile)
    resume = _stored_one(db, ResumeDocument)
    failure_trace = [item for item in db.added if isinstance(item, ExtractionTrace)][-1]

    assert result[0]["status"] == "failed"
    assert result[0]["candidate_profile_id"] == str(profile.id)
    assert profile.full_name == "Fallback Name"
    assert profile.email == "fallback@example.com"
    assert profile.submitted_full_name == "Fallback Name"
    assert profile.submitted_email == "fallback@example.com"
    assert resume.upload_status == "failed"
    assert failure_trace.payload["createdFallbackProfile"] is True
    assert failure_trace.payload["candidateName"] == "Fallback Name"
    assert failure_trace.payload["candidateEmail"] == "fallback@example.com"
    assert failure_trace.payload["usedSubmittedFullName"] is True
    assert failure_trace.payload["usedSubmittedEmail"] is True


def test_parse_pdf_to_sections_uses_vision_fallback_for_image_only_pdf(monkeypatch):
    db = FakeSession()
    job_id = uuid.uuid4()
    owner_user_id = uuid.uuid4()

    monkeypatch.setattr(resume_service, "extract_text_from_pdf", lambda filepath: "")
    monkeypatch.setattr(
        resume_service,
        "_render_pdf_pages_as_images",
        lambda pdf_source, max_pages=3: [b"fake-image-bytes"],
    )
    monkeypatch.setattr(
        resume_service.build_prompts,
        "build_cv_vision_prompt",
        lambda: "vision prompt",
    )

    def _unexpected_ocr(*args, **kwargs):
        raise AssertionError("HF OCR should not be used when vision fallback succeeds")

    monkeypatch.setattr(resume_service, "extract_text_via_hf_ocr", _unexpected_ocr)

    class FakeProvider:
        def generate(self, prompt):
            raise AssertionError("text-only prompt path should not be used")

        def generate_with_images(self, prompt, images):
            assert prompt == "vision prompt"
            assert images == [b"fake-image-bytes"]
            return types.SimpleNamespace(
                text=json.dumps(
                    {
                        "name": "Vision Candidate",
                        "email": "vision@example.com",
                        "skills": "Python, SQL",
                    }
                ),
                provider="fake",
                model="fake-vision-model",
            )

    monkeypatch.setattr(resume_service, "LLMProvider", lambda **kwargs: FakeProvider())

    result = resume_service.parse_pdf_to_sections(
        filepaths=["resume.pdf"],
        db=db,
        job_id=job_id,
        uploaded_by_user_id=owner_user_id,
        original_filenames=["resume.pdf"],
    )

    assert result[0]["status"] == "processed"

    profile = _stored_one(db, CandidateProfile)
    trace = [item for item in db.added if isinstance(item, ExtractionTrace)][-1]

    assert profile.full_name == "Vision Candidate"
    assert profile.email == "vision@example.com"
    assert profile.skills_text == "Python, SQL"
    assert trace.payload["llmModel"] == "fake-vision-model"


def test_parse_pdf_to_sections_retries_invalid_json_and_records_trace(
    monkeypatch, tmp_path
):
    db = FakeSession()
    job_id = uuid.uuid4()
    owner_user_id = uuid.uuid4()

    monkeypatch.setattr(resume_service, "extract_text_from_pdf", lambda filepath: "cv text")
    monkeypatch.setattr(
        resume_service.build_prompts,
        "build_cv_parsing_prompt",
        lambda cv_text: "prompt",
    )
    monkeypatch.setenv("LANGGRAPH_TRACE_LOG_DIR", str(tmp_path))

    class FakeProvider:
        def __init__(self):
            self.calls = 0
            self.model_name = "llama-3.1-8b-instant"
            self.provider = "groq"

        def clone_with_model(self, *, provider=None, model_name=None, allow_fallback=None):
            self.provider = provider or self.provider
            self.model_name = model_name or self.model_name
            return self

        def generate(self, prompt):
            self.calls += 1
            if self.calls == 1:
                return types.SimpleNamespace(
                    text='{"name":"Broken"',
                    provider="groq",
                    model="llama-3.1-8b-instant",
                )
            return types.SimpleNamespace(
                text=json.dumps(
                    {
                        "name": "Retry Candidate",
                        "email": "retry@example.com",
                        "skills": "Python",
                    }
                ),
                provider="groq",
                model="llama-3.1-8b-instant",
            )

    provider = FakeProvider()
    monkeypatch.setattr(resume_service, "LLMProvider", lambda **kwargs: provider)
    monkeypatch.setattr(resume_service, "_resume_trace_logger", None, raising=False)

    result = resume_service.parse_pdf_to_sections(
        filepaths=["resume.pdf"],
        db=db,
        job_id=job_id,
        uploaded_by_user_id=owner_user_id,
        original_filenames=["resume.pdf"],
    )

    assert result[0]["status"] == "processed"
    assert provider.calls == 2

    trace = [item for item in db.added if isinstance(item, ExtractionTrace)][-1]
    trace_path = Path(trace.payload["resumeParseTraceFile"])
    assert trace.payload["jsonAttemptCount"] == 2
    assert trace_path.exists()

    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace_payload["status"] == "success"
    assert trace_payload["metadata"]["file_name"] == "resume.pdf"
    assert trace_payload["metadata"]["extraction_mode"] == "text"
    assert trace_payload["result"]["llm"]["model"] == "llama-3.1-8b-instant"
    assert len(trace_payload["llm_attempts"]) == 2
    assert trace_payload["llm_attempts"][0]["parse_error"]["message"] == "LLM did not return a valid JSON object"


def test_generate_resume_json_with_retries_uses_groq_shopaikey_sequence():
    call_order = []
    provider_call_counts = {"groq": 0, "shopaikey": 0}
    logged_attempts = []

    class FakeProvider:
        def __init__(self, provider="groq", model_name="groq-model"):
            self.provider = provider
            self.model_name = model_name
            self.temperature = 0.2
            self.max_tokens = 4096
            self.timeout_seconds = 60
            self.max_retries = 0

        def clone_with_model(self, *, provider=None, model_name=None, allow_fallback=None):
            selected_provider = provider or self.provider
            return FakeProvider(
                provider=selected_provider,
                model_name=model_name or f"{selected_provider}-model",
            )

        def generate(self, prompt):
            provider_call_counts[self.provider] += 1
            attempt = provider_call_counts[self.provider]
            call_order.append(self.provider)

            if self.provider == "groq" and attempt == 2:
                text = json.dumps(
                    {
                        "name": "Retry Candidate",
                        "email": "retry@example.com",
                        "skills": "Python",
                    }
                )
            else:
                text = '{"name":"Broken"'

            return types.SimpleNamespace(
                text=text,
                provider=self.provider,
                model=self.model_name,
            )

    class FakeTraceLogger:
        def record_llm_attempt(self, *, trace_id, payload):
            logged_attempts.append(payload)

    original_logger_factory = resume_service.get_resume_parse_trace_logger
    resume_service.get_resume_parse_trace_logger = lambda: FakeTraceLogger()
    try:
        parsed, llm_response, attempt_count = resume_service._generate_resume_json_with_retries(
            trace_id="trace-123",
            stage="text",
            llm_provider=FakeProvider(),
            prompt="prompt",
        )
    finally:
        resume_service.get_resume_parse_trace_logger = original_logger_factory

    assert parsed["name"] == "Retry Candidate"
    assert llm_response.provider == "groq"
    assert attempt_count == 3
    assert call_order == ["groq", "shopaikey", "groq"]
    assert len(logged_attempts) == 3
    assert logged_attempts[0]["provider"] == "groq"
    assert logged_attempts[0]["model"] == resume_service.settings.RESUME_PARSE_MODEL_NAME
    assert logged_attempts[0]["response_chars"] == len('{"name":"Broken"')
    assert logged_attempts[0]["prompt_chars"] == len("prompt")
    assert isinstance(logged_attempts[0]["request_duration_ms"], (int, float))
    assert logged_attempts[0]["request_duration_ms"] >= 0
    assert logged_attempts[0]["parse_error"]["message"] == "LLM did not return a valid JSON object"
    assert logged_attempts[1]["provider"] == "shopaikey"
    assert logged_attempts[1]["model"] == resume_service.settings.SHOPAIKEY_MODEL_NAME
    assert logged_attempts[2]["provider"] == "groq"
    assert logged_attempts[2]["parse_error"] is None


def test_parse_pdf_to_sections_records_failure_trace_file(monkeypatch, tmp_path):
    db = FakeSession()
    job_id = uuid.uuid4()

    monkeypatch.setattr(resume_service, "extract_text_from_pdf", lambda filepath: "cv text")
    monkeypatch.setattr(
        resume_service.build_prompts,
        "build_cv_parsing_prompt",
        lambda cv_text: "prompt",
    )
    monkeypatch.setenv("LANGGRAPH_TRACE_LOG_DIR", str(tmp_path))

    class FakeProvider:
        def __init__(self):
            self.calls = 0
            self.provider = "groq"
            self.model_name = "llama-3.1-8b-instant"

        def clone_with_model(self, *, provider=None, model_name=None, allow_fallback=None):
            self.provider = provider or self.provider
            self.model_name = model_name or self.model_name
            return self

        def generate(self, prompt):
            self.calls += 1
            return types.SimpleNamespace(
                text="not-json",
                provider="groq",
                model="llama-3.1-8b-instant",
            )

    provider = FakeProvider()
    monkeypatch.setattr(resume_service, "LLMProvider", lambda **kwargs: provider)
    monkeypatch.setattr(resume_service, "_resume_trace_logger", None, raising=False)

    result = resume_service.parse_pdf_to_sections(
        filepaths=["resume.pdf"],
        db=db,
        job_id=job_id,
        original_filenames=["resume.pdf"],
    )

    assert result[0]["status"] == "failed"
    assert provider.calls >= 2

    failure_trace = [item for item in db.added if isinstance(item, ExtractionTrace)][-1]
    trace_path = Path(failure_trace.payload["resumeParseTraceFile"])
    assert trace_path.exists()

    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace_payload["status"] == "failed"
    assert trace_payload["error"]["message"].startswith("resume parsing failed after retries:")
    assert len(trace_payload["llm_attempts"]) == provider.calls


def test_latest_extraction_mode_prefers_newest_success_trace():
    resume_id = uuid.uuid4()
    older = ExtractionTrace(
        resume_document_id=resume_id,
        stage="cv_parsing",
        status="success",
        payload={"extractionMode": "ocr"},
    )
    newer = ExtractionTrace(
        resume_document_id=resume_id,
        stage="cv_parsing",
        status="success",
        payload={"extractionMode": "vision"},
    )

    mode = resume_service._latest_extraction_mode_from_traces([older, newer])

    assert mode == "vision"


def test_resume_to_dict_includes_extraction_mode():
    resume = ResumeDocument(
        id=uuid.uuid4(),
        job_id=uuid.uuid4(),
        original_file_name="resume.pdf",
        storage_uri="s3://bucket/key.pdf",
        upload_status="processed",
        uploaded_by_user_id=uuid.uuid4(),
    )

    data = resume_service._resume_to_dict(resume, extraction_mode="text")

    assert data["extraction_mode"] == "text"


def test_resume_llm_provider_uses_resume_parse_model_and_higher_token_budget(monkeypatch):
    captured = {}

    class FakeProvider:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(resume_service, "LLMProvider", FakeProvider)
    monkeypatch.setattr(resume_service.settings, "LLM_MAX_TOKENS", 1024)
    monkeypatch.setattr(resume_service.settings, "RESUME_PARSE_MAX_TOKENS", 4096)
    monkeypatch.setattr(
        resume_service.settings,
        "RESUME_PARSE_MODEL_NAME",
        "llama-3.1-8b-instant",
    )

    resume_service._resume_llm_provider()

    assert captured["model_name"] == "llama-3.1-8b-instant"
    assert captured["max_tokens"] == 4096
