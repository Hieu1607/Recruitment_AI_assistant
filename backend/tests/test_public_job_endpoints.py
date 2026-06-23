import sys
import types
import uuid
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from fastapi import FastAPI
from fastapi import UploadFile
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

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

if "passlib.context" not in sys.modules:
    passlib_stub = types.ModuleType("passlib")
    passlib_context_stub = types.ModuleType("passlib.context")

    class CryptContext:
        def __init__(self, *args, **kwargs):
            pass

        def hash(self, value):
            return f"hashed:{value}"

        def verify(self, plain_value, hashed_value):
            return hashed_value == f"hashed:{plain_value}"

    passlib_context_stub.CryptContext = CryptContext
    sys.modules["passlib"] = passlib_stub
    sys.modules["passlib.context"] = passlib_context_stub

if "src.services.ai_agent.graph" not in sys.modules:
    graph_stub = types.ModuleType("src.services.ai_agent.graph")
    graph_stub.get_graph = lambda: types.SimpleNamespace(invoke=lambda payload: payload)
    sys.modules["src.services.ai_agent.graph"] = graph_stub

if "src.services.job_description_service" not in sys.modules:
    jd_stub = types.ModuleType("src.services.job_description_service")
    jd_stub.create_job_description = lambda *args, **kwargs: None
    jd_stub.delete_job_description = lambda *args, **kwargs: True
    jd_stub.get_job_description = lambda *args, **kwargs: None
    jd_stub.list_job_descriptions = lambda *args, **kwargs: []
    jd_stub.update_job_description = lambda *args, **kwargs: None
    jd_stub._jd_to_dict = lambda jd: {}
    sys.modules["src.services.job_description_service"] = jd_stub

if "src.services.google_oauth" not in sys.modules:
    google_oauth_stub = types.ModuleType("src.services.google_oauth")
    google_oauth_stub.build_authorize_url = lambda redirect_path: ("https://accounts.google.com/o/oauth2/v2/auth", "state")
    google_oauth_stub.verify_state = lambda state: "/dashboard"
    google_oauth_stub.exchange_code_for_tokens = lambda code: {}
    google_oauth_stub.verify_id_token = lambda token: {}
    google_oauth_stub.upsert_user_from_google = lambda db, claims: None
    sys.modules["src.services.google_oauth"] = google_oauth_stub

if "src.services.resume_service" not in sys.modules:
    resume_stub = types.ModuleType("src.services.resume_service")
    resume_stub._get_resume_extraction_mode = lambda resume: None
    resume_stub._normalize_location_name = lambda value: value
    resume_stub._resume_to_dict = lambda resume: {}
    resume_stub.create_resume_document = lambda **kwargs: types.SimpleNamespace(
        id=uuid.uuid4()
    )
    resume_stub.delete_resume = lambda **kwargs: True
    resume_stub.get_resume = lambda **kwargs: None
    resume_stub.list_resumes = lambda **kwargs: []
    resume_stub.parse_pdf_to_sections = lambda **kwargs: []
    resume_stub.update_resume = lambda **kwargs: None
    sys.modules["src.services.resume_service"] = resume_stub

if "src.services.score_candidate" not in sys.modules:
    score_stub = types.ModuleType("src.services.score_candidate")
    score_stub.score_candidates = lambda **kwargs: {}
    sys.modules["src.services.score_candidate"] = score_stub

from src.models.deps import get_db  # noqa: E402
from src.api.v1.endpoints import public_jobs as public_jobs_module  # noqa: E402
from src.api.v1.endpoints.public_jobs import router as public_jobs_router  # noqa: E402


app = FastAPI()
app.include_router(public_jobs_router, prefix="/api/v1/public")


def _build_client(db):
    def _override_db():
        yield db

    app.dependency_overrides[get_db] = _override_db
    return TestClient(app, follow_redirects=False)


def _job(*, enabled=True):
    return SimpleNamespace(
        id=uuid.uuid4(),
        owner_user_id=uuid.uuid4(),
        title="Platform Engineer",
        candidate_message="Upload your latest PDF resume",
        public_apply_enabled=enabled,
    )


def _upload_file(filename, body, content_type):
    return UploadFile(
        file=BytesIO(body),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )


def test_get_public_job_returns_safe_payload_for_enabled_token():
    db = MagicMock()
    client = _build_client(db)
    job = _job(enabled=True)

    try:
        with (
            patch(
                "src.api.v1.endpoints.public_jobs.resolve_public_job_by_token",
                return_value=job,
            ) as resolve_job,
            patch(
                "src.api.v1.endpoints.public_jobs.require_public_job_enabled",
                side_effect=lambda value: value,
            ) as require_enabled,
        ):
            response = client.get("/api/v1/public/jobs/enabled-token")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "job_title": "Platform Engineer",
        "candidate_message": "Upload your latest PDF resume",
        "public_apply_enabled": True,
    }
    resolve_job.assert_called_once_with(db, "enabled-token")
    require_enabled.assert_called_once_with(job)


def test_get_public_job_returns_410_for_disabled_token():
    db = MagicMock()
    client = _build_client(db)
    job = _job(enabled=False)

    try:
        with (
            patch(
                "src.api.v1.endpoints.public_jobs.resolve_public_job_by_token",
                return_value=job,
            ) as resolve_job,
            patch(
                "src.api.v1.endpoints.public_jobs.require_public_job_enabled",
                side_effect=HTTPException(
                    status_code=410,
                    detail="Public application link is disabled",
                ),
            ) as require_enabled,
        ):
            response = client.get("/api/v1/public/jobs/disabled-token")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 410
    assert response.json() == {"detail": "Public application link is disabled"}
    resolve_job.assert_called_once_with(db, "disabled-token")
    require_enabled.assert_called_once_with(job)


def test_get_public_job_returns_404_for_unknown_token():
    db = MagicMock()
    client = _build_client(db)

    try:
        with patch(
            "src.api.v1.endpoints.public_jobs.resolve_public_job_by_token",
            side_effect=HTTPException(
                status_code=404,
                detail="Public application link not found",
            ),
        ) as resolve_job:
            response = client.get("/api/v1/public/jobs/missing-token")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
    assert response.json() == {"detail": "Public application link not found"}
    resolve_job.assert_called_once_with(db, "missing-token")


def test_upload_public_resume_requires_full_name():
    db = MagicMock()
    job = _job(enabled=True)
    file = _upload_file("resume.pdf", b"%PDF-1.4", "application/pdf")

    with (
        patch(
            "src.api.v1.endpoints.public_jobs.resolve_public_job_by_token",
            return_value=job,
        ),
        patch(
            "src.api.v1.endpoints.public_jobs.require_public_job_enabled",
            side_effect=lambda value: value,
        ),
        patch("src.api.v1.endpoints.public_jobs.create_resume_document") as create_resume,
    ):
        try:
            import asyncio

            asyncio.run(
                public_jobs_module.upload_public_resume(
                    token="enabled-token",
                    full_name="   ",
                    email="candidate@example.com",
                    file=file,
                    db=db,
                )
            )
            raise AssertionError("Expected HTTPException")
        except HTTPException as exc:
            assert exc.status_code == 422
            assert exc.detail == "full_name is required"

    create_resume.assert_not_called()


def test_upload_public_resume_requires_valid_email():
    db = MagicMock()
    job = _job(enabled=True)
    file = _upload_file("resume.pdf", b"%PDF-1.4", "application/pdf")

    with (
        patch(
            "src.api.v1.endpoints.public_jobs.resolve_public_job_by_token",
            return_value=job,
        ),
        patch(
            "src.api.v1.endpoints.public_jobs.require_public_job_enabled",
            side_effect=lambda value: value,
        ),
        patch("src.api.v1.endpoints.public_jobs.create_resume_document") as create_resume,
    ):
        try:
            import asyncio

            asyncio.run(
                public_jobs_module.upload_public_resume(
                    token="enabled-token",
                    full_name="Candidate Name",
                    email="not-an-email",
                    file=file,
                    db=db,
                )
            )
            raise AssertionError("Expected HTTPException")
        except HTTPException as exc:
            assert exc.status_code == 422
            assert exc.detail == "email must be a valid email address"

    create_resume.assert_not_called()


def test_upload_public_resume_requires_pdf():
    db = MagicMock()
    job = _job(enabled=True)
    file = _upload_file("resume.txt", b"plain text", "text/plain")

    with (
        patch(
            "src.api.v1.endpoints.public_jobs.resolve_public_job_by_token",
            return_value=job,
        ),
        patch(
            "src.api.v1.endpoints.public_jobs.require_public_job_enabled",
            side_effect=lambda value: value,
        ),
        patch("src.api.v1.endpoints.public_jobs.create_resume_document") as create_resume,
    ):
        try:
            import asyncio

            asyncio.run(
                public_jobs_module.upload_public_resume(
                    token="enabled-token",
                    full_name="Candidate Name",
                    email="candidate@example.com",
                    file=file,
                    db=db,
                )
            )
            raise AssertionError("Expected HTTPException")
        except HTTPException as exc:
            assert exc.status_code == 400
            assert exc.detail == "Only PDF files are allowed"

    create_resume.assert_not_called()


def test_upload_public_resume_succeeds_without_auth_and_returns_minimal_payload():
    db = MagicMock()
    job = _job(enabled=True)
    file = _upload_file("resume.pdf", b"%PDF-1.4 test content", "application/pdf")
    storage = MagicMock()
    storage.upload_bytes.return_value = "s3://resumes/resumes/job-id/object_resume.pdf"
    created_resume = SimpleNamespace(id=uuid.uuid4())
    task = SimpleNamespace(id="queued-task-id")

    with (
        patch(
            "src.api.v1.endpoints.public_jobs.resolve_public_job_by_token",
            return_value=job,
        ) as resolve_job,
        patch(
            "src.api.v1.endpoints.public_jobs.require_public_job_enabled",
            side_effect=lambda value: value,
        ) as require_enabled,
        patch(
            "src.api.v1.endpoints.public_jobs.get_object_storage",
            return_value=storage,
        ) as get_storage,
        patch(
            "src.api.v1.endpoints.public_jobs.create_resume_document",
            return_value=created_resume,
        ) as create_resume,
        patch(
            "src.api.v1.endpoints.public_jobs.process_resume.delay",
            return_value=task,
        ) as enqueue_resume,
        patch(
            "src.api.v1.endpoints.public_jobs.create_notification",
            create=True,
        ) as create_notification,
    ):
        import asyncio

        response = asyncio.run(
            public_jobs_module.upload_public_resume(
                token="enabled-token",
                full_name="  Candidate Name  ",
                email=" candidate@example.com ",
                file=file,
                db=db,
            )
        )

    assert response.model_dump() == {
        "submitted": True,
        "resume_document_id": str(created_resume.id),
        "status": "queued",
        "task_id": "queued-task-id",
    }
    resolve_job.assert_called_once_with(db, "enabled-token")
    require_enabled.assert_called_once_with(job)
    get_storage.assert_called_once_with()
    storage.upload_bytes.assert_called_once()
    create_resume.assert_called_once_with(
        db=db,
        storage_uri=storage.upload_bytes.return_value,
        original_file_name="resume.pdf",
        job_id=job.id,
        uploaded_by_user_id=job.owner_user_id,
    )
    enqueue_resume.assert_called_once_with(
        str(created_resume.id),
        submitted_full_name="Candidate Name",
        submitted_email="candidate@example.com",
    )
    create_notification.assert_called_once()
    assert create_notification.call_args.kwargs["db"] is db
    assert create_notification.call_args.kwargs["user_id"] == job.owner_user_id
    assert create_notification.call_args.kwargs["notification_type"] == "candidate_applied"
    assert create_notification.call_args.kwargs["target_url"] == f"/candidates/{created_resume.id}"
