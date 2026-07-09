import sys
import types
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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

if "minio" not in sys.modules:
    minio_stub = types.ModuleType("minio")
    minio_error_stub = types.ModuleType("minio.error")

    class Minio:
        def __init__(self, *args, **kwargs):
            pass

    class S3Error(Exception):
        code = "S3Error"

    minio_stub.Minio = Minio
    minio_error_stub.S3Error = S3Error
    sys.modules["minio"] = minio_stub
    sys.modules["minio.error"] = minio_error_stub

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

from src.api.v1.endpoints.resume import get_resume_document_file  # noqa: E402


def _db_with_resume(resume):
    db = MagicMock()
    db.execute.return_value.scalars.return_value.first.return_value = resume
    return db


def test_get_resume_document_file_supports_unicode_filename_in_content_disposition():
    resume = SimpleNamespace(
        id=uuid.uuid4(),
        original_file_name="0004_Lê_Thị_Hòa_CorporateDark.pdf",
        storage_uri="s3://resumes/resumes/job/resume.pdf",
    )
    db = _db_with_resume(resume)
    current_user = SimpleNamespace(id=uuid.uuid4())

    storage = MagicMock()
    storage.download_bytes.return_value = b"%PDF-1.4 synthetic"

    with patch(
        "src.api.v1.endpoints.resume.get_object_storage",
        return_value=storage,
    ):
        response = get_resume_document_file(
            resume_id=resume.id,
            db=db,
            current_user=current_user,
        )

    assert response.status_code == 200
    assert response.media_type == "application/pdf"
    assert response.body == b"%PDF-1.4 synthetic"
    assert (
        response.headers["content-disposition"]
        == "inline; filename=\"0004_L__Th__H_a_CorporateDark.pdf\"; "
        "filename*=UTF-8''0004_L%C3%AA_Th%E1%BB%8B_H%C3%B2a_CorporateDark.pdf"
    )
