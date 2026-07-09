import base64
import json
import sys
import types
from pathlib import Path

from sqlalchemy.dialects.sqlite.base import SQLiteTypeCompiler


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


if not hasattr(SQLiteTypeCompiler, "visit_JSONB"):
    SQLiteTypeCompiler.visit_JSONB = SQLiteTypeCompiler.visit_JSON


if "jose" not in sys.modules:
    jose_stub = types.ModuleType("jose")

    class JWTError(Exception):
        pass

    def _b64(data: dict) -> str:
        raw = json.dumps(data, default=str).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")

    jose_stub.JWTError = JWTError
    jose_stub.jwt = types.SimpleNamespace(
        encode=lambda payload, key, algorithm=None: ".".join(
            [_b64({"alg": algorithm or "HS256", "typ": "JWT"}), _b64(payload), "sig"]
        ),
        decode=lambda token, key, algorithms=None: {},
    )
    sys.modules["jose"] = jose_stub


if "joserfc" not in sys.modules:
    joserfc_stub = types.ModuleType("joserfc")
    joserfc_jwt_stub = types.ModuleType("joserfc.jwt")
    joserfc_jwk_stub = types.ModuleType("joserfc.jwk")

    class _DecodedToken:
        def __init__(self, claims: dict):
            self.claims = claims

    class KeySet:
        @classmethod
        def import_key_set(cls, payload: dict):
            return payload

    joserfc_jwt_stub.decode = lambda token, keyset: _DecodedToken({})
    joserfc_jwk_stub.KeySet = KeySet
    joserfc_stub.jwt = joserfc_jwt_stub
    sys.modules["joserfc"] = joserfc_stub
    sys.modules["joserfc.jwt"] = joserfc_jwt_stub
    sys.modules["joserfc.jwk"] = joserfc_jwk_stub


if "passlib.context" not in sys.modules:
    passlib_stub = types.ModuleType("passlib")
    passlib_context_stub = types.ModuleType("passlib.context")

    class CryptContext:
        def __init__(self, *args, **kwargs):
            pass

        def hash(self, value: str) -> str:
            return f"hashed:{value}"

        def verify(self, plain_value: str, hashed_value: str) -> bool:
            return hashed_value == f"hashed:{plain_value}"

    passlib_context_stub.CryptContext = CryptContext
    sys.modules["passlib"] = passlib_stub
    sys.modules["passlib.context"] = passlib_context_stub


if "langchain_core.messages" not in sys.modules:
    langchain_core = types.ModuleType("langchain_core")
    messages = types.ModuleType("langchain_core.messages")

    class BaseMessage:
        def __init__(self, content):
            self.content = content

    class HumanMessage:
        def __init__(self, content):
            self.content = content

    class AIMessage:
        def __init__(self, content):
            self.content = content

    messages.BaseMessage = BaseMessage
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


if "worker.tasks" not in sys.modules:
    worker_pkg = types.ModuleType("worker")
    tasks_stub = types.ModuleType("worker.tasks")
    tasks_stub.process_resume = types.SimpleNamespace(
        delay=lambda *args, **kwargs: types.SimpleNamespace(id="test-task-id")
    )
    tasks_stub.evaluate_candidate = types.SimpleNamespace(
        delay=lambda *args, **kwargs: types.SimpleNamespace(id="test-evaluation-task-id")
    )
    tasks_stub.evaluate_resume_batch = types.SimpleNamespace(
        delay=lambda *args, **kwargs: types.SimpleNamespace(id="test-batch-evaluation-task-id")
    )
    tasks_stub.send_interview_invitation_email = types.SimpleNamespace(
        delay=lambda *args, **kwargs: types.SimpleNamespace(id="test-interview-email-task-id")
    )
    worker_pkg.tasks = tasks_stub
    sys.modules["worker"] = worker_pkg
    sys.modules["worker.tasks"] = tasks_stub
