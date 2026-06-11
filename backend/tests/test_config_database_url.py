import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pydantic_settings  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("pydantic_settings")

    class BaseSettings:
        pass

    stub.BaseSettings = BaseSettings
    sys.modules["pydantic_settings"] = stub

from src.core import config  # noqa: E402


def test_rewrites_docker_db_host_to_localhost_outside_container():
    url = "postgresql://postgres:postgres@db:5432/recruitment_db"

    normalized = config._normalize_database_url_for_runtime(url, in_docker=False)

    assert normalized == "postgresql://postgres:postgres@localhost:5432/recruitment_db"


def test_keeps_docker_db_host_inside_container():
    url = "postgresql://postgres:postgres@db:5432/recruitment_db"

    normalized = config._normalize_database_url_for_runtime(url, in_docker=True)

    assert normalized == url


def test_keeps_non_docker_hosts_unchanged():
    url = "postgresql://postgres:postgres@localhost:5432/recruitment_db"

    normalized = config._normalize_database_url_for_runtime(url, in_docker=False)

    assert normalized == url
