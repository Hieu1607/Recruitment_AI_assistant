import logging
import sys
import types

import pytest
from fastapi.testclient import TestClient

from src.services import llm_service


def test_groq_rate_limit_is_wrapped_and_logged(monkeypatch, caplog):
    class FakeCompletions:
        def create(self, **kwargs):
            raise Exception(
                "Error code: 429 - {'error': {'message': 'Rate limit reached for model on tokens per day', "
                "'code': 'rate_limit_exceeded'}}"
            )

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = types.SimpleNamespace(
                completions=FakeCompletions(),
            )

    monkeypatch.setitem(sys.modules, "groq", types.SimpleNamespace(Groq=FakeGroq))
    adapter = llm_service._GroqAdapter(
        api_key="test-key",
        model="llama-test",
        temperature=0,
        max_tokens=128,
        timeout_seconds=5,
        max_retries=2,
    )
    caplog.set_level(logging.ERROR, logger="src.services.llm_service")

    with pytest.raises(llm_service.LLMProviderLimitError):
        adapter.chat([{"role": "user", "content": "hello"}])

    assert any(record.levelno == logging.ERROR for record in caplog.records)
    assert any("quota or rate limit" in record.getMessage().lower() for record in caplog.records)
    assert not any(record.exc_info for record in caplog.records)


def test_app_returns_429_for_uncaught_llm_provider_limit_error(caplog):
    from fastapi import FastAPI

    from src.core.exception_handlers import llm_provider_limit_exception_handler

    app = FastAPI()
    app.add_exception_handler(
        llm_service.LLMProviderLimitError,
        llm_provider_limit_exception_handler,
    )

    route_path = "/__test__/llm-provider-limit"
    def raise_limit_error():
        raise llm_service.LLMProviderLimitError("quota exhausted")

    app.add_api_route(route_path, raise_limit_error, methods=["GET"])

    caplog.set_level(logging.ERROR, logger="src.core.exception_handlers")
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get(route_path)

    assert response.status_code == 429
    assert "quota" in response.json()["detail"].lower()
    assert any(record.levelno == logging.ERROR for record in caplog.records)
    assert not any(record.exc_info for record in caplog.records)
