import logging
import json
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


def test_retry_backoff_uses_groq_retry_after_with_exponential_cap():
    exc = RuntimeError("Please try again in 2.5s due to rate limiting")

    assert llm_service._retry_backoff_seconds(0, exc) == 2.5
    assert llm_service._retry_backoff_seconds(1, exc) == 5.0
    assert llm_service._retry_backoff_seconds(2, exc) == 10.0
    assert llm_service._retry_backoff_seconds(3, exc) == 10.0


def test_groq_adapter_uses_retry_after_with_exponential_backoff_cap(monkeypatch):
    sleep_calls = []
    attempts = {"count": 0}

    class FakeCompletions:
        def create(self, **kwargs):
            attempts["count"] += 1
            raise RuntimeError("temporary upstream failure; Please try again in 2.5s")

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "groq", types.SimpleNamespace(Groq=FakeGroq))
    monkeypatch.setattr(llm_service.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    adapter = llm_service._GroqAdapter(
        api_key="test-key",
        model="llama-test",
        temperature=0,
        max_tokens=128,
        timeout_seconds=5,
        max_retries=3,
    )

    with pytest.raises(llm_service.LLMProviderError):
        adapter.chat([{"role": "user", "content": "hello"}])

    assert attempts["count"] == 4
    assert sleep_calls == [2.5, 5.0, 10.0]


def test_groq_adapter_retries_rate_limits_before_raising_limit_error(monkeypatch):
    sleep_calls = []
    attempts = {"count": 0}

    class FakeCompletions:
        def create(self, **kwargs):
            attempts["count"] += 1
            raise RuntimeError(
                "Error code: 429 - {'error': {'message': 'Rate limit reached. Please try again in 1.5s', "
                "'code': 'rate_limit_exceeded'}}"
            )

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "groq", types.SimpleNamespace(Groq=FakeGroq))
    monkeypatch.setattr(llm_service.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    adapter = llm_service._GroqAdapter(
        api_key="test-key",
        model="llama-test",
        temperature=0,
        max_tokens=128,
        timeout_seconds=5,
        max_retries=2,
    )

    with pytest.raises(llm_service.LLMProviderLimitError):
        adapter.chat([{"role": "user", "content": "hello"}])

    assert attempts["count"] == 3
    assert sleep_calls == [1.5, 3.0]


def test_llm_provider_falls_back_to_shopaikey_on_groq_rate_limit(monkeypatch):
    class FakeGroqCompletions:
        def create(self, **kwargs):
            raise RuntimeError(
                "Error code: 429 - {'error': {'message': 'Rate limit reached. Please try again in 1.5s', "
                "'code': 'rate_limit_exceeded'}}"
            )

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = types.SimpleNamespace(completions=FakeGroqCompletions())

    class FakeHTTPResponse:
        def __init__(self, payload):
            self._payload = payload

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "fallback response",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )

    monkeypatch.setitem(sys.modules, "groq", types.SimpleNamespace(Groq=FakeGroq))
    monkeypatch.setattr(llm_service, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_service.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(llm_service.settings, "GROQ_API_KEY", "groq-key")
    monkeypatch.setattr(llm_service.settings, "GROQ_MODEL_NAME", "llama-3.1-8b-instant")
    monkeypatch.setattr(llm_service.settings, "SHOPAIKEY_API_KEY", "shop-key")
    monkeypatch.setattr(llm_service.settings, "SHOPAIKEY_BASE_URL", "https://api.shopaikey.com/v1")
    monkeypatch.setattr(llm_service.settings, "SHOPAIKEY_MODEL_NAME", "llama-3.1-8b")
    monkeypatch.setattr(llm_service.settings, "LLM_MAX_RETRIES", 0)

    provider = llm_service.LLMProvider(provider="groq")

    response = provider.chat([{"role": "user", "content": "hello"}])

    assert response.text == "fallback response"
    assert response.provider == "shopaikey"
    assert response.model == "llama-3.1-8b"
    assert captured["url"] == "https://api.shopaikey.com/v1/chat/completions"
    assert captured["body"]["model"] == "llama-3.1-8b"
    assert captured["body"]["messages"] == [{"role": "user", "content": "hello"}]


def test_llm_provider_falls_back_to_shopaikey_on_generic_groq_error(monkeypatch):
    class FakeGroqCompletions:
        def create(self, **kwargs):
            raise RuntimeError("upstream socket closed")

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = types.SimpleNamespace(completions=FakeGroqCompletions())

    class FakeHTTPResponse:
        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "generic fallback response",
                            }
                        }
                    ]
                }
            ).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setitem(sys.modules, "groq", types.SimpleNamespace(Groq=FakeGroq))
    monkeypatch.setattr(llm_service, "urlopen", lambda req, timeout: FakeHTTPResponse())
    monkeypatch.setattr(llm_service.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(llm_service.settings, "GROQ_API_KEY", "groq-key")
    monkeypatch.setattr(llm_service.settings, "GROQ_MODEL_NAME", "llama-3.1-8b-instant")
    monkeypatch.setattr(llm_service.settings, "SHOPAIKEY_API_KEY", "shop-key")
    monkeypatch.setattr(llm_service.settings, "SHOPAIKEY_BASE_URL", "https://api.shopaikey.com/v1")
    monkeypatch.setattr(llm_service.settings, "SHOPAIKEY_MODEL_NAME", "llama-3.1-8b")
    monkeypatch.setattr(llm_service.settings, "LLM_MAX_RETRIES", 0)

    provider = llm_service.LLMProvider(provider="groq")

    response = provider.generate("hello")

    assert response.text == "generic fallback response"
    assert response.provider == "shopaikey"


def test_groq_adapter_logs_warning_when_finish_reason_is_length(monkeypatch, caplog):
    class FakeCompletion:
        def __init__(self):
            self.choices = [
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content='{"ok": true}'),
                    finish_reason="length",
                )
            ]
            self.usage = types.SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=128,
                total_tokens=228,
            )

        def model_dump(self):
            return {
                "choices": [{"finish_reason": "length"}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 128,
                    "total_tokens": 228,
                },
            }

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeCompletion()

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "groq", types.SimpleNamespace(Groq=FakeGroq))
    caplog.set_level(logging.WARNING, logger="src.services.llm_service")

    adapter = llm_service._GroqAdapter(
        api_key="test-key",
        model="llama-test",
        temperature=0,
        max_tokens=128,
        timeout_seconds=5,
        max_retries=0,
    )

    response = adapter.chat([{"role": "user", "content": "hello"}])

    assert response.text == '{"ok": true}'
    assert any(
        "output token limit was reached" in record.getMessage().lower()
        and "provider=groq" in record.getMessage().lower()
        and "finish_reason=length" in record.getMessage().lower()
        for record in caplog.records
    )


def test_ollama_adapter_logs_warning_when_done_reason_is_length(monkeypatch, caplog):
    class FakeHTTPResponse:
        def read(self):
            return json.dumps(
                {
                    "message": {"content": '{"ok": true}'},
                    "done": True,
                    "done_reason": "length",
                    "prompt_eval_count": 24,
                    "eval_count": 128,
                }
            ).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(llm_service, "urlopen", lambda req, timeout: FakeHTTPResponse())
    caplog.set_level(logging.WARNING, logger="src.services.llm_service")

    adapter = llm_service._OllamaAdapter(
        base_url="http://ollama.local",
        model="llama-test",
        temperature=0,
        max_tokens=128,
        timeout_seconds=5,
        max_retries=0,
        keep_alive="5m",
    )

    response = adapter.chat([{"role": "user", "content": "hello"}])

    assert response.text == '{"ok": true}'
    assert any(
        "output token limit was reached" in record.getMessage().lower()
        and "provider=ollama" in record.getMessage().lower()
        and "finish_reason=length" in record.getMessage().lower()
        for record in caplog.records
    )
