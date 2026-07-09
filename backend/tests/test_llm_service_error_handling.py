import logging
import json
import pytest
from fastapi.testclient import TestClient

from src.services import llm_service


def test_shopaikey_rate_limit_is_wrapped_and_logged(monkeypatch, caplog):
    class FakeHTTPErrorResponse:
        def read(self):
            return json.dumps(
                {
                    "error": {
                        "message": "Rate limit reached for model on tokens per day",
                        "code": "rate_limit_exceeded",
                    }
                }
            ).encode("utf-8")

        def close(self):
            return None

    def fake_urlopen(req, timeout):
        from urllib.error import HTTPError

        raise HTTPError(
            url=req.full_url,
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=FakeHTTPErrorResponse(),
        )

    monkeypatch.setattr(llm_service, "urlopen", fake_urlopen)
    adapter = llm_service._ShopAIKeyAdapter(
        api_key="test-key",
        base_url="https://api.shopaikey.com/v1",
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


def test_retry_backoff_uses_retry_after_with_exponential_cap():
    exc = RuntimeError("Please try again in 2.5s due to rate limiting")

    assert llm_service._retry_backoff_seconds(0, exc) == 2.5
    assert llm_service._retry_backoff_seconds(1, exc) == 5.0
    assert llm_service._retry_backoff_seconds(2, exc) == 10.0
    assert llm_service._retry_backoff_seconds(3, exc) == 10.0


def test_shopaikey_adapter_uses_retry_after_with_exponential_backoff_cap(monkeypatch):
    sleep_calls = []
    attempts = {"count": 0}

    class FakeHTTPResponse:
        def read(self):
            attempts["count"] += 1
            raise RuntimeError("temporary upstream failure; Please try again in 2.5s")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(llm_service, "urlopen", lambda req, timeout: FakeHTTPResponse())
    monkeypatch.setattr(llm_service.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    adapter = llm_service._ShopAIKeyAdapter(
        api_key="test-key",
        base_url="https://api.shopaikey.com/v1",
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


def test_shopaikey_adapter_logs_each_attempt_timing_without_prompt_content(monkeypatch, caplog):
    attempts = {"count": 0}

    class FakeHTTPResponse:
        def read(self):
            return json.dumps(
                {
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                }
            ).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("upstream timed out")
        return FakeHTTPResponse()

    monkeypatch.setattr(llm_service, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_service.time, "sleep", lambda seconds: None)
    caplog.set_level(logging.INFO, logger="src.services.llm_service")
    adapter = llm_service._ShopAIKeyAdapter(
        api_key="secret-test-key",
        base_url="https://api.shopaikey.com/v1",
        model="gpt-test",
        temperature=0,
        max_tokens=128,
        timeout_seconds=5,
        max_retries=1,
    )

    response = adapter.chat([{"role": "user", "content": "private prompt"}])

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert response.text == "ok"
    assert "shopaikey_request_attempt_started" in messages
    assert "shopaikey_request_attempt_failed" in messages
    assert "shopaikey_request_attempt_succeeded" in messages
    assert "attempt=1/2" in messages
    assert "attempt=2/2" in messages
    assert "error_type=TimeoutError" in messages
    assert "retrying=True" in messages
    assert "backoff_seconds=1.000" in messages
    assert "input_chars=14" in messages
    assert "secret-test-key" not in messages
    assert "private prompt" not in messages


def test_shopaikey_adapter_retries_rate_limits_before_raising_limit_error(monkeypatch):
    sleep_calls = []
    attempts = {"count": 0}

    class FakeHTTPErrorResponse:
        def read(self):
            attempts["count"] += 1
            return json.dumps(
                {
                    "error": {
                        "message": "Rate limit reached. Please try again in 1.5s",
                        "code": "rate_limit_exceeded",
                    }
                }
            ).encode("utf-8")

        def close(self):
            return None

    def fake_urlopen(req, timeout):
        from urllib.error import HTTPError

        raise HTTPError(
            url=req.full_url,
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=FakeHTTPErrorResponse(),
        )

    monkeypatch.setattr(llm_service, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_service.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    adapter = llm_service._ShopAIKeyAdapter(
        api_key="test-key",
        base_url="https://api.shopaikey.com/v1",
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


def test_llm_provider_defaults_to_shopaikey(monkeypatch):
    class FakeHTTPResponse:
        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "default provider response",
                            }
                        }
                    ]
                }
            ).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout):
        return FakeHTTPResponse()

    monkeypatch.setattr(llm_service, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_service.settings, "SHOPAIKEY_API_KEY", "shop-key")
    monkeypatch.setattr(llm_service.settings, "SHOPAIKEY_BASE_URL", "https://api.shopaikey.com/v1")
    monkeypatch.setattr(llm_service.settings, "SHOPAIKEY_MODEL_NAME", "llama-3.1-8b")
    monkeypatch.setattr(llm_service.settings, "LLM_MAX_RETRIES", 0)

    provider = llm_service.LLMProvider()

    response = provider.generate("hello")

    assert response.text == "default provider response"
    assert response.provider == "shopaikey"
    assert response.model == "llama-3.1-8b"


def test_shopaikey_adapter_logs_warning_when_finish_reason_is_length(monkeypatch, caplog):
    class FakeHTTPResponse:
        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {"content": '{"ok": true}'},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 128,
                        "total_tokens": 228,
                    },
                }
            ).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(llm_service, "urlopen", lambda req, timeout: FakeHTTPResponse())
    caplog.set_level(logging.WARNING, logger="src.services.llm_service")

    adapter = llm_service._ShopAIKeyAdapter(
        api_key="test-key",
        base_url="https://api.shopaikey.com/v1",
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
        and "provider=shopaikey" in record.getMessage().lower()
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
