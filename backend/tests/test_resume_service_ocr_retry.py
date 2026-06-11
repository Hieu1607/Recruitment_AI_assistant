import sys
import types
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pydantic_settings  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("pydantic_settings")

    class BaseSettings:
        pass

    stub.BaseSettings = BaseSettings
    sys.modules["pydantic_settings"] = stub

from src.services import resume_service  # noqa: E402


class _FakeResponse:
    def __init__(self, *, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Server Error",
                response=self,
            )

    def json(self):
        return self._json_data


def test_extract_text_via_hf_ocr_retries_transient_submit_failure(monkeypatch):
    submit_attempts = {"count": 0}

    def fake_post(url, files=None, timeout=None):
        submit_attempts["count"] += 1
        if submit_attempts["count"] == 1:
            return _FakeResponse(status_code=500, text="temporary failure")
        return _FakeResponse(
            json_data={"job_id": "job-123"},
            text='{"job_id":"job-123"}',
        )

    def fake_get(url, timeout=None):
        return _FakeResponse(json_data={"status": "done", "text": "OCR text"})

    monkeypatch.setattr(resume_service.requests, "post", fake_post)
    monkeypatch.setattr(resume_service.requests, "get", fake_get)
    monkeypatch.setattr(resume_service.requests, "delete", lambda *args, **kwargs: None)
    monkeypatch.setattr(resume_service.time, "sleep", lambda *_: None)
    monkeypatch.setattr(resume_service.settings, "HF_OCR_POLL_TIMEOUT", 5)
    monkeypatch.setattr(resume_service.settings, "HF_OCR_POLL_INTERVAL", 0)

    text = resume_service.extract_text_via_hf_ocr(b"%PDF-1.4 fake", "resume.pdf")

    assert text == "OCR text"
    assert submit_attempts["count"] == 2


def test_extract_text_via_hf_ocr_raises_after_exhausting_submit_retries(monkeypatch):
    def fake_post(url, files=None, timeout=None):
        return _FakeResponse(status_code=500, text="still failing")

    monkeypatch.setattr(resume_service.requests, "post", fake_post)
    monkeypatch.setattr(resume_service.time, "sleep", lambda *_: None)
    monkeypatch.setattr(resume_service.settings, "HF_OCR_POLL_TIMEOUT", 5)
    monkeypatch.setattr(resume_service.settings, "HF_OCR_POLL_INTERVAL", 0)

    with pytest.raises(requests.HTTPError):
        resume_service.extract_text_via_hf_ocr(b"%PDF-1.4 fake", "resume.pdf")
