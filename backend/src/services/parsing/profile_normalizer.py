from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.services.llm.llm_client import LLMClientError, LLMClientFactory, LLMRequest
from src.services.parsing.resume_extractor import ResumeExtractionResult

_SYSTEM_PROMPT = (
    "You are an expert CV/Resume parser that extracts structured candidate data."
)
_DEFAULT_NAME = "Unknown Candidate"
_MAX_PROMPT_CHARS = 24_000


def _compact_text(value: Any, *, max_len: int = 8_000) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > max_len:
        return text[:max_len].rstrip()
    return text


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    lowered = str(value).strip().lower()
    if lowered in {"true", "yes", "1"}:
        return True
    if lowered in {"false", "no", "0"}:
        return False
    return default


def _to_experience_years(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return round(float(value), 1)
    if value is None:
        return None

    cleaned = str(value).strip().lower().replace(",", ".")
    for token in (
        cleaned.replace("years", " ")
        .replace("year", " ")
        .replace("yrs", " ")
        .replace("yr", " ")
        .replace("+", " ")
        .split()
    ):
        try:
            return round(float(token), 1)
        except ValueError:
            continue
    return None


def _extract_json_payload(raw: str) -> dict[str, Any]:
    candidate = raw.strip()
    if not candidate:
        return {}

    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    try:
        parsed = json.loads(candidate[start : end + 1])
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _build_prompt(cv_text: str) -> str:
    clipped = cv_text[:_MAX_PROMPT_CHARS]
    return f"""
Analyze the CV text and return ONLY one valid JSON object (no markdown, no explanation).

Required schema:
{{
  "name": string|null,
  "phone": string|null,
  "email": string|null,
  "location": string|null,
  "contact": string|null,
  "current_job_title": string|null,
  "educated": boolean,
  "ever_studied_abroad": boolean,
  "major": string|null,
  "cpa": string|null,
  "education": string|null,
  "experience": string|null,
  "experience_years": number|null,
  "skills": string|null,
  "languages": string|null,
  "projects": string|null,
  "summary": string|null,
  "achievements": string|null,
  "publications": string|null,
  "certifications": string|null,
  "references": string|null,
  "other": string|null
}}

Rules:
- Use null when unknown.
- Keep extracted text concise and faithful to CV.
- "experience_years" must be numeric (e.g., 3 or 4.5) or null.

CV text:
{clipped}
""".strip()


@dataclass
class NormalizedProfile:
    full_name: str
    phone: str | None
    email: str | None
    location_normalized: str | None
    contact: str | None
    current_job_title: str | None
    educated: bool
    ever_studied_abroad: bool
    major: str | None
    cpa: str | None
    education_text: str | None
    experience_text: str | None
    experience_years: float | None
    skills_text: str | None
    languages_text: str | None
    projects_text: str | None
    summary_text: str | None
    achievements_text: str | None
    publications_text: str | None
    certifications_text: str | None
    references_text: str | None
    other_text: str | None
    parse_method: str
    llm_invoked: bool
    llm_succeeded: bool


def _fallback_profile() -> NormalizedProfile:
    return NormalizedProfile(
        full_name=_DEFAULT_NAME,
        phone=None,
        email=None,
        location_normalized=None,
        contact=None,
        current_job_title=None,
        educated=False,
        ever_studied_abroad=False,
        major=None,
        cpa=None,
        education_text=None,
        experience_text=None,
        experience_years=None,
        skills_text=None,
        languages_text=None,
        projects_text=None,
        summary_text=None,
        achievements_text=None,
        publications_text=None,
        certifications_text=None,
        references_text=None,
        other_text=None,
        parse_method="llm_fallback",
        llm_invoked=True,
        llm_succeeded=False,
    )


def normalize_profile(extraction: ResumeExtractionResult) -> NormalizedProfile:
    if not extraction.full_text.strip():
        return NormalizedProfile(
            full_name=_DEFAULT_NAME,
            phone=None,
            email=None,
            location_normalized=None,
            contact=None,
            current_job_title=None,
            educated=False,
            ever_studied_abroad=False,
            major=None,
            cpa=None,
            education_text=None,
            experience_text=None,
            experience_years=None,
            skills_text=None,
            languages_text=None,
            projects_text=None,
            summary_text=None,
            achievements_text=None,
            publications_text=None,
            certifications_text=None,
            references_text=None,
            other_text=None,
            parse_method="empty_text_fallback",
            llm_invoked=False,
            llm_succeeded=False,
        )

    prompt = _build_prompt(extraction.full_text)
    payload: dict[str, Any] = {}

    for temperature in (0.0, 0.1):
        try:
            raw = LLMClientFactory.create().generate(
                LLMRequest(
                    prompt=prompt, system_prompt=_SYSTEM_PROMPT, temperature=temperature
                )
            )
        except LLMClientError:
            continue

        payload = _extract_json_payload(raw)
        if payload:
            break

    if not payload:
        return _fallback_profile()

    full_name = _compact_text(payload.get("name"), max_len=255) or _DEFAULT_NAME

    return NormalizedProfile(
        full_name=full_name,
        phone=_compact_text(payload.get("phone"), max_len=50),
        email=_compact_text(payload.get("email"), max_len=320),
        location_normalized=_compact_text(payload.get("location"), max_len=255),
        contact=_compact_text(payload.get("contact"), max_len=255),
        current_job_title=_compact_text(payload.get("current_job_title"), max_len=255),
        educated=_to_bool(payload.get("educated"), default=False),
        ever_studied_abroad=_to_bool(payload.get("ever_studied_abroad"), default=False),
        major=_compact_text(payload.get("major"), max_len=255),
        cpa=_compact_text(payload.get("cpa"), max_len=255),
        education_text=_compact_text(payload.get("education")),
        experience_text=_compact_text(payload.get("experience")),
        experience_years=_to_experience_years(
            payload.get("experience_years") or payload.get("experiment_years")
        ),
        skills_text=_compact_text(payload.get("skills")),
        languages_text=_compact_text(payload.get("languages")),
        projects_text=_compact_text(payload.get("projects")),
        summary_text=_compact_text(payload.get("summary")),
        achievements_text=_compact_text(payload.get("achievements")),
        publications_text=_compact_text(payload.get("publications")),
        certifications_text=_compact_text(payload.get("certifications")),
        references_text=_compact_text(payload.get("references")),
        other_text=_compact_text(payload.get("other")),
        parse_method="llm",
        llm_invoked=True,
        llm_succeeded=True,
    )
