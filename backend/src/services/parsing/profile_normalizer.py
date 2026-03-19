from __future__ import annotations

import re
from dataclasses import dataclass

from src.services.parsing.resume_extractor import ResumeExtractionResult

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"\+?\d[\d\s().-]{7,}\d")

EDUCATION_MARKERS = ("bachelor", "master", "phd", "university", "college")
ABROAD_MARKERS = ("exchange", "study abroad", "erasmus", "international campus")


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
    education_text: str | None
    experience_text: str | None
    skills_text: str | None
    summary_text: str | None


def _extract_name(full_text: str) -> str:
    for line in full_text.splitlines():
        cleaned = line.strip()
        if cleaned and len(cleaned.split()) >= 2 and len(cleaned) <= 80:
            return cleaned
    return "Unknown Candidate"


def _find_section(full_text: str, keywords: tuple[str, ...]) -> str | None:
    lines = full_text.splitlines()
    hits = [line.strip() for line in lines if any(k in line.lower() for k in keywords) and line.strip()]
    return "\n".join(hits[:8]) if hits else None


def normalize_profile(extraction: ResumeExtractionResult) -> NormalizedProfile:
    full_text = extraction.full_text
    lowered = full_text.lower()

    email_match = EMAIL_PATTERN.search(full_text)
    phone_match = PHONE_PATTERN.search(full_text)

    education_text = _find_section(full_text, ("education", "university", "college", "degree"))
    experience_text = _find_section(full_text, ("experience", "employment", "work history"))
    skills_text = _find_section(full_text, ("skills", "technologies", "stack"))

    first_lines = [line.strip() for line in full_text.splitlines()[:5] if line.strip()]
    summary_text = " ".join(first_lines) if first_lines else None

    educated = any(marker in lowered for marker in EDUCATION_MARKERS)
    studied_abroad = any(marker in lowered for marker in ABROAD_MARKERS)

    return NormalizedProfile(
        full_name=_extract_name(full_text),
        phone=phone_match.group(0) if phone_match else None,
        email=email_match.group(0) if email_match else None,
        location_normalized=None,
        contact=None,
        current_job_title=None,
        educated=educated,
        ever_studied_abroad=studied_abroad,
        education_text=education_text,
        experience_text=experience_text,
        skills_text=skills_text,
        summary_text=summary_text,
    )
