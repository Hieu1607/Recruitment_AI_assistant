from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from src.models.candidate import CandidateProfile


@dataclass
class SQLSearchResult:
    candidate_ids: list[str]
    matched_count: int
    trace: dict


def _has_any_keyword(question: str, keywords: tuple[str, ...]) -> bool:
    lowered = question.lower()
    return any(keyword in lowered for keyword in keywords)


def run_sql_search(session: Session, question: str, limit: int = 200) -> SQLSearchResult:
    lowered = question.lower().strip()
    filters = []
    trace_filters: list[str] = []

    if _has_any_keyword(lowered, ("educated", "degree", "graduated")):
        filters.append(CandidateProfile.educated.is_(True))
        trace_filters.append("educated=true")

    if _has_any_keyword(lowered, ("abroad", "international study", "studied overseas")):
        filters.append(CandidateProfile.ever_studied_abroad.is_(True))
        trace_filters.append("ever_studied_abroad=true")

    if _has_any_keyword(lowered, ("cpa", "certified public accountant")):
        filters.append(CandidateProfile.cpa.is_not(None))
        trace_filters.append("cpa is not null")

    years_match = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)", lowered)
    if years_match:
        years = float(years_match.group(1))
        filters.append(CandidateProfile.experience_years.is_not(None))
        filters.append(CandidateProfile.experience_years >= years)
        trace_filters.append(f"experience_years>={years}")

    location_match = re.search(r"(?:in|from|based in)\s+([a-zA-Z\s]{2,40})", lowered)
    if location_match:
        location = location_match.group(1).strip()
        pattern = f"%{location}%"
        filters.append(CandidateProfile.location_normalized.ilike(pattern))
        trace_filters.append(f"location ilike {pattern}")

    skill_match = re.search(r"(?:with|having)\s+([a-zA-Z0-9\s+\-#/]{2,50})\s+(?:skill|skills)", lowered)
    if skill_match:
        skill = skill_match.group(1).strip()
        pattern = f"%{skill}%"
        filters.append(CandidateProfile.skills_text.ilike(pattern))
        trace_filters.append(f"skills ilike {pattern}")

    stmt = select(CandidateProfile.id)
    if filters:
        stmt = stmt.where(and_(*filters))
    elif _has_any_keyword(lowered, ("engineer", "developer", "analyst", "accountant", "manager")):
        title_filters = [
            CandidateProfile.current_job_title.ilike("%engineer%"),
            CandidateProfile.current_job_title.ilike("%developer%"),
            CandidateProfile.current_job_title.ilike("%analyst%"),
            CandidateProfile.current_job_title.ilike("%accountant%"),
            CandidateProfile.current_job_title.ilike("%manager%"),
        ]
        stmt = stmt.where(or_(*title_filters))
        trace_filters.append("title keyword heuristic")

    rows = list(session.scalars(stmt.limit(max(1, min(limit, 500)))))
    candidate_ids = [str(row) if isinstance(row, uuid.UUID) else str(row) for row in rows]
    return SQLSearchResult(
        candidate_ids=candidate_ids,
        matched_count=len(candidate_ids),
        trace={
            "tool": "sql_search",
            "filters": trace_filters,
        },
    )
