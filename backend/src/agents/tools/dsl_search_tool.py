from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from src.models.candidate import CandidateProfile
from src.services.llm.llm_client import LLMClientError, LLMRequest, generate_json


@dataclass
class DSLFilter:
    field: str
    op: str
    value: Any


@dataclass
class DSLQueryIntent:
    logic: str
    filters: list[DSLFilter]
    limit: int


@dataclass
class DSLSearchResult:
    candidate_ids: list[str]
    matched_count: int
    trace: dict[str, Any]


_TEXT_FIELDS = {
    "location_normalized",
    "current_job_title",
    "skills_text",
    "major",
    "cpa",
    "certifications_text",
    "languages_text",
}
_NUMBER_FIELDS = {"experience_years"}
_BOOLEAN_FIELDS = {"educated", "ever_studied_abroad"}

_ALLOWED_FIELDS = _TEXT_FIELDS | _NUMBER_FIELDS | _BOOLEAN_FIELDS
_ALLOWED_OPS = {
    "eq",
    "contains",
    "contains_any",
    "contains_all",
    "gte",
    "lte",
    "between",
    "exists",
}

_FIELD_TO_COLUMN = {
    "location_normalized": CandidateProfile.location_normalized,
    "current_job_title": CandidateProfile.current_job_title,
    "skills_text": CandidateProfile.skills_text,
    "major": CandidateProfile.major,
    "cpa": CandidateProfile.cpa,
    "certifications_text": CandidateProfile.certifications_text,
    "languages_text": CandidateProfile.languages_text,
    "experience_years": CandidateProfile.experience_years,
    "educated": CandidateProfile.educated,
    "ever_studied_abroad": CandidateProfile.ever_studied_abroad,
}

_TABLE_FORMAT = """
candidate_profiles columns (supported by DSL):
- educated: boolean
- ever_studied_abroad: boolean
- experience_years: number
- location_normalized: text
- current_job_title: text
- skills_text: text
- major: text
- cpa: text
- certifications_text: text
- languages_text: text

Allowed operators per type:
- boolean: eq
- number: eq, gte, lte, between
- text: eq, contains, contains_any, contains_all, exists
""".strip()

_TECH_STACK = """
Current technology stack:
- Backend language: Python
- API framework: FastAPI
- ORM/query builder: SQLAlchemy
- Database: PostgreSQL
- LLM providers: Groq or Ollama
- Execution policy: Return only safe, deterministic JSON intent (no SQL text)
""".strip()


def _clamp_limit(limit: int) -> int:
    return max(1, min(int(limit), 500))


def _safe_text(value: Any, max_len: int = 120) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    return text[:max_len]


def _safe_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _normalize_logic(value: Any) -> str:
    if isinstance(value, str) and value.lower() == "or":
        return "or"
    return "and"


def _sanitize_filter(raw_filter: Any) -> DSLFilter | None:
    if not isinstance(raw_filter, dict):
        return None

    field = str(raw_filter.get("field", "")).strip()
    op = str(raw_filter.get("op", "")).strip().lower()
    if field not in _ALLOWED_FIELDS or op not in _ALLOWED_OPS:
        return None

    raw_value = raw_filter.get("value")

    if field in _BOOLEAN_FIELDS:
        if op != "eq":
            return None
        parsed_bool = _safe_bool(raw_value)
        if parsed_bool is None:
            return None
        return DSLFilter(field=field, op=op, value=parsed_bool)

    if field in _NUMBER_FIELDS:
        if op == "between":
            if not isinstance(raw_value, list) or len(raw_value) != 2:
                return None
            min_value = _safe_number(raw_value[0])
            max_value = _safe_number(raw_value[1])
            if min_value is None or max_value is None:
                return None
            low, high = sorted([min_value, max_value])
            return DSLFilter(field=field, op=op, value=[low, high])

        parsed_number = _safe_number(raw_value)
        if parsed_number is None or op not in {"eq", "gte", "lte"}:
            return None
        return DSLFilter(field=field, op=op, value=parsed_number)

    if op == "exists":
        return DSLFilter(field=field, op=op, value=True)

    if op in {"contains_any", "contains_all"}:
        if not isinstance(raw_value, list):
            return None
        normalized_values = [
            item for item in (_safe_text(item) for item in raw_value) if item
        ]
        if not normalized_values:
            return None
        return DSLFilter(field=field, op=op, value=normalized_values[:10])

    if op in {"eq", "contains"}:
        text_value = _safe_text(raw_value)
        if not text_value:
            return None
        return DSLFilter(field=field, op=op, value=text_value)

    return None


def _parse_intent(payload: dict[str, Any], requested_limit: int) -> DSLQueryIntent:
    root = (
        payload.get("queryIntent")
        if isinstance(payload.get("queryIntent"), dict)
        else payload
    )
    logic = _normalize_logic(root.get("logic"))

    parsed_filters: list[DSLFilter] = []
    raw_filters = root.get("filters")
    if isinstance(raw_filters, list):
        for raw_filter in raw_filters:
            parsed = _sanitize_filter(raw_filter)
            if parsed is not None:
                parsed_filters.append(parsed)

    limit_from_intent = root.get("limit")
    if isinstance(limit_from_intent, (int, float)):
        final_limit = _clamp_limit(min(int(limit_from_intent), requested_limit))
    else:
        final_limit = _clamp_limit(requested_limit)

    return DSLQueryIntent(logic=logic, filters=parsed_filters, limit=final_limit)


def _fallback_intent(question: str, requested_limit: int) -> DSLQueryIntent:
    lowered = question.lower().strip()
    filters: list[DSLFilter] = []

    if any(keyword in lowered for keyword in ("educated", "degree", "graduated")):
        filters.append(DSLFilter(field="educated", op="eq", value=True))

    if any(
        keyword in lowered
        for keyword in ("abroad", "international study", "studied overseas")
    ):
        filters.append(DSLFilter(field="ever_studied_abroad", op="eq", value=True))

    if any(keyword in lowered for keyword in ("cpa", "certified public accountant")):
        filters.append(DSLFilter(field="cpa", op="exists", value=True))

    years_match = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)", lowered)
    if years_match:
        years = float(years_match.group(1))
        filters.append(DSLFilter(field="experience_years", op="gte", value=years))

    location_match = re.search(r"(?:in|from|based in)\s+([a-zA-Z\s]{2,40})", lowered)
    if location_match:
        location = location_match.group(1).strip()
        if location:
            filters.append(
                DSLFilter(field="location_normalized", op="contains", value=location)
            )

    skill_match = re.search(
        r"(?:with|having)\s+([a-zA-Z0-9\s+\-#/]{2,50})\s+(?:skill|skills)", lowered
    )
    if skill_match:
        skill = skill_match.group(1).strip()
        if skill:
            filters.append(DSLFilter(field="skills_text", op="contains", value=skill))

    if not filters and any(
        keyword in lowered
        for keyword in ("engineer", "developer", "analyst", "accountant", "manager")
    ):
        filters.append(
            DSLFilter(
                field="current_job_title",
                op="contains_any",
                value=["engineer", "developer", "analyst", "accountant", "manager"],
            )
        )

    return DSLQueryIntent(
        logic="and", filters=filters, limit=_clamp_limit(requested_limit)
    )


def _as_expression(item: DSLFilter) -> ColumnElement[bool] | None:
    column = _FIELD_TO_COLUMN.get(item.field)
    if column is None:
        return None

    if item.op == "eq":
        if item.field in _TEXT_FIELDS:
            return column.ilike(str(item.value))
        return column == item.value

    if item.op == "gte":
        return column >= item.value

    if item.op == "lte":
        return column <= item.value

    if item.op == "between":
        low, high = item.value
        return and_(column >= low, column <= high)

    if item.op == "contains":
        pattern = f"%{str(item.value)}%"
        return column.ilike(pattern)

    if item.op == "contains_any":
        values = [str(value) for value in item.value]
        return or_(*[column.ilike(f"%{value}%") for value in values])

    if item.op == "contains_all":
        values = [str(value) for value in item.value]
        return and_(*[column.ilike(f"%{value}%") for value in values])

    if item.op == "exists":
        return and_(column.is_not(None), column != "")

    return None


def _build_intent_prompt(question: str, requested_limit: int) -> str:
    return (
        "Convert the recruiter question into a structured JSON query intent for candidate filtering.\n\n"
        f"Recruiter question: {question}\n\n"
        f"{_TABLE_FORMAT}\n\n"
        f"{_TECH_STACK}\n\n"
        "Output requirements:\n"
        "- Return valid JSON only, no markdown, no explanation.\n"
        "- Use top-level key queryIntent.\n"
        "- queryIntent must contain: logic, filters, limit.\n"
        "- logic must be either and/or.\n"
        "- filters must be an array of objects: {field, op, value}.\n"
        "- Use only supported fields and operators.\n"
        "- Keep limit <= requested_limit.\n\n"
        f"requested_limit: {_clamp_limit(requested_limit)}\n\n"
        "Expected JSON shape:\n"
        "{\n"
        '  "queryIntent": {\n'
        '    "logic": "and",\n'
        '    "filters": [{"field": "experience_years", "op": "gte", "value": 3}],\n'
        f'    "limit": {_clamp_limit(requested_limit)}\n'
        "  }\n"
        "}"
    )


def run_dsl_search(
    session: Session, question: str, limit: int = 200
) -> DSLSearchResult:
    requested_limit = _clamp_limit(limit)
    prompt = _build_intent_prompt(question=question, requested_limit=requested_limit)

    fallback_reason: str | None = None
    try:
        llm_payload = generate_json(
            LLMRequest(
                prompt=prompt,
                system_prompt=(
                    "You convert recruiter questions to strict JSON DSL intents. "
                    "Do not output SQL. Do not output markdown."
                ),
                temperature=0.0,
            )
        )
        intent = _parse_intent(llm_payload, requested_limit=requested_limit)
        if not intent.filters:
            fallback_reason = "llm_intent_empty_filters"
            intent = _fallback_intent(
                question=question, requested_limit=requested_limit
            )
    except (LLMClientError, TypeError, ValueError) as exc:
        fallback_reason = f"llm_failed:{exc.__class__.__name__}"
        intent = _fallback_intent(question=question, requested_limit=requested_limit)

    expressions = [
        expression
        for expression in (_as_expression(item) for item in intent.filters)
        if expression is not None
    ]

    if not expressions:
        return DSLSearchResult(
            candidate_ids=[],
            matched_count=0,
            trace={
                "tool": "dsl_search",
                "intent": {
                    "logic": intent.logic,
                    "filters": [
                        {
                            "field": item.field,
                            "op": item.op,
                            "value": item.value,
                        }
                        for item in intent.filters
                    ],
                    "limit": intent.limit,
                },
                "fallback_reason": fallback_reason,
                "matched_via": "no_valid_filters",
            },
        )

    where_clause = or_(*expressions) if intent.logic == "or" else and_(*expressions)
    stmt = select(CandidateProfile.id).where(where_clause).limit(intent.limit)
    rows = list(session.scalars(stmt))
    candidate_ids = [
        str(row) if isinstance(row, uuid.UUID) else str(row) for row in rows
    ]

    return DSLSearchResult(
        candidate_ids=candidate_ids,
        matched_count=len(candidate_ids),
        trace={
            "tool": "dsl_search",
            "intent": {
                "logic": intent.logic,
                "filters": [
                    {
                        "field": item.field,
                        "op": item.op,
                        "value": item.value,
                    }
                    for item in intent.filters
                ],
                "limit": intent.limit,
            },
            "fallback_reason": fallback_reason,
            "matched_via": "dsl_filters",
        },
    )
