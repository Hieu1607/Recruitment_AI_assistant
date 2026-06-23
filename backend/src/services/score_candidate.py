from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session, joinedload

from src.models.candidate_profile import CandidateProfile
from src.models.enums import MatchRunStatus
from src.models.job_matching import JobDescription, MatchResult, MatchRun
from src.models.resume_document import ResumeDocument
from src.prompts.build_prompts import build_prompts
from src.services.llm_service import (
    LLMProvider,
    LLMProviderError,
    LLMProviderLimitError,
    is_provider_limit_error as is_llm_provider_limit_error,
)
from src.services.scoring_batching import build_scoring_batch_plan
from src.services.scoring_debug import ScoringDebugLogger, preview_text
from src.services.scoring_errors import ScoringProviderLimitError
from src.services.token_budget import BudgetWindow, estimate_json_tokens, estimate_tokens
from src.core.config import settings

logger = logging.getLogger(__name__)


SUPPORTED_SCORING_SECTIONS = tuple(build_prompts.SUPPORTED_SCORING_SECTIONS)
SUPPORTED_CRITERION_TYPES = {"must_have", "semantic", "upper_bound"}
SCORING_LLM_MAX_TOKENS = 4096
PROVIDER_LIMIT_ERROR_MARKERS = (
    "429",
    "quota",
    "rate limit",
    "rate_limit_exceeded",
    "tokens per day",
    "requests per day",
    "too many requests",
)
NUMERIC_OPERATORS = {">=", ">", "<=", "<", "==", "="}
NUMERIC_COMPARISON_OPERATORS = {">=", ">", "<=", "<"}
EQUALITY_OPERATORS = {"==", "="}
BOOLEAN_OPERATORS = EQUALITY_OPERATORS
SUPPORTED_MEASURABLE_FIELDS: Dict[str, Dict[str, Any]] = {
    "experience_years": {
        "value_type": "number",
        "operators": NUMERIC_OPERATORS,
    },
    "graduation_status": {
        "value_type": "string",
        "operators": BOOLEAN_OPERATORS,
        "allowed_values": {"graduated", "final_year", "studying", "unknown"},
    },
    "ever_studied_abroad": {
        "value_type": "boolean",
        "operators": BOOLEAN_OPERATORS,
    },
}


def _ui_language() -> str:
    return "en" if str(settings.APP_UI_LANGUAGE or "").strip().lower().startswith("en") else "vi"


def _flatten_exception_messages(exc: Exception) -> str:
    parts: List[str] = []
    current: Optional[BaseException] = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).strip()
        if message:
            parts.append(message.lower())
        current = current.__cause__ or current.__context__
    return " | ".join(parts)


def _is_provider_limit_error(exc: Exception) -> bool:
    if isinstance(exc, ScoringProviderLimitError):
        return True
    if isinstance(exc, LLMProviderLimitError):
        return True
    if not isinstance(exc, LLMProviderError):
        return False
    if is_llm_provider_limit_error(exc):
        return True
    flattened = _flatten_exception_messages(exc)
    return any(marker in flattened for marker in PROVIDER_LIMIT_ERROR_MARKERS)


def _duration_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)


def _merge_semantic_scores(
    target: Dict[str, Dict[str, Any]],
    update: Dict[str, Dict[str, Any]],
) -> None:
    for candidate_id, candidate_result in (update or {}).items():
        if not isinstance(candidate_result, dict):
            continue
        existing = target.setdefault(str(candidate_id), {"criteria": {}})
        if candidate_result.get("rationale") and not existing.get("rationale"):
            existing["rationale"] = candidate_result.get("rationale")
        existing_criteria = existing.setdefault("criteria", {})
        incoming_criteria = candidate_result.get("criteria") or {}
        if isinstance(incoming_criteria, dict):
            existing_criteria.update(incoming_criteria)


def _mark_match_run_failed(db: Session, match_run_id: uuid.UUID) -> None:
    match_run_db = db.get(MatchRun, match_run_id)
    if match_run_db is None:
        return
    match_run_db.run_status = MatchRunStatus.FAILED.value
    match_run_db.completed_at = datetime.now(timezone.utc)
    db.commit()


def _clamp_score(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return round(max(0.0, min(100.0, numeric)), 2)


def _normalize_llm_score(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if 0.0 <= numeric <= 1.0:
        numeric *= 100.0
    return _clamp_score(numeric)


def _parse_json_object(raw_text: str) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for candidate in _json_parse_candidates(raw_text):
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict):
                raise ValueError("LLM response is not a JSON object")
            return parsed
        except Exception as exc:
            last_error = exc
            continue
    raise ValueError(str(last_error) if last_error else "LLM did not return a valid JSON object")


def _strip_markdown_fences(text: str) -> str:
    content = (text or "").strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            content = "\n".join(lines[1:-1]).strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()
    return content


def _extract_json_slice(text: str) -> str:
    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        return text[start_obj : end_obj + 1]
    return text


def _repair_common_json_issues(text: str) -> str:
    repaired = text.strip()
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    opens = repaired.count("{")
    closes = repaired.count("}")
    if opens > closes:
        repaired += "}" * (opens - closes)
    opens = repaired.count("[")
    closes = repaired.count("]")
    if opens > closes:
        repaired += "]" * (opens - closes)
    return repaired


def _json_parse_candidates(raw_text: str) -> List[str]:
    stripped = _strip_markdown_fences(raw_text)
    sliced = _extract_json_slice(stripped)
    repaired = _repair_common_json_issues(sliced)
    candidates = []
    for item in [stripped, sliced, repaired]:
        if item and item not in candidates:
            candidates.append(item)
    return candidates


def _profile_to_candidate_dict(profile: CandidateProfile) -> Dict[str, Any]:
    """Convert a CandidateProfile ORM row to the dict shape expected by the scoring pipeline."""
    candidate_name = str(profile.full_name or "").strip()
    resume_file_name = str(getattr(getattr(profile, "resume_document", None), "original_file_name", "") or "").strip()
    display_name = candidate_name or resume_file_name
    return {
        "id": str(profile.id),
        "full_name": candidate_name,
        "resume_file_name": resume_file_name,
        "display_name": display_name,
        "current_job_title": profile.current_job_title,
        "graduation_status": profile.graduation_status,
        "ever_studied_abroad": bool(profile.ever_studied_abroad),
        "experience_years": float(profile.experience_years) if profile.experience_years is not None else None,
        "education_text": profile.education_text,
        "experience_text": profile.experience_text,
        "skills_text": profile.skills_text,
        "projects_text": profile.projects_text,
        "summary_text": profile.summary_text,
        "languages_text": profile.languages_text,
        "achievements_text": profile.achievements_text,
        "certifications_text": profile.certifications_text,
        "publications_text": profile.publications_text,
        "other_text": profile.other_text,
    }


def _attach_candidate_metadata(score_data: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(score_data)
    candidate_name = str(candidate.get("full_name") or "").strip()
    resume_file_name = str(candidate.get("resume_file_name") or "").strip()
    normalized["candidateName"] = candidate_name
    normalized["resumeFileName"] = resume_file_name
    normalized["candidateDisplayName"] = str(candidate.get("display_name") or candidate_name or resume_file_name).strip()
    return normalized


def _scoring_llm_provider() -> LLMProvider:
    return LLMProvider(max_tokens=max(settings.LLM_MAX_TOKENS, SCORING_LLM_MAX_TOKENS))


def _parse_llm_scores(raw_text: str) -> List[Dict[str, Any]]:
    parsed = _parse_json_object(raw_text)
    scores = parsed.get("scores", [])
    if not isinstance(scores, list):
        raise ValueError("LLM response 'scores' field is not a list")
    return scores


def _parse_rubric_response(raw_text: str) -> Dict[str, Any]:
    parsed = _parse_json_object(raw_text)
    if "rubric" in parsed and isinstance(parsed["rubric"], dict):
        parsed = parsed["rubric"]
    criteria = parsed.get("criteria", [])
    if not isinstance(criteria, list):
        raise ValueError("Rubric response 'criteria' field is not a list")
    return {"criteria": criteria}


def _build_scoring_job_description_text(
    *,
    public_job_description: str,
    hidden_text: Optional[str] = None,
) -> str:
    """Combine public JD and recruiter-only criteria for scoring prompts."""
    parts = [
        "Public job description:",
        (public_job_description or "").strip(),
    ]
    hidden = (hidden_text or "").strip()
    if hidden:
        parts.extend(
            [
                "",
                "Recruiter-only hidden information:",
                hidden,
            ]
        )
    return "\n".join(parts).strip()


def _normalize_runtime_section_weights(
    section_weights: Optional[Dict[str, float]],
    active_sections: List[str],
) -> Dict[str, float]:
    raw_weights = (
        {str(k): float(v) for k, v in section_weights.items() if v is not None and float(v) > 0}
        if section_weights is not None
        else dict(build_prompts.DEFAULT_SECTION_WEIGHTS)
    )
    filtered: Dict[str, float] = {}
    for section in active_sections:
        configured_weight = raw_weights.get(section)
        if configured_weight is None:
            filtered[section] = 1.0
            continue
        if configured_weight > 0:
            filtered[section] = configured_weight
    if not filtered:
        filtered = {section: 1.0 for section in active_sections}
    total = sum(filtered.values())
    if total <= 0:
        raise ValueError("section_weights total must be > 0")
    return {section: round(weight / total, 4) for section, weight in filtered.items()}


def _normalize_measurable(measurable: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(measurable, dict):
        return None
    field = str(measurable.get("field") or measurable.get("path") or "").strip()
    operator = str(measurable.get("operator") or measurable.get("op") or "").strip()
    value = measurable.get("value")
    if not field or not operator or value in (None, ""):
        return None
    spec = SUPPORTED_MEASURABLE_FIELDS.get(field)
    if spec is None or operator not in spec["operators"]:
        return None
    if spec["value_type"] == "number":
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
    elif spec["value_type"] == "boolean":
        if isinstance(value, bool):
            pass
        elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            value = value.strip().lower() == "true"
        else:
            return None
    elif spec["value_type"] == "string":
        allowed_values = spec.get("allowed_values", set())
        if isinstance(value, (list, tuple, set)):
            normalized_values: List[str] = []
            for item in value:
                normalized_item = str(item).strip().lower()
                if normalized_item not in allowed_values or normalized_item in normalized_values:
                    continue
                normalized_values.append(normalized_item)
            if not normalized_values:
                return None
            value = normalized_values if len(normalized_values) > 1 else normalized_values[0]
        else:
            value = str(value).strip().lower()
            if value not in allowed_values:
                return None
    return {
        "field": field,
        "operator": operator,
        "value": value,
    }


def _merge_equivalent_measurable_criteria(criteria: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []

    for criterion in criteria:
        measurable = criterion.get("measurable")
        if not isinstance(measurable, dict):
            merged.append(criterion)
            continue

        merged_into_existing = False
        for existing in merged:
            existing_measurable = existing.get("measurable")
            if not isinstance(existing_measurable, dict):
                continue
            if existing["section"] != criterion["section"]:
                continue
            if existing["requirementText"] != criterion["requirementText"]:
                continue
            if existing["type"] != criterion["type"]:
                continue
            if existing_measurable.get("field") != measurable.get("field"):
                continue
            if existing_measurable.get("operator") != measurable.get("operator"):
                continue

            existing_value = existing_measurable.get("value")
            incoming_value = measurable.get("value")
            existing_values = (
                list(existing_value)
                if isinstance(existing_value, list)
                else [existing_value]
            )
            incoming_values = (
                list(incoming_value)
                if isinstance(incoming_value, list)
                else [incoming_value]
            )
            merged_values: List[Any] = []
            for value in existing_values + incoming_values:
                if value not in merged_values:
                    merged_values.append(value)
            if all(isinstance(value, str) for value in merged_values):
                merged_values.sort()
            existing_measurable["value"] = merged_values if len(merged_values) > 1 else merged_values[0]
            merged_into_existing = True
            break

        if not merged_into_existing:
            merged.append(criterion)

    return merged


def _format_threshold_number(value: Any) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric).rstrip("0").rstrip(".")


def _source_has_explicit_year_threshold(source_text: str, value: Any) -> bool:
    if not source_text:
        return False
    number = re.escape(_format_threshold_number(value))
    decimal_suffix = r"(?:\.0)?"
    numeric_pattern = rf"{number}{decimal_suffix}"
    year_terms = r"(?:years?|yrs?|yoe|năm)"
    patterns = [
        rf"\b{numeric_pattern}\s*\+?\s*{year_terms}\b",
        rf"\b{year_terms}\s*(?:of\s+)?(?:experience\s*)?[:>=+\-\s]{{0,12}}{numeric_pattern}\b",
        rf"\b(?:at\s+least|minimum|min\.?|from|over|more\s+than|trên|hơn|tối\s+thiểu|ít\s+nhất)\s+{numeric_pattern}\s*{year_terms}\b",
    ]
    return any(re.search(pattern, source_text, re.IGNORECASE) for pattern in patterns)


def _source_has_boolean_requirement(source_text: str, field: str, expected_value: bool) -> bool:
    if not source_text:
        return False
    if field == "graduation_status":
        if expected_value == "graduated":
            pattern = (
                r"\b(bachelor'?s degree|master'?s degree|phd|doctorate|graduated|graduate|degree required)\b"
                r"|đã tốt nghiệp|tốt nghiệp|bằng cử nhân|bằng đại học|cử nhân|thạc sĩ|tiến sĩ"
            )
        elif expected_value == "final_year":
            pattern = (
                r"\b(final-year|final year|last-year student|expected graduation)\b"
                r"|năm cuối|sinh viên năm cuối|dự kiến tốt nghiệp|sắp tốt nghiệp"
            )
        elif expected_value == "studying":
            pattern = (
                r"\b(currently studying|still studying|student)\b"
                r"|đang học|sinh viên"
            )
        elif expected_value == "unknown":
            return False
        else:
            return False
        return re.search(pattern, source_text, re.IGNORECASE) is not None
    elif field == "ever_studied_abroad":
        positive_pattern = r"\b(stud(?:y|ied) abroad|overseas education|international education)\b|du học|học ở nước ngoài"
        negative_pattern = r"\b(no study abroad|studied abroad not required)\b|không yêu cầu du học"
    else:
        return False

    pattern = positive_pattern if expected_value else negative_pattern
    return re.search(pattern, source_text, re.IGNORECASE) is not None


def _measurable_supported_by_source(measurable: Dict[str, Any], source_text: Optional[str]) -> bool:
    if source_text is None:
        return True

    field = str(measurable.get("field") or "")
    value = measurable.get("value")
    if field == "experience_years":
        return _source_has_explicit_year_threshold(source_text, value)
    if field == "graduation_status" and isinstance(value, str):
        return _source_has_boolean_requirement(source_text, field, value)
    if field == "ever_studied_abroad" and isinstance(value, bool):
        return _source_has_boolean_requirement(source_text, field, value)
    return False


def _looks_like_year_threshold(requirement_text: str) -> bool:
    return re.search(r"\b\d+(?:\.\d+)?\s*\+?\s*(?:years?|yrs?|yoe|năm)\b", requirement_text, re.IGNORECASE) is not None


def _normalize_rubric(
    rubric: Dict[str, Any],
    section_weights: Optional[Dict[str, float]] = None,
    source_text: Optional[str] = None,
) -> Dict[str, Any]:
    raw_criteria = rubric.get("criteria", [])
    normalized_criteria: List[Dict[str, Any]] = []

    for idx, criterion in enumerate(raw_criteria):
        if not isinstance(criterion, dict):
            continue
        section = str(criterion.get("section") or "").strip().lower()
        requirement_text = str(criterion.get("requirementText") or "").strip()
        criterion_type = str(criterion.get("type") or "semantic").strip().lower()
        if section not in SUPPORTED_SCORING_SECTIONS or not requirement_text:
            continue
        if criterion_type not in SUPPORTED_CRITERION_TYPES:
            criterion_type = "semantic"
        key = str(criterion.get("key") or f"{section}.{idx + 1}").strip() or f"{section}.{idx + 1}"
        measurable = _normalize_measurable(criterion.get("measurable"))
        if measurable is not None and not _measurable_supported_by_source(measurable, source_text):
            continue
        if measurable is None and source_text is not None and _looks_like_year_threshold(requirement_text):
            continue
        if criterion_type != "upper_bound" and measurable is None:
            criterion_type = "semantic"
        normalized_criteria.append(
            {
                "key": key,
                "section": section,
                "requirementText": requirement_text,
                "type": criterion_type,
                "measurable": measurable,
            }
        )

    normalized_criteria = _merge_equivalent_measurable_criteria(normalized_criteria)

    if not normalized_criteria:
        return {"criteria": [], "sectionWeights": {}}

    active_sections = list(dict.fromkeys(criterion["section"] for criterion in normalized_criteria))
    normalized_section_weights = _normalize_runtime_section_weights(section_weights, active_sections)
    section_counts = Counter(criterion["section"] for criterion in normalized_criteria)

    for criterion in normalized_criteria:
        section_weight = normalized_section_weights[criterion["section"]]
        criterion["weight"] = round(section_weight / section_counts[criterion["section"]], 4)

    return {
        "criteria": normalized_criteria,
        "sectionWeights": normalized_section_weights,
    }


def _extract_candidate_field(candidate: Dict[str, Any], field: str) -> Any:
    return candidate.get(field)


def _compare_measurable(candidate_value: Any, operator: str, expected_value: Any) -> bool:
    if candidate_value in (None, ""):
        return False

    if operator == "contains":
        return str(expected_value).lower() in str(candidate_value).lower()

    if operator in NUMERIC_COMPARISON_OPERATORS:
        try:
            left = float(candidate_value)
            right = float(expected_value)
        except (TypeError, ValueError):
            return False
        if operator == ">=":
            return left >= right
        if operator == ">":
            return left > right
        if operator == "<=":
            return left <= right
        if operator == "<":
            return left < right
        return left == right

    if operator in EQUALITY_OPERATORS:
        if isinstance(expected_value, (list, tuple, set)):
            normalized_expected = {str(value).strip().lower() for value in expected_value}
            return str(candidate_value).strip().lower() in normalized_expected

        left_text = str(candidate_value).strip().lower()
        right_text = str(expected_value).strip().lower()
        return left_text == right_text

    return False


def _score_measurable_criterion(
    candidate: Dict[str, Any],
    criterion: Dict[str, Any],
) -> tuple[float, str, Dict[str, Any]]:
    measurable = criterion.get("measurable") or {}
    field = measurable.get("field", "")
    operator = measurable.get("operator", "")
    expected_value = measurable.get("value")
    actual_value = _extract_candidate_field(candidate, field)
    matched = _compare_measurable(actual_value, operator, expected_value)
    score = 100.0 if matched else 0.0
    if isinstance(expected_value, bool):
        expected_label = "true" if expected_value else "false"
        actual_label = "true" if actual_value else "false"
        if _ui_language() == "vi":
            if matched:
                evidence = f"Đáp ứng điều kiện logic cho {field}: kỳ vọng {expected_label}, ứng viên là {actual_label}."
            else:
                evidence = f"Chưa đáp ứng điều kiện logic cho {field}: kỳ vọng {expected_label}, ứng viên là {actual_label}."
        else:
            if matched:
                evidence = f"Matched boolean requirement for {field}: expected {expected_label}, candidate is {actual_label}."
            else:
                evidence = f"Did not match boolean requirement for {field}: expected {expected_label}, candidate is {actual_label}."
    else:
        if _ui_language() == "vi":
            if matched:
                evidence = f"Đáp ứng yêu cầu cho {field}: ứng viên có {actual_value} và điều kiện là {operator} {expected_value}."
            else:
                evidence = f"Chưa đáp ứng yêu cầu cho {field}: ứng viên có {actual_value} và điều kiện là {operator} {expected_value}."
        else:
            if matched:
                evidence = f"Matched requirement for {field}: candidate has {actual_value} and requirement is {operator} {expected_value}."
            else:
                evidence = f"Did not match requirement for {field}: candidate has {actual_value} and requirement is {operator} {expected_value}."
    return score, evidence, {
        "field": field,
        "operator": operator,
        "expectedValue": expected_value,
        "actualValue": actual_value,
        "matched": matched,
    }


def _operation_event_prefix(operation_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", operation_name.strip().lower()).strip("_")


def _parse_semantic_scores(raw_text: str) -> Dict[str, Dict[str, Any]]:
    scores = _parse_llm_scores(raw_text)
    by_candidate: Dict[str, Dict[str, Any]] = {}
    for entry in scores:
        candidate_id = str(entry.get("candidateId") or "").strip()
        if not candidate_id:
            continue
        criteria_rows = entry.get("criteria")
        if not isinstance(criteria_rows, list):
            criteria_rows = entry.get("componentScores", [])
        mapped_criteria: Dict[str, Dict[str, Any]] = {}
        if isinstance(criteria_rows, list):
            for row in criteria_rows:
                if not isinstance(row, dict):
                    continue
                criterion_key = str(row.get("criterionKey") or "").strip()
                if not criterion_key:
                    continue
                mapped_criteria[criterion_key] = {
                    "score": _normalize_llm_score(row.get("score", 0)),
                    "evidenceSummary": str(row.get("evidenceSummary") or "").strip(),
                }
        by_candidate[candidate_id] = {
            "rationale": str(entry.get("rationale") or "").strip(),
            "criteria": mapped_criteria,
        }
    return by_candidate


def _safe_parse_semantic_scores(raw_text: str) -> Dict[str, Dict[str, Any]]:
    try:
        return _parse_semantic_scores(raw_text)
    except Exception:
        logger.warning("Semantic scoring response could not be parsed; continuing without semantic scores.")
        return {}


def _generate_json_with_retries(
    *,
    llm: LLMProvider,
    prompt: str,
    parser,
    operation_name: str,
    debug_logger: Optional[ScoringDebugLogger] = None,
) -> Dict[str, Any]:
    json_fix_suffix = (
        "\n\nIMPORTANT: Return one valid JSON object only. "
        "Do not include markdown fences, commentary, or trailing commas."
    )
    attempts: List[tuple[LLMProvider, str]] = [
        (llm, prompt),
        (llm, f"{prompt}{json_fix_suffix}"),
    ]

    last_error: Optional[Exception] = None
    event_prefix = _operation_event_prefix(operation_name)
    for idx, (attempt_llm, attempt_prompt) in enumerate(attempts, start=1):
        if debug_logger is not None:
            debug_logger.record_event(
                f"{event_prefix}_attempt",
                {
                    "attempt": idx,
                    "provider": getattr(attempt_llm, "provider", "unknown"),
                    "model": getattr(attempt_llm, "model_name", "unknown"),
                    "prompt": preview_text(attempt_prompt),
                },
            )
        try:
            response = attempt_llm.generate(attempt_prompt)
        except Exception as exc:
            if debug_logger is not None:
                debug_logger.record_error(
                    f"{event_prefix}_generation_failed",
                    exc,
                    {
                        "attempt": idx,
                        "provider": getattr(attempt_llm, "provider", "unknown"),
                        "model": getattr(attempt_llm, "model_name", "unknown"),
                    },
                )
            raise
        try:
            parsed = parser(response.text)
            if debug_logger is not None:
                debug_logger.record_event(
                    f"{event_prefix}_completed",
                    {
                        "attempt": idx,
                        "provider": getattr(response, "provider", getattr(attempt_llm, "provider", "unknown")),
                        "model": getattr(response, "model", getattr(attempt_llm, "model_name", "unknown")),
                        "response": preview_text(getattr(response, "text", "")),
                    },
                )
            return parsed
        except Exception as exc:
            last_error = exc
            if debug_logger is not None:
                debug_logger.record_error(
                    f"{event_prefix}_parse_failed",
                    exc,
                    {
                        "attempt": idx,
                        "provider": getattr(attempt_llm, "provider", "unknown"),
                        "model": getattr(attempt_llm, "model_name", "unknown"),
                        "response": preview_text(getattr(response, "text", "")),
                    },
                )
            logger.warning(
                "%s JSON parse failed on attempt %s using model %s: %s",
                operation_name,
                idx,
                getattr(attempt_llm, "model_name", "unknown"),
                exc,
            )
            if idx == len(attempts):
                break
    raise ValueError(f"{operation_name} failed after retries: {last_error}") from last_error


def _generate_semantic_scores_with_retries(
    *,
    llm: LLMProvider,
    prompt: str,
    debug_logger: Optional[ScoringDebugLogger] = None,
) -> Dict[str, Dict[str, Any]]:
    if debug_logger is not None:
        debug_logger.record_event(
            "semantic_scoring_started",
            {
                "provider": getattr(llm, "provider", "unknown"),
                "model": getattr(llm, "model_name", "unknown"),
                "prompt": preview_text(prompt),
            },
        )
    try:
        return _generate_json_with_retries(
            llm=llm,
            prompt=prompt,
            parser=_parse_semantic_scores,
            operation_name="semantic scoring",
            debug_logger=debug_logger,
        )
    except Exception as exc:
        if _is_provider_limit_error(exc):
            raise
        logger.warning("Semantic scoring response could not be parsed after retries; continuing without semantic scores.")
        return {}


def _format_component_label(component: Dict[str, Any]) -> str:
    return str(component.get("requirementText") or component.get("criterionKey") or "criterion").strip()


def _format_rationale_item(component: Dict[str, Any]) -> str:
    label = _format_component_label(component)
    evidence = str(component.get("evidenceSummary") or "").strip().rstrip(".")
    return f"{label} - {evidence}" if evidence else label


def _build_rationale_summary(total_score: float, component_scores: List[Dict[str, Any]]) -> str:
    vi = _ui_language() == "vi"
    parts = [f"Điểm tổng {round(total_score, 2)}/100." if vi else f"Overall score {round(total_score, 2)}/100."]
    scored_components = sorted(
        component_scores,
        key=lambda component: float(component.get("score") or 0),
        reverse=True,
    )
    strong_matches = [
        component
        for component in scored_components
        if float(component.get("score") or 0) >= 70 and str(component.get("evidenceSummary") or "").strip()
    ]
    gaps = [
        component
        for component in reversed(scored_components)
        if float(component.get("score") or 0) < 40 and str(component.get("evidenceSummary") or "").strip()
    ]
    partial_matches = [
        component
        for component in scored_components
        if 40 <= float(component.get("score") or 0) < 70 and str(component.get("evidenceSummary") or "").strip()
    ]

    if strong_matches:
        prefix = "Điểm mạnh: " if vi else "Strong matches: "
        parts.append(prefix + "; ".join(_format_rationale_item(component) for component in strong_matches[:2]) + ".")
    if gaps:
        prefix = "Khoảng trống: " if vi else "Gaps: "
        parts.append(prefix + "; ".join(_format_rationale_item(component) for component in gaps[:2]) + ".")
    elif partial_matches:
        prefix = "Phù hợp một phần: " if vi else "Partial matches: "
        parts.append(prefix + "; ".join(_format_rationale_item(component) for component in partial_matches[:2]) + ".")

    return " ".join(parts)[:1000]


def _build_candidate_score(
    *,
    candidate: Dict[str, Any],
    rubric: Dict[str, Any],
    semantic_result: Dict[str, Any],
    score_threshold: Decimal,
    debug_logger: Optional[ScoringDebugLogger] = None,
) -> Dict[str, Any]:
    semantic_criteria = semantic_result.get("criteria", {}) if isinstance(semantic_result, dict) else {}
    component_scores: List[Dict[str, Any]] = []
    debug_components: List[Dict[str, Any]] = []

    for criterion in rubric.get("criteria", []):
        weight = float(criterion.get("weight", 0.0))
        if criterion.get("measurable"):
            score, evidence, measurable_detail = _score_measurable_criterion(candidate, criterion)
            evaluation_mode = "measurable"
        else:
            semantic_detail = semantic_criteria.get(criterion["key"], {})
            score = _normalize_llm_score(semantic_detail.get("score", 0))
            evidence = str(semantic_detail.get("evidenceSummary") or "").strip()
            evaluation_mode = "semantic"
            measurable_detail = None
        component_scores.append(
            {
                "criterionKey": criterion["key"],
                "criterionType": criterion["type"],
                "evaluationMode": evaluation_mode,
                "requirementText": criterion["requirementText"],
                "weight": round(weight, 4),
                "score": score,
                "weightedScore": round(weight * score, 2),
                "evidenceSummary": evidence,
            }
        )
        debug_component = {
            "criterionKey": criterion["key"],
            "criterionType": criterion["type"],
            "evaluationMode": evaluation_mode,
            "requirementText": criterion["requirementText"],
            "weight": round(weight, 4),
            "score": score,
            "weightedScore": round(weight * score, 2),
            "evidenceSummary": evidence,
        }
        if measurable_detail is not None:
            debug_component["measurable"] = measurable_detail
        debug_components.append(debug_component)

    total_score = round(sum(component["weightedScore"] for component in component_scores), 2)
    rationale = _build_rationale_summary(total_score, component_scores)
    normalized_total_score = _clamp_score(total_score)
    passed_threshold = normalized_total_score >= float(score_threshold)

    if debug_logger is not None:
        debug_logger.record_event(
            "candidate_scored",
            {
                "candidateId": str(candidate.get("id") or candidate.get("candidateId") or ""),
                "candidateDisplayName": str(
                    candidate.get("display_name") or candidate.get("full_name") or candidate.get("resume_file_name") or ""
                ).strip(),
                "totalScore": normalized_total_score,
                "passedThreshold": passed_threshold,
                "componentScores": debug_components,
            },
        )

    return {
        "candidateId": str(candidate.get("id") or candidate.get("candidateId") or ""),
        "candidateName": str(candidate.get("full_name") or "").strip(),
        "resumeFileName": str(candidate.get("resume_file_name") or "").strip(),
        "candidateDisplayName": str(
            candidate.get("display_name") or candidate.get("full_name") or candidate.get("resume_file_name") or ""
        ).strip(),
        "totalScore": normalized_total_score,
        "passedThreshold": passed_threshold,
        "rationale": rationale,
        "componentScores": component_scores,
    }


def _coerce_passed_threshold(score_data: Dict[str, Any], score_threshold: Decimal) -> Dict[str, Any]:
    total_score = _clamp_score(score_data.get("totalScore", 0))
    normalized = dict(score_data)
    normalized["totalScore"] = total_score
    normalized["passedThreshold"] = total_score >= float(score_threshold)
    return normalized


def _extract_locked_rubric(
    *,
    llm: LLMProvider,
    job_description_text: str,
    section_weights: Optional[Dict[str, float]],
    debug_logger: Optional[ScoringDebugLogger] = None,
) -> Optional[Dict[str, Any]]:
    try:
        rubric_payload = _generate_json_with_retries(
            llm=llm,
            prompt=build_prompts.build_jd_rubric_extraction_prompt(
                job_description_text=job_description_text,
                section_weights=section_weights,
            ),
            parser=_parse_rubric_response,
            operation_name="rubric extraction",
            debug_logger=debug_logger,
        )
        rubric = _normalize_rubric(rubric_payload, section_weights, source_text=job_description_text)
        if debug_logger is not None:
            debug_logger.record_event(
                "rubric_normalized",
                {
                    "criteriaCount": len(rubric.get("criteria", [])),
                    "sectionWeights": rubric.get("sectionWeights", {}),
                    "criteria": rubric.get("criteria", []),
                },
            )
        return rubric if rubric.get("criteria") else None
    except Exception as exc:
        if _is_provider_limit_error(exc):
            raise
        if debug_logger is not None:
            debug_logger.record_error("rubric_extraction_failed", exc)
        return None


def _save_batch_scores(
    *,
    db: Session,
    match_run_id: uuid.UUID,
    batch_scores: List[Dict[str, Any]],
    valid_candidate_ids: set[uuid.UUID],
    score_threshold: Decimal,
    starting_index: int,
    debug_logger: Optional[ScoringDebugLogger] = None,
) -> tuple[int, int]:
    global_idx = starting_index
    passed_candidates_count = 0
    persisted_candidates: List[Dict[str, Any]] = []

    for raw_score in batch_scores:
        score_data = _coerce_passed_threshold(raw_score, score_threshold)
        candidate_id_text = score_data.get("candidateId")
        if not candidate_id_text:
            continue
        try:
            candidate_id = uuid.UUID(str(candidate_id_text))
        except ValueError:
            continue
        if candidate_id not in valid_candidate_ids:
            continue

        if score_data["passedThreshold"]:
            passed_candidates_count += 1

        db.add(
            MatchResult(
                match_run_id=match_run_id,
                candidate_profile_id=candidate_id,
                score_list_index=global_idx,
                total_score=Decimal(str(score_data["totalScore"])),
                passed_threshold=bool(score_data["passedThreshold"]),
                rationale_summary=str(score_data.get("rationale") or ""),
                component_scores=score_data.get("componentScores") or [],
            )
        )
        persisted_candidates.append(
            {
                "candidateId": str(candidate_id),
                "scoreListIndex": global_idx,
                "totalScore": score_data["totalScore"],
                "passedThreshold": bool(score_data["passedThreshold"]),
            }
        )
        global_idx += 1

    if debug_logger is not None:
        debug_logger.record_event(
            "batch_persist_completed",
            {
                "matchRunId": str(match_run_id),
                "startingIndex": starting_index,
                "nextIndex": global_idx,
                "persistedCount": len(persisted_candidates),
                "persistedCandidates": persisted_candidates,
            },
        )

    return global_idx, passed_candidates_count


def score_candidates(
    *,
    db: Session,
    job_description_id: uuid.UUID,
    initiated_by_user_id: uuid.UUID,
    score_threshold: Decimal = Decimal("50.0"),
    candidate_profile_ids: Optional[List[uuid.UUID]] = None,
    section_weights: Optional[Dict[str, float]] = None,
    batch_size: int = 10,
) -> Dict[str, Any]:
    """Score candidate profiles against a job description using rubric-locked scoring."""
    jd: Optional[JobDescription] = db.get(JobDescription, job_description_id)
    if jd is None:
        raise ValueError(f"Job description {job_description_id} not found")

    query = (
        db.query(CandidateProfile)
        .options(joinedload(CandidateProfile.resume_document))
        .join(ResumeDocument, ResumeDocument.id == CandidateProfile.resume_document_id)
        .filter(ResumeDocument.job_id == jd.job_id)
    )
    if candidate_profile_ids:
        query = query.filter(CandidateProfile.id.in_(candidate_profile_ids))
    profiles: List[CandidateProfile] = query.all()
    if not profiles:
        raise ValueError("No candidate profiles found for the given parameters")

    candidate_dicts = [_profile_to_candidate_dict(profile) for profile in profiles]
    valid_candidate_ids = {profile.id for profile in profiles}
    scoring_jd_text = _build_scoring_job_description_text(
        public_job_description=jd.jd_text,
        hidden_text=getattr(jd, "hidden_text", ""),
    )

    match_run = MatchRun(
        job_description_id=jd.id,
        score_threshold=score_threshold,
        initiated_by_user_id=initiated_by_user_id,
        run_status=MatchRunStatus.RUNNING.value,
    )
    db.add(match_run)
    db.commit()
    db.refresh(match_run)
    debug_logger = ScoringDebugLogger(str(match_run.id))
    debug_logger.record_event(
        "run_started",
        {
            "matchRunId": str(match_run.id),
            "jobDescriptionId": str(job_description_id),
            "initiatedByUserId": str(initiated_by_user_id),
            "scoreThreshold": score_threshold,
            "requestedCandidateIds": candidate_profile_ids or [],
            "candidateCount": len(candidate_dicts),
            "batchSize": batch_size,
        },
    )
    debug_logger.record_event(
        "job_description_prepared",
        {
            "publicJobDescription": preview_text(jd.jd_text),
            "hiddenText": preview_text(getattr(jd, "hidden_text", "")),
            "combinedTextLength": len(scoring_jd_text),
            "sectionWeights": section_weights or {},
        },
    )

    all_scores: List[Dict[str, Any]] = []
    requested_batch_size = max(1, min(batch_size, 50))
    batches_run = 0
    passed_candidates_count = 0

    try:
        llm = _scoring_llm_provider()
        rubric = _extract_locked_rubric(
            llm=llm,
            job_description_text=scoring_jd_text,
            section_weights=section_weights,
            debug_logger=debug_logger,
        )
        semantic_criteria = (
            [criterion for criterion in rubric["criteria"] if criterion.get("measurable") is None]
            if rubric is not None
            else []
        )
        scoring_window = BudgetWindow(
            context_window=settings.SCORING_CONTEXT_WINDOW_TOKENS,
            output_budget=settings.SCORING_OUTPUT_TOKEN_BUDGET,
            reserve=settings.SCORING_CONTEXT_RESERVE_TOKENS,
        )
        static_prompt_tokens = estimate_tokens(scoring_jd_text) + estimate_json_tokens(
            {
                "rubric": rubric or {},
                "sectionWeights": section_weights or {},
            }
        )
        batch_plan = build_scoring_batch_plan(
            candidates=candidate_dicts,
            semantic_criteria=semantic_criteria,
            static_prompt_tokens=static_prompt_tokens,
            window=scoring_window,
            max_candidates_per_batch=min(
                settings.SCORING_MAX_CANDIDATES_PER_BATCH,
                requested_batch_size,
            ),
            max_criteria_per_call=settings.SCORING_MAX_SEMANTIC_CRITERIA_PER_CALL,
        )
        debug_logger.record_event(
            "adaptive_batch_plan_created",
            {
                "matchRunId": str(match_run.id),
                "candidateCount": batch_plan.total_candidates,
                "semanticCriteriaCount": batch_plan.total_criteria,
                "candidateBatchCount": len(batch_plan.candidate_batches),
                "criterionBatchCount": len(batch_plan.criterion_batches),
                "plannerSettings": batch_plan.planner_settings,
            },
        )
        global_idx = 0
        for candidate_batch_index, candidate_batch in enumerate(batch_plan.candidate_batches):
            batch = candidate_batch.candidates
            batch_started_at = time.perf_counter()
            debug_logger.record_event(
                "batch_started",
                {
                    "batchIndex": candidate_batch_index,
                    "candidateIds": [str(candidate.get("id") or "") for candidate in batch],
                    "candidateDisplayNames": [str(candidate.get("display_name") or "") for candidate in batch],
                    "usesLockedRubric": rubric is not None,
                    "estimatedInputTokens": candidate_batch.estimated_input_tokens,
                    "estimatedOutputTokens": candidate_batch.estimated_output_tokens,
                },
            )
            debug_logger.record_event(
                "candidate_batch_started",
                {
                    "batchIndex": candidate_batch_index,
                    "candidateCount": len(batch),
                    "estimatedInputTokens": candidate_batch.estimated_input_tokens,
                    "estimatedOutputTokens": candidate_batch.estimated_output_tokens,
                    "usesLockedRubric": rubric is not None,
                },
            )

            if rubric is None:
                fallback_started_at = time.perf_counter()
                parsed_scores = _generate_json_with_retries(
                    llm=llm,
                    prompt=build_prompts.build_batch_scoring_prompt(
                        job_description_text=scoring_jd_text,
                        candidates=batch,
                        section_weights=section_weights,
                    ),
                    parser=lambda text: {"scores": _parse_llm_scores(text)},
                    operation_name="fallback batch scoring",
                    debug_logger=debug_logger,
                )
                debug_logger.record_event(
                    "fallback_batch_scoring_completed",
                    {
                        "batchIndex": candidate_batch_index,
                        "durationMs": _duration_ms(fallback_started_at),
                        "scoreCount": len(parsed_scores.get("scores") or []),
                    },
                )
                candidate_by_id = {str(candidate["id"]): candidate for candidate in batch}
                batch_scores = [
                    _attach_candidate_metadata(
                        _coerce_passed_threshold(score, score_threshold),
                        candidate_by_id.get(str(score.get("candidateId")), {}),
                    )
                    for score in parsed_scores["scores"]
                ]
            else:
                semantic_by_candidate: Dict[str, Dict[str, Any]] = {}
                if semantic_criteria:
                    for criterion_batch_index, criterion_batch in enumerate(batch_plan.criterion_batches):
                        semantic_started_at = time.perf_counter()
                        debug_logger.record_event(
                            "semantic_criteria_batch_started",
                            {
                                "candidateBatchIndex": candidate_batch_index,
                                "criterionBatchIndex": criterion_batch_index,
                                "candidateCount": len(batch),
                                "criterionKeys": [str(criterion.get("key") or "") for criterion in criterion_batch],
                                "criterionCount": len(criterion_batch),
                            },
                        )
                        semantic_update = _generate_semantic_scores_with_retries(
                            llm=llm,
                            prompt=build_prompts.build_locked_rubric_semantic_scoring_prompt(
                                candidates=batch,
                                rubric={"criteria": criterion_batch},
                            ),
                            debug_logger=debug_logger,
                        )
                        _merge_semantic_scores(semantic_by_candidate, semantic_update)
                        debug_logger.record_event(
                            "semantic_criteria_batch_completed",
                            {
                                "candidateBatchIndex": candidate_batch_index,
                                "criterionBatchIndex": criterion_batch_index,
                                "durationMs": _duration_ms(semantic_started_at),
                                "candidateResultCount": len(semantic_update),
                            },
                        )

                batch_scores = []
                for candidate in batch:
                    semantic_result = semantic_by_candidate.get(str(candidate["id"]), {})
                    batch_scores.append(
                        _build_candidate_score(
                            candidate=candidate,
                            rubric=rubric,
                            semantic_result=semantic_result,
                            score_threshold=score_threshold,
                            debug_logger=debug_logger,
                        )
                    )

            all_scores.extend(batch_scores)
            batches_run += 1
            debug_logger.record_event(
                "candidate_batch_scored",
                {
                    "batchIndex": candidate_batch_index,
                    "durationMs": _duration_ms(batch_started_at),
                    "scoreCount": len(batch_scores),
                },
            )

            persist_started_at = time.perf_counter()
            global_idx, passed_delta = _save_batch_scores(
                db=db,
                match_run_id=match_run.id,
                batch_scores=batch_scores,
                valid_candidate_ids=valid_candidate_ids,
                score_threshold=score_threshold,
                starting_index=global_idx,
                debug_logger=debug_logger,
            )
            passed_candidates_count += passed_delta
            db.commit()
            debug_logger.record_event(
                "candidate_batch_persist_completed",
                {
                    "batchIndex": candidate_batch_index,
                    "durationMs": _duration_ms(persist_started_at),
                    "passedCandidates": passed_delta,
                },
            )

        match_run.run_status = MatchRunStatus.COMPLETED.value
        match_run.completed_at = datetime.now(timezone.utc)
        db.commit()
        completion_payload = {
            "matchRunId": str(match_run.id),
            "totalCandidates": len(candidate_dicts),
            "totalPassedCandidates": passed_candidates_count,
            "batches": batches_run,
            "candidateBatchCount": len(batch_plan.candidate_batches),
            "criterionBatchCount": len(batch_plan.criterion_batches),
        }
        debug_logger.record_event(
            "run_completed",
            completion_payload,
        )
        debug_logger.record_event(
            "scoring_run_completed",
            completion_payload,
        )

    except Exception as exc:
        db.rollback()
        _mark_match_run_failed(db, match_run.id)
        debug_logger.record_error(
            "run_failed",
            exc,
            {
                "matchRunId": str(match_run.id),
                "jobDescriptionId": str(job_description_id),
                "batchesCompleted": batches_run,
            },
        )
        if _is_provider_limit_error(exc):
            logger.error(
                "Candidate scoring failed because the configured LLM provider hit a quota or rate limit. "
                "match_run_id=%s job_description_id=%s error=%s",
                match_run.id,
                job_description_id,
                exc,
            )
            raise ScoringProviderLimitError(
                "Scoring is temporarily unavailable because the configured LLM quota has been exhausted. Please retry later."
            ) from exc
        logger.exception(
            "Candidate scoring failed. match_run_id=%s job_description_id=%s",
            match_run.id,
            job_description_id,
        )
        raise ValueError(f"Matching failed: {exc}") from exc

    return {
        "match_run_id": str(match_run.id),
        "job_description_id": str(job_description_id),
        "total_candidates": len(candidate_dicts),
        "total_passed_candidates": passed_candidates_count,
        "batches": batches_run,
        "scores": all_scores,
    }
