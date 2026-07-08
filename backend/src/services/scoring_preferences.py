from __future__ import annotations

from decimal import Decimal
from typing import Any

DEFAULT_SECTION_WEIGHTS = {
    "skills": 25.0,
    "experience": 25.0,
    "education": 20.0,
    "projects": 20.0,
    "summary": 10.0,
}


def derive_default_section_weights(
    *,
    raw_component_scores: list[dict[str, Any]] | None = None,
    rubric_payload: dict[str, Any] | None = None,
) -> dict[str, float]:
    section_counts: dict[str, int] = {}

    for component in raw_component_scores or []:
        section = str(component.get("section") or "").strip()
        if section:
            section_counts[section] = section_counts.get(section, 0) + 1

    if not section_counts and isinstance(rubric_payload, dict):
        for criterion in rubric_payload.get("criteria", []):
            if not isinstance(criterion, dict):
                continue
            section = str(criterion.get("section") or "").strip()
            if section:
                section_counts[section] = section_counts.get(section, 0) + 1

    total = sum(section_counts.values())
    if total <= 0:
        return {}

    return {
        section: round(count / total, 4)
        for section, count in section_counts.items()
    }


def derive_default_section_weights_percent(
    *,
    raw_component_scores: list[dict[str, Any]] | None = None,
    rubric_payload: dict[str, Any] | None = None,
) -> dict[str, float]:
    normalized = derive_default_section_weights(
        raw_component_scores=raw_component_scores,
        rubric_payload=rubric_payload,
    )
    return {section: round(weight * 100, 2) for section, weight in normalized.items()}


def _clamp_score(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return round(max(0.0, min(100.0, numeric)), 2)


def _positive_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric > 0 else None


def normalize_section_weights(section_weights: dict[str, float] | None) -> dict[str, float]:
    source = DEFAULT_SECTION_WEIGHTS if section_weights is None else section_weights
    cleaned: dict[str, float] = {}
    for key, raw_value in source.items():
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            cleaned[key] = numeric
    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError("Section weight total must be greater than zero")
    return {key: round(value / total, 6) for key, value in cleaned.items()}


def calculate_weighted_score(
    *,
    raw_component_scores: list[dict[str, Any]],
    section_weights: dict[str, float] | None,
    score_threshold: Decimal,
) -> dict[str, Any]:
    normalized_weights = (
        normalize_section_weights(section_weights)
        if section_weights is not None
        else None
    )
    component_scores: list[dict[str, Any]] = []
    total_score = 0.0
    criterion_weight_totals: dict[str, float] = {}
    criterion_weight_sum = 0.0
    section_counts: dict[str, int] = {}

    for component in raw_component_scores:
        section = str(component.get("section") or "").strip()
        if section:
            section_counts[section] = section_counts.get(section, 0) + 1
        raw_weight = _positive_float(component.get("weight"))
        if raw_weight is None:
            continue
        criterion_weight_sum += raw_weight
        if section:
            criterion_weight_totals[section] = criterion_weight_totals.get(section, 0.0) + raw_weight

    for component in raw_component_scores:
        section = str(component.get("section") or "").strip()
        score_percent = _clamp_score(component.get("scorePercent", component.get("score", 0)))
        raw_weight = _positive_float(component.get("weight"))
        if normalized_weights is None:
            if raw_weight is not None and criterion_weight_sum > 0:
                effective_weight = raw_weight / criterion_weight_sum
            else:
                fallback_weights = derive_default_section_weights(
                    raw_component_scores=raw_component_scores,
                )
                if not fallback_weights:
                    fallback_weights = normalize_section_weights(None)
                section_weight = fallback_weights.get(section, 0.0)
                section_count = section_counts.get(section, 0)
                effective_weight = section_weight / section_count if section_weight > 0 and section_count > 0 else 0.0
        else:
            section_weight = normalized_weights.get(section, 0.0)
            section_total = criterion_weight_totals.get(section, 0.0)
            if raw_weight is not None and section_total > 0:
                effective_weight = section_weight * (raw_weight / section_total)
            else:
                section_count = section_counts.get(section, 0)
                effective_weight = section_weight / section_count if section_weight > 0 and section_count > 0 else 0.0
        weighted_score = round(score_percent * effective_weight, 2)
        total_score += weighted_score
        enriched = dict(component)
        enriched["scorePercent"] = score_percent
        enriched["effectiveWeight"] = round(effective_weight, 6)
        enriched["weightedScore"] = weighted_score
        component_scores.append(enriched)

    total_score = round(total_score, 2)
    return {
        "componentScores": component_scores,
        "totalScore": total_score,
        "passedThreshold": total_score >= float(score_threshold),
        "normalizedSectionWeights": normalized_weights,
    }
