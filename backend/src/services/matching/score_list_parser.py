from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from src.models.matching import MatchResult


@dataclass
class ParsedScoreItem:
    candidate_id: uuid.UUID
    total_score: float
    passed_threshold: bool
    rationale: str
    component_scores: list[dict[str, Any]]
    score_list_index: int


def _normalize_component_scores(component_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in component_scores:
        weight = float(raw.get("weight", 0))
        score = float(raw.get("score", 0))
        weighted_score = raw.get("weightedScore")
        if weighted_score is None:
            weighted_score = score * max(0.0, min(1.0, weight))
        normalized.append(
            {
                "criterionKey": str(raw.get("criterionKey", "unknown")),
                "weight": max(0.0, min(1.0, weight)),
                "score": max(0.0, min(100.0, score)),
                "weightedScore": max(0.0, min(100.0, float(weighted_score))),
                "evidenceSummary": raw.get("evidenceSummary"),
            }
        )
    return normalized


def parse_score_items(raw_items: list[dict[str, Any]], threshold: float) -> list[ParsedScoreItem]:
    parsed: list[ParsedScoreItem] = []
    for index, raw_item in enumerate(raw_items):
        candidate_id = uuid.UUID(str(raw_item["candidateId"]))
        component_scores = _normalize_component_scores(list(raw_item.get("componentScores") or []))

        total_score = raw_item.get("totalScore")
        if total_score is None:
            total_score = sum(item["weightedScore"] for item in component_scores)

        total_score = max(0.0, min(100.0, float(total_score)))
        parsed.append(
            ParsedScoreItem(
                candidate_id=candidate_id,
                total_score=total_score,
                passed_threshold=bool(raw_item.get("passedThreshold", total_score >= threshold)),
                rationale=str(raw_item.get("rationale", "No rationale supplied.")),
                component_scores=component_scores,
                score_list_index=int(raw_item.get("scoreListIndex", index)),
            )
        )
    return parsed


def persist_match_results(
    session: Session,
    *,
    match_run_id: uuid.UUID,
    threshold: float,
    raw_items: list[dict[str, Any]],
) -> list[MatchResult]:
    parsed_items = parse_score_items(raw_items, threshold)
    persisted: list[MatchResult] = []
    for item in parsed_items:
        result = MatchResult(
            match_run_id=match_run_id,
            candidate_profile_id=item.candidate_id,
            score_list_index=item.score_list_index,
            total_score=item.total_score,
            passed_threshold=item.passed_threshold,
            rationale_summary=item.rationale,
            confidence_level=None,
            component_scores=item.component_scores,
        )
        session.add(result)
        persisted.append(result)
    session.flush()
    return persisted
