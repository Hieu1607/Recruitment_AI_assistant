from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.services.token_budget import BudgetWindow, estimate_json_tokens


_MAP_OUTPUT_PER_CANDIDATE_TOKENS = 80


class AnswerMode(str, Enum):
    DETAILED = "detailed"
    COMPACT_ID_NAME = "compact_id_name"
    NO_MATCH = "no_match"


@dataclass(frozen=True)
class ChatMapBatch:
    candidates: list[dict[str, Any]]
    estimated_input_tokens: int
    estimated_output_tokens: int


def compact_candidate_identity(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": candidate.get("id"),
        "full_name": candidate.get("full_name"),
    }


def compact_candidate_for_map(candidate: dict[str, Any]) -> dict[str, Any]:
    compact = compact_candidate_identity(candidate)
    for field in ("current_job_title", "experience_years", "summary_text", "skills_text"):
        if candidate.get(field) is not None:
            compact[field] = candidate.get(field)
    return compact


def compact_map_result(map_result: dict[str, Any]) -> dict[str, Any]:
    candidates = map_result.get("qualifiedCandidates") or []
    return {
        "batchQualifiedCount": int(map_result.get("batchQualifiedCount") or len(candidates)),
        "qualifiedCandidates": [
            {
                "id": item.get("id"),
                "name": item.get("name") or item.get("full_name"),
                "score": item.get("score", 0),
                "reason": item.get("reason") or "",
            }
            for item in candidates
            if isinstance(item, dict) and item.get("id")
        ],
    }


def limit_compact_candidates(candidates: list[dict[str, Any]], max_count: int) -> list[dict[str, Any]]:
    return [compact_candidate_identity(candidate) for candidate in candidates[: max(0, int(max_count))]]


def _candidate_cost(candidate: dict[str, Any]) -> int:
    return estimate_json_tokens(candidate)


def _map_output_tokens(candidate_count: int) -> int:
    return max(1, int(candidate_count)) * _MAP_OUTPUT_PER_CANDIDATE_TOKENS


def build_chat_map_batches(
    *,
    question: str,
    candidates: list[dict[str, Any]],
    job_context: dict[str, Any] | None,
    static_prompt_tokens: int,
    window: BudgetWindow,
    max_candidates_per_batch: int,
) -> list[ChatMapBatch]:
    del question, job_context
    batches: list[ChatMapBatch] = []
    max_candidates = max(1, int(max_candidates_per_batch))
    current: list[dict[str, Any]] = []
    current_tokens = int(static_prompt_tokens)

    def flush_current() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        batches.append(
            ChatMapBatch(
                candidates=current,
                estimated_input_tokens=current_tokens,
                estimated_output_tokens=_map_output_tokens(len(current)),
            )
        )
        current = []
        current_tokens = int(static_prompt_tokens)

    for raw_candidate in candidates:
        candidate = raw_candidate
        candidate_tokens = _candidate_cost(candidate)
        if int(static_prompt_tokens) + candidate_tokens > window.input_budget:
            candidate = compact_candidate_for_map(raw_candidate)
            candidate_tokens = _candidate_cost(candidate)

        projected_tokens = current_tokens + candidate_tokens
        projected_count = len(current) + 1
        projected_output = _map_output_tokens(projected_count)
        if current and (
            projected_tokens > window.input_budget
            or projected_output > window.output_budget
            or projected_count > max_candidates
        ):
            flush_current()
            projected_tokens = current_tokens + candidate_tokens

        current.append(candidate)
        current_tokens = projected_tokens

    flush_current()
    return batches


def choose_answer_mode(
    *,
    candidates: list[dict[str, Any]],
    detailed_threshold: int,
    compact_threshold: int,
    estimated_full_tokens: int,
    final_input_budget: int,
) -> AnswerMode:
    if not candidates:
        return AnswerMode.NO_MATCH
    if len(candidates) <= int(detailed_threshold) and estimated_full_tokens <= final_input_budget:
        return AnswerMode.DETAILED
    return AnswerMode.COMPACT_ID_NAME
