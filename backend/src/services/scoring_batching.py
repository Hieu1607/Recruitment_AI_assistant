from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.services.token_budget import BudgetWindow, estimate_json_tokens


_PER_CRITERION_OUTPUT_TOKENS = 40
_PER_CANDIDATE_OUTPUT_TOKENS = 80


@dataclass(frozen=True)
class CandidateBatch:
    candidates: list[dict[str, Any]]
    estimated_input_tokens: int
    estimated_output_tokens: int


@dataclass(frozen=True)
class ScoringBatchPlan:
    candidate_batches: list[CandidateBatch]
    criterion_batches: list[list[dict[str, Any]]]
    total_candidates: int
    total_criteria: int
    planner_settings: dict[str, Any]


def _candidate_cost(candidate: dict[str, Any]) -> int:
    return estimate_json_tokens(candidate)


def _estimate_output_tokens(candidate_count: int, criterion_count: int) -> int:
    effective_criteria = max(1, int(criterion_count))
    return candidate_count * _PER_CANDIDATE_OUTPUT_TOKENS + candidate_count * effective_criteria * _PER_CRITERION_OUTPUT_TOKENS


def _chunk_criteria(
    semantic_criteria: list[dict[str, Any]],
    max_criteria_per_call: int,
) -> list[list[dict[str, Any]]]:
    if not semantic_criteria:
        return []
    chunk_size = max(1, int(max_criteria_per_call))
    return [
        semantic_criteria[idx : idx + chunk_size]
        for idx in range(0, len(semantic_criteria), chunk_size)
    ]


def build_scoring_batch_plan(
    *,
    candidates: list[dict[str, Any]],
    semantic_criteria: list[dict[str, Any]],
    static_prompt_tokens: int,
    window: BudgetWindow,
    max_candidates_per_batch: int,
    max_criteria_per_call: int,
) -> ScoringBatchPlan:
    criterion_batches = _chunk_criteria(semantic_criteria, max_criteria_per_call)
    criteria_per_candidate = max(1, min(max(1, max_criteria_per_call), len(semantic_criteria) or 1))
    max_candidates = max(1, int(max_candidates_per_batch))
    candidate_batches: list[CandidateBatch] = []
    current: list[dict[str, Any]] = []
    current_input_tokens = int(static_prompt_tokens)

    def flush_current() -> None:
        nonlocal current, current_input_tokens
        if not current:
            return
        candidate_batches.append(
            CandidateBatch(
                candidates=current,
                estimated_input_tokens=current_input_tokens,
                estimated_output_tokens=_estimate_output_tokens(len(current), criteria_per_candidate),
            )
        )
        current = []
        current_input_tokens = int(static_prompt_tokens)

    for candidate in candidates:
        candidate_tokens = _candidate_cost(candidate)
        projected_input = current_input_tokens + candidate_tokens
        projected_count = len(current) + 1
        projected_output = _estimate_output_tokens(projected_count, criteria_per_candidate)
        exceeds_input = bool(current) and projected_input > window.input_budget
        exceeds_output = bool(current) and projected_output > window.output_budget
        exceeds_count = bool(current) and projected_count > max_candidates

        if exceeds_input or exceeds_output or exceeds_count:
            flush_current()
            projected_input = current_input_tokens + candidate_tokens

        current.append(candidate)
        current_input_tokens = projected_input

    flush_current()

    return ScoringBatchPlan(
        candidate_batches=candidate_batches,
        criterion_batches=criterion_batches,
        total_candidates=len(candidates),
        total_criteria=len(semantic_criteria),
        planner_settings={
            "staticPromptTokens": int(static_prompt_tokens),
            "inputBudgetTokens": window.input_budget,
            "outputBudgetTokens": window.output_budget,
            "contextWindowTokens": window.context_window,
            "reserveTokens": window.reserve,
            "maxCandidatesPerBatch": max_candidates,
            "maxCriteriaPerCall": max(1, int(max_criteria_per_call)),
        },
    )
