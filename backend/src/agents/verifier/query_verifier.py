from __future__ import annotations

from dataclasses import dataclass

from src.models.matching import RoutingStrategy


@dataclass
class QueryExecutionContext:
    question: str
    strategy: RoutingStrategy
    sql_candidate_ids: list[str]
    llm_candidate_ids: list[str]


@dataclass
class VerifiedQueryResult:
    answer_text: str
    matched_candidate_ids: list[str]
    matched_count: int
    strategy: RoutingStrategy
    tool_trace: dict


def _is_count_question(question: str) -> bool:
    lowered = question.lower()
    return "how many" in lowered or "count" in lowered or lowered.startswith("number of")


def verify_query_result(context: QueryExecutionContext) -> VerifiedQueryResult:
    sql_set = set(context.sql_candidate_ids)
    llm_set = set(context.llm_candidate_ids)

    if context.strategy == RoutingStrategy.SQL_ONLY:
        final_ids = list(context.sql_candidate_ids)
    elif context.strategy == RoutingStrategy.LLM_ONLY:
        final_ids = list(context.llm_candidate_ids)
    else:
        intersection = [candidate_id for candidate_id in context.sql_candidate_ids if candidate_id in llm_set]
        final_ids = intersection if intersection else list(context.sql_candidate_ids)

    if _is_count_question(context.question) and context.sql_candidate_ids:
        final_ids = list(context.sql_candidate_ids)

    matched_count = len(final_ids)
    answer = f"Matched {matched_count} candidate(s) for: {context.question}"

    return VerifiedQueryResult(
        answer_text=answer,
        matched_candidate_ids=final_ids,
        matched_count=matched_count,
        strategy=context.strategy,
        tool_trace={
            "sql_count": len(context.sql_candidate_ids),
            "llm_count": len(context.llm_candidate_ids),
            "final_count": matched_count,
            "fallback_sql_for_count": _is_count_question(context.question),
        },
    )
