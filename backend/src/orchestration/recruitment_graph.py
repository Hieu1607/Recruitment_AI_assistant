from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecruitmentGraphState:
    user_id: str
    question: str
    routing_strategy: str | None = None
    sql_result: dict[str, Any] | None = None
    llm_result: dict[str, Any] | None = None
    final_response: dict[str, Any] = field(default_factory=dict)


def route_query(state: RecruitmentGraphState) -> RecruitmentGraphState:
    question = state.question.lower()
    if "how many" in question or "count" in question:
        state.routing_strategy = "sql_only"
    elif "summarize" in question or "fit" in question:
        state.routing_strategy = "llm_only"
    else:
        state.routing_strategy = "hybrid"
    return state


def run_sql_tool(state: RecruitmentGraphState) -> RecruitmentGraphState:
    state.sql_result = {"matched_count": 0, "candidate_ids": []}
    return state


def run_llm_tool(state: RecruitmentGraphState) -> RecruitmentGraphState:
    state.llm_result = {"reasoning": "placeholder", "candidate_ids": []}
    return state


def compose_response(state: RecruitmentGraphState) -> RecruitmentGraphState:
    state.final_response = {
        "routing_strategy": state.routing_strategy,
        "matched_count": (state.sql_result or {}).get("matched_count", 0),
        "matched_candidate_ids": (state.sql_result or {}).get("candidate_ids", []),
        "answer": "Foundation graph skeleton executed.",
    }
    return state


def run_graph(state: RecruitmentGraphState) -> RecruitmentGraphState:
    state = route_query(state)

    if state.routing_strategy in {"sql_only", "hybrid"}:
        state = run_sql_tool(state)

    if state.routing_strategy in {"llm_only", "hybrid"}:
        state = run_llm_tool(state)

    return compose_response(state)
