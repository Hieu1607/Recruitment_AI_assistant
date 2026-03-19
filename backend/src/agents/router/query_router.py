from __future__ import annotations

from dataclasses import dataclass

from src.models.matching import RoutingStrategy


@dataclass
class RoutingPlan:
    strategy: RoutingStrategy
    run_sql: bool
    run_llm: bool


def build_routing_plan(question: str) -> RoutingPlan:
    lowered = question.lower()
    semantic_signals = (
        "fit",
        "suitable",
        "best",
        "strong",
        "culture",
        "leadership",
        "communication",
        "potential",
    )
    deterministic_signals = (
        "count",
        "how many",
        "educated",
        "abroad",
        "cpa",
        "years",
        "in ",
        "from ",
        "skills",
    )

    has_semantic = any(signal in lowered for signal in semantic_signals)
    has_deterministic = any(signal in lowered for signal in deterministic_signals)

    if has_semantic and has_deterministic:
        return RoutingPlan(strategy=RoutingStrategy.HYBRID, run_sql=True, run_llm=True)
    if has_semantic:
        return RoutingPlan(strategy=RoutingStrategy.LLM_ONLY, run_sql=False, run_llm=True)
    return RoutingPlan(strategy=RoutingStrategy.SQL_ONLY, run_sql=True, run_llm=False)
