from __future__ import annotations

import uuid
from typing import Any, cast

from src.agents.tools import dsl_search_tool as tool
from src.services.llm.llm_client import LLMClientError


class FakeSession:
    def __init__(self, rows: list[uuid.UUID] | None = None) -> None:
        self._rows = rows or []
        self.last_stmt = None

    def scalars(self, stmt):
        self.last_stmt = stmt
        return list(self._rows)


def test_run_dsl_search_builds_prompt_with_question_schema_and_stack(
    monkeypatch,
) -> None:
    captured = {}

    def fake_generate_json(request):
        captured["prompt"] = request.prompt
        return {
            "queryIntent": {
                "logic": "and",
                "filters": [{"field": "educated", "op": "eq", "value": True}],
                "limit": 20,
            }
        }

    monkeypatch.setattr(tool, "generate_json", fake_generate_json)

    _ = tool.run_dsl_search(
        cast(Any, FakeSession(rows=[])), "Find Python candidates in Hanoi", limit=20
    )

    prompt = captured["prompt"]
    assert "Recruiter question: Find Python candidates in Hanoi" in prompt
    assert "candidate_profiles columns" in prompt
    assert "Current technology stack" in prompt
    assert "FastAPI" in prompt
    assert "SQLAlchemy" in prompt
    assert "requested_limit: 20" in prompt


def test_run_dsl_search_uses_llm_intent_and_returns_ids(monkeypatch) -> None:
    test_ids = [uuid.uuid4(), uuid.uuid4()]
    session = FakeSession(rows=test_ids)

    def fake_generate_json(_request):
        return {
            "queryIntent": {
                "logic": "and",
                "filters": [
                    {"field": "location_normalized", "op": "contains", "value": "Hanoi"}
                ],
                "limit": 2,
            }
        }

    monkeypatch.setattr(tool, "generate_json", fake_generate_json)

    result = tool.run_dsl_search(
        cast(Any, session), "Find candidates in Hanoi", limit=10
    )

    assert result.candidate_ids == [str(item) for item in test_ids]
    assert result.matched_count == 2
    assert result.trace["tool"] == "dsl_search"
    assert result.trace["fallback_reason"] is None
    assert result.trace["matched_via"] == "dsl_filters"


def test_run_dsl_search_fallback_when_llm_fails(monkeypatch) -> None:
    session = FakeSession(rows=[])

    def fail_generate_json(_request):
        raise LLMClientError("boom")

    monkeypatch.setattr(tool, "generate_json", fail_generate_json)

    result = tool.run_dsl_search(
        cast(Any, session), "completely random query", limit=10
    )

    assert result.candidate_ids == []
    assert result.matched_count == 0
    assert result.trace["matched_via"] == "no_valid_filters"
    assert str(result.trace["fallback_reason"]).startswith("llm_failed:")


def test_run_dsl_search_uses_heuristic_when_llm_returns_empty_filters(
    monkeypatch,
) -> None:
    expected_id = uuid.uuid4()
    session = FakeSession(rows=[expected_id])

    def empty_intent_generate_json(_request):
        return {"queryIntent": {"logic": "and", "filters": [], "limit": 10}}

    monkeypatch.setattr(tool, "generate_json", empty_intent_generate_json)

    result = tool.run_dsl_search(
        cast(Any, session), "Find people with Python skills and 3 years", limit=10
    )

    assert result.candidate_ids == [str(expected_id)]
    assert result.matched_count == 1
    assert result.trace["fallback_reason"] == "llm_intent_empty_filters"
    assert result.trace["matched_via"] == "dsl_filters"
