from src.services.scoring_batching import build_scoring_batch_plan
from src.services.token_budget import BudgetWindow


def _candidate(candidate_id: str, skills: str = "Python") -> dict:
    return {"id": candidate_id, "full_name": f"Candidate {candidate_id}", "skills_text": skills}


def test_scoring_plan_keeps_short_candidates_together():
    plan = build_scoring_batch_plan(
        candidates=[_candidate("1"), _candidate("2"), _candidate("3")],
        semantic_criteria=[{"key": "skills.python"}],
        static_prompt_tokens=50,
        window=BudgetWindow(context_window=1000, output_budget=500, reserve=100),
        max_candidates_per_batch=8,
        max_criteria_per_call=12,
    )

    assert [len(batch.candidates) for batch in plan.candidate_batches] == [3]
    assert plan.criterion_batches == [[{"key": "skills.python"}]]


def test_scoring_plan_splits_long_candidates_by_input_budget():
    long_text = "Python " * 500
    plan = build_scoring_batch_plan(
        candidates=[_candidate("1", long_text), _candidate("2", long_text)],
        semantic_criteria=[{"key": "skills.python"}],
        static_prompt_tokens=50,
        window=BudgetWindow(context_window=900, output_budget=200, reserve=100),
        max_candidates_per_batch=8,
        max_criteria_per_call=12,
    )

    assert [len(batch.candidates) for batch in plan.candidate_batches] == [1, 1]


def test_scoring_plan_splits_many_criteria_by_output_risk():
    criteria = [{"key": f"skills.{idx}"} for idx in range(25)]
    plan = build_scoring_batch_plan(
        candidates=[_candidate("1")],
        semantic_criteria=criteria,
        static_prompt_tokens=50,
        window=BudgetWindow(context_window=2000, output_budget=300, reserve=100),
        max_candidates_per_batch=8,
        max_criteria_per_call=10,
    )

    assert [len(batch) for batch in plan.criterion_batches] == [10, 10, 5]
