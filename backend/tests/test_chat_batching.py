from src.services.ai_agent.chat_batching import (
    AnswerMode,
    build_chat_map_batches,
    choose_answer_mode,
    compact_candidate_identity,
)
from src.services.token_budget import BudgetWindow


def test_chat_map_batches_group_short_candidates():
    candidates = [{"id": str(i), "full_name": f"C{i}", "skills_text": "Python"} for i in range(5)]
    batches = build_chat_map_batches(
        question="Who knows Python?",
        candidates=candidates,
        job_context={},
        static_prompt_tokens=100,
        window=BudgetWindow(context_window=2000, output_budget=500, reserve=200),
        max_candidates_per_batch=40,
    )

    assert [len(batch.candidates) for batch in batches] == [5]


def test_chat_map_batches_split_long_candidates():
    candidates = [
        {"id": "1", "full_name": "One", "skills_text": "Python " * 500},
        {"id": "2", "full_name": "Two", "skills_text": "Python " * 500},
    ]
    batches = build_chat_map_batches(
        question="Who knows Python?",
        candidates=candidates,
        job_context={},
        static_prompt_tokens=100,
        window=BudgetWindow(context_window=1000, output_budget=300, reserve=100),
        max_candidates_per_batch=40,
    )

    assert [len(batch.candidates) for batch in batches] == [1, 1]


def test_choose_answer_mode_uses_compact_mode_for_large_result_sets():
    candidates = [{"id": str(i), "full_name": f"C{i}", "skills_text": "Python"} for i in range(100)]
    mode = choose_answer_mode(
        candidates=candidates,
        detailed_threshold=10,
        compact_threshold=50,
        estimated_full_tokens=9000,
        final_input_budget=3000,
    )

    assert mode == AnswerMode.COMPACT_ID_NAME


def test_compact_candidate_identity_drops_profile_fields():
    compact = compact_candidate_identity({"id": "1", "full_name": "A", "skills_text": "Python"})

    assert compact == {"id": "1", "full_name": "A"}
