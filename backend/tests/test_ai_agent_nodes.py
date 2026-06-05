import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.ai_agent.nodes import _apply_dsl, _resolve_candidates, answer_node, dsl_node  # noqa: E402


def test_resolve_candidates_uses_scoped_current_candidates(monkeypatch):
    monkeypatch.setattr(
        "src.services.ai_agent.nodes._fetch_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("DB fetch should not run when current_candidates is provided")
        ),
    )

    state = {
        "current_candidates": [
            {
                "id": "candidate-1",
                "full_name": "Scoped One",
                "email": "one@example.com",
                "experience_years": "3.5",
                "skills_text": "Python",
            },
            {
                "id": "candidate-2",
                "full_name": "Scoped Two",
                "email": "two@example.com",
                "experience_years": "7",
                "skills_text": "Go",
            },
        ]
    }

    rows = _resolve_candidates(
        state,
        fields=["email", "experience_years"],
        candidate_ids=["candidate-2"],
    )

    assert rows == [
        {
            "id": "candidate-2",
            "full_name": "Scoped Two",
            "email": "two@example.com",
            "experience_years": 7.0,
        }
    ]


def test_resolve_candidates_falls_back_to_db_fetch_without_scope(monkeypatch):
    monkeypatch.setattr(
        "src.services.ai_agent.nodes._fetch_candidates",
        lambda fields, candidate_ids=None: [
            {"id": "db-candidate", "full_name": "DB Candidate"}
        ],
    )

    rows = _resolve_candidates({}, fields=["skills_text"], candidate_ids=["db-candidate"])

    assert rows == [{"id": "db-candidate", "full_name": "DB Candidate"}]


def test_apply_dsl_matches_accent_insensitive_full_name_queries():
    candidates = [
        {"id": "cand-1", "full_name": "Nguyen Minh Hieu"},
        {"id": "cand-2", "full_name": "Nguyễn Minh Hiếu"},
        {"id": "cand-3", "full_name": "Tran Thi Lan"},
    ]

    dsl = {"filters": {}, "must": [{"field": "full_name", "contains": "Hieu"}], "should": []}

    rows = _apply_dsl(candidates, dsl)

    assert rows == [
        {"id": "cand-1", "full_name": "Nguyen Minh Hieu"},
        {"id": "cand-2", "full_name": "Nguyễn Minh Hiếu"},
    ]


def test_answer_node_falls_back_to_dsl_candidates_when_llm_result_is_empty(monkeypatch):
    seen = {}

    def fake_build_answer_prompt(question, candidates, job_context=None):
        seen["call"] = {"question": question, "candidates": candidates, "job_context": job_context}
        return "prompt"

    monkeypatch.setattr(
        "src.services.ai_agent.nodes._get_llm",
        lambda: SimpleNamespace(
            generate=lambda prompt: SimpleNamespace(
                text="Nguyễn Minh Hiếu và Hoàng Lê Quân là hai ứng viên cần so sánh.",
                provider="test",
                model="fake",
                usage=None,
                raw=None,
            )
        ),
    )
    monkeypatch.setattr(
        "src.services.ai_agent.nodes.build_prompts.build_answer_prompt",
        fake_build_answer_prompt,
    )

    result = answer_node(
        {
            "question": "So sánh 2 CV của Nguyễn Minh Hiếu và Hoàng Lê Quân cho công việc này xem ai tốt hơn",
            "router_output": {"dsl_relevant_fields": ["full_name"], "llm_relevant_fields": ["skills_text"]},
            "dsl_candidates": [
                {"id": "cand-1", "full_name": "Nguyễn Minh Hiếu", "skills_text": "Python"},
                {"id": "cand-2", "full_name": "Hoàng Lê Quân", "skills_text": "Java"},
            ],
            "llm_result": {"total_qualified_candidates": 0, "qualified_candidates": {}},
            "current_candidates": [
                {"id": "cand-1", "full_name": "Nguyễn Minh Hiếu", "skills_text": "Python"},
                {"id": "cand-2", "full_name": "Hoàng Lê Quân", "skills_text": "Java"},
            ],
        }
    )

    assert seen["call"]["candidates"] == [
        {"id": "cand-1", "full_name": "Nguyễn Minh Hiếu", "skills_text": "Python"},
        {"id": "cand-2", "full_name": "Hoàng Lê Quân", "skills_text": "Java"},
    ]
    assert result["answer"] == "Nguyễn Minh Hiếu và Hoàng Lê Quân là hai ứng viên cần so sánh."


def test_answer_node_uses_llm_for_natural_no_match_response(monkeypatch):
    seen = {}

    def fake_build_answer_prompt(question, candidates, job_context=None):
        seen["call"] = {"question": question, "candidates": candidates, "job_context": job_context}
        return "prompt"

    monkeypatch.setattr(
        "src.services.ai_agent.nodes._get_llm",
        lambda: SimpleNamespace(
            generate=lambda prompt: SimpleNamespace(
                text="Hiện chưa có ứng viên nào trong danh sách phù hợp với tiêu chí này.",
                provider="test",
                model="fake",
                usage=None,
                raw=None,
            )
        ),
    )
    monkeypatch.setattr(
        "src.services.ai_agent.nodes.build_prompts.build_answer_prompt",
        fake_build_answer_prompt,
    )

    result = answer_node(
        {
            "question": "Who has 5+ years of Python experience?",
            "router_output": {"dsl_relevant_fields": ["experience_years"], "llm_relevant_fields": ["skills_text"]},
            "dsl_candidates": [],
            "llm_result": None,
            "current_candidates": [],
        }
    )

    assert seen["call"]["candidates"] == []
    assert result["answer"] == "Hiện chưa có ứng viên nào trong danh sách phù hợp với tiêu chí này."


def test_answer_node_keeps_all_named_candidates_for_comparison_even_if_llm_qualifies_one(monkeypatch):
    seen = {}

    def fake_build_answer_prompt(question, candidates, job_context=None):
        seen["call"] = {"question": question, "candidates": candidates, "job_context": job_context}
        return "prompt"

    monkeypatch.setattr(
        "src.services.ai_agent.nodes._get_llm",
        lambda: SimpleNamespace(
            generate=lambda prompt: SimpleNamespace(
                text="So sánh hai ứng viên hoàn tất.",
                provider="test",
                model="fake",
                usage=None,
                raw=None,
            )
        ),
    )
    monkeypatch.setattr(
        "src.services.ai_agent.nodes.build_prompts.build_answer_prompt",
        fake_build_answer_prompt,
    )

    result = answer_node(
        {
            "question": "So sánh 2 ứng viên Nguyễn Minh Hiếu và An Nguyen cho công việc này",
            "router_output": {"dsl_relevant_fields": ["full_name"], "llm_relevant_fields": ["skills_text"]},
            "dsl_candidates": [
                {"id": "cand-1", "full_name": "Nguyễn Minh Hiếu", "skills_text": "Python"},
                {"id": "cand-2", "full_name": "AN NGUYEN", "skills_text": "Java"},
            ],
            "llm_result": {"total_qualified_candidates": 1, "qualified_candidates": {"cand-1": "stronger fit"}},
            "current_candidates": [
                {"id": "cand-1", "full_name": "Nguyễn Minh Hiếu", "skills_text": "Python"},
                {"id": "cand-2", "full_name": "AN NGUYEN", "skills_text": "Java"},
            ],
        }
    )

    assert seen["call"]["candidates"] == [
        {"id": "cand-1", "full_name": "Nguyễn Minh Hiếu", "skills_text": "Python"},
        {"id": "cand-2", "full_name": "AN NGUYEN", "skills_text": "Java"},
    ]
    assert result["answer"] == "So sánh hai ứng viên hoàn tất."


def test_dsl_node_recovers_named_candidates_when_generated_dsl_filters_everything(monkeypatch):
    monkeypatch.setattr(
        "src.services.ai_agent.nodes._get_llm",
        lambda: SimpleNamespace(
            generate=lambda prompt: SimpleNamespace(
                text='{"filters": {}, "must": [{"field": "full_name", "contains": "Nguyễn Minh Hiếu"}, {"field": "full_name", "contains": "An Nguyen về công việc này"}], "should": []}',
                provider="test",
                model="fake",
                usage=None,
                raw=None,
            )
        ),
    )

    result = dsl_node(
        {
            "question": "So sánh 2 ứng viên Nguyễn Minh Hiếu và An Nguyen về công việc này",
            "router_output": {"dsl_relevant_fields": ["full_name"]},
            "current_candidates": [
                {"id": "cand-1", "full_name": "Nguyen Minh Hieu"},
                {"id": "cand-2", "full_name": "AN NGUYEN"},
                {"id": "cand-3", "full_name": "Tran Thi Lan"},
            ],
        }
    )

    assert result["dsl_candidates"] == [
        {"id": "cand-1", "full_name": "Nguyen Minh Hieu"},
        {"id": "cand-2", "full_name": "AN NGUYEN"},
    ]
