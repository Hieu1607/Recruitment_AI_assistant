import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.ai_agent.nodes import _apply_dsl, _resolve_candidates, answer_node, dsl_node, llm_node, router_node  # noqa: E402


def test_get_llm_uses_higher_output_token_budget_for_ai_agent(monkeypatch):
    import src.services.ai_agent.nodes as nodes_module

    captured = {}

    class DummyLLM:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr(nodes_module, "LLMProvider", DummyLLM)
    monkeypatch.setattr(nodes_module, "_llm", None)

    llm = nodes_module._get_llm()

    assert isinstance(llm, DummyLLM)
    assert captured["kwargs"]["max_tokens"] == 8192


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
        {"id": "cand-1", "full_name": "Nguyễn Minh Hiếu", "skills_text": "Python", "summary_text": None},
        {"id": "cand-2", "full_name": "Hoàng Lê Quân", "skills_text": "Java", "summary_text": None},
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
        {"id": "cand-1", "full_name": "Nguyễn Minh Hiếu", "skills_text": "Python", "summary_text": None},
        {"id": "cand-2", "full_name": "AN NGUYEN", "skills_text": "Java", "summary_text": None},
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


def test_dsl_node_drops_current_job_title_filters_when_field_not_allowed(monkeypatch):
    monkeypatch.setattr(
        "src.services.ai_agent.nodes._get_llm",
        lambda: SimpleNamespace(
            generate=lambda prompt: SimpleNamespace(
                text='{"filters": {"experience_years": {"operator": "gte", "value": 2}, "current_job_title": {"operator": "contains", "value": "Giáo viên dạy toán"}}, "must": [{"field": "current_job_title", "contains": "Giáo viên dạy toán"}], "should": []}',
                provider="test",
                model="fake",
                usage=None,
                raw=None,
            )
        ),
    )

    result = dsl_node(
        {
            "question": "Đánh giá ứng viên phù hợp nhất với công việc này",
            "router_output": {"dsl_relevant_fields": ["experience_years"]},
            "current_candidates": [
                {
                    "id": "cand-1",
                    "full_name": "Lê Thị Hòa",
                    "experience_years": 3,
                    "current_job_title": "Math Teacher at Nguyễn Du Secondary School",
                },
                {
                    "id": "cand-2",
                    "full_name": "Nguyễn Văn An",
                    "experience_years": 5,
                    "current_job_title": "Senior Software Engineer",
                },
                {
                    "id": "cand-3",
                    "full_name": "Intern One",
                    "experience_years": 1,
                    "current_job_title": "Math Teacher",
                },
            ],
        }
    )

    assert result["dsl_candidates"] == [
        {"id": "cand-1", "full_name": "Lê Thị Hòa", "experience_years": 3},
        {"id": "cand-2", "full_name": "Nguyễn Văn An", "experience_years": 5},
    ]


def test_dsl_node_drops_semantic_only_filters_when_fields_not_allowed(monkeypatch):
    monkeypatch.setattr(
        "src.services.ai_agent.nodes._get_llm",
        lambda: SimpleNamespace(
            generate=lambda prompt: SimpleNamespace(
                text='{"filters": {"experience_years": {"operator": "gte", "value": 2}, "major": {"operator": "contains", "value": "Education"}, "cpa": {"operator": "contains", "value": "CPA"}, "contact": {"operator": "contains", "value": "Lan"}}, "must": [{"field": "major", "contains": "Education"}, {"field": "contact", "contains": "Lan"}], "should": [{"field": "cpa", "contains": "CPA"}]}',
                provider="test",
                model="fake",
                usage=None,
                raw=None,
            )
        ),
    )

    result = dsl_node(
        {
            "question": "Tìm ứng viên phù hợp công việc này",
            "router_output": {"dsl_relevant_fields": ["experience_years"]},
            "current_candidates": [
                {
                    "id": "cand-1",
                    "full_name": "Lê Thị Hòa",
                    "experience_years": 3,
                    "major": "Education",
                    "cpa": "3.5 GPA",
                    "contact": "Ms. Lan",
                },
                {
                    "id": "cand-2",
                    "full_name": "Nguyễn Văn An",
                    "experience_years": 5,
                    "major": "Computer Science",
                    "cpa": "CPA inactive",
                    "contact": "Mr. Minh",
                },
                {
                    "id": "cand-3",
                    "full_name": "Intern One",
                    "experience_years": 1,
                    "major": "Education",
                    "cpa": "CPA",
                    "contact": "Ms. Lan",
                },
            ],
        }
    )

    assert result["dsl_candidates"] == [
        {"id": "cand-1", "full_name": "Lê Thị Hòa", "experience_years": 3},
        {"id": "cand-2", "full_name": "Nguyễn Văn An", "experience_years": 5},
    ]


def test_llm_node_always_includes_summary_text_for_semantic_filtering(monkeypatch):
    seen = {}

    def fake_build_llm_query_prompt(question, candidates, job_context=None):
        seen["call"] = {"question": question, "candidates": candidates, "job_context": job_context}
        return "prompt"

    monkeypatch.setattr(
        "src.services.ai_agent.nodes._get_llm",
        lambda: SimpleNamespace(
            generate=lambda prompt: SimpleNamespace(
                text='{"total_qualified_candidates": 1, "qualified_candidates": {"cand-1": "summary match"}}',
                provider="test",
                model="fake",
                usage=None,
                raw=None,
            )
        ),
    )
    monkeypatch.setattr(
        "src.services.ai_agent.nodes.build_prompts.build_llm_query_prompt",
        fake_build_llm_query_prompt,
    )

    result = llm_node(
        {
            "question": "Ai phù hợp với vị trí thiên về product mindset?",
            "router_output": {"llm_relevant_fields": ["skills_text"]},
            "current_candidates": [
                {
                    "id": "cand-1",
                    "full_name": "Candidate One",
                    "skills_text": "Python",
                    "summary_text": "Strong product sense and customer discovery background.",
                }
            ],
        }
    )

    assert seen["call"]["candidates"] == [
        {
            "id": "cand-1",
            "full_name": "Candidate One",
            "skills_text": "Python",
            "summary_text": "Strong product sense and customer discovery background.",
        }
    ]
    assert result["llm_result"]["qualified_candidates"] == {"cand-1": "summary match"}


def test_router_node_overrides_educated_only_route_for_not_graduated_queries(monkeypatch):
    monkeypatch.setattr(
        "src.services.ai_agent.nodes._get_llm",
        lambda: SimpleNamespace(
            generate=lambda prompt: SimpleNamespace(
                text='{"is_recruitment_related": true, "refusal_message": null, "relevant_fields": ["graduation_status"], "dsl_question_query": "graduation_status = final_year", "llm_question_query": null, "dsl_relevant_fields": ["graduation_status"], "llm_relevant_fields": [], "reasoning": "Structured graduation status filter"}',
                provider="test",
                model="fake",
                usage=None,
                raw=None,
            )
        ),
    )

    result = router_node({"question": "Tìm các ứng viên chưa tốt nghiệp đại học"})

    assert result["router_output"]["dsl_question_query"] is None
    assert result["router_output"]["llm_question_query"] == "Tìm các ứng viên chưa tốt nghiệp đại học"
    assert result["router_output"]["llm_relevant_fields"] == ["education_text", "summary_text"]
    assert "graduation-status semantics" in result["router_output"]["reasoning"]


def test_router_node_moves_semantic_only_fields_from_dsl_to_llm(monkeypatch):
    monkeypatch.setattr(
        "src.services.ai_agent.nodes._get_llm",
        lambda: SimpleNamespace(
            generate=lambda prompt: SimpleNamespace(
                text='{"is_recruitment_related": true, "refusal_message": null, "relevant_fields": ["current_job_title", "major", "cpa", "contact", "experience_years"], "dsl_question_query": "SELECT * FROM candidates WHERE current_job_title = \'Giáo viên dạy toán\' AND major = \'Education\' AND experience_years >= 2", "llm_question_query": null, "dsl_relevant_fields": ["current_job_title", "major", "cpa", "contact", "experience_years"], "llm_relevant_fields": [], "reasoning": "Structured title, profile, and years filter"}',
                provider="test",
                model="fake",
                usage=None,
                raw=None,
            )
        ),
    )

    result = router_node({"question": "Đánh giá ứng viên phù hợp nhất với công việc này"})

    assert result["router_output"]["dsl_relevant_fields"] == ["experience_years"]
    assert result["router_output"]["llm_relevant_fields"] == [
        "current_job_title",
        "major",
        "cpa",
        "contact",
    ]
    assert result["router_output"]["llm_question_query"] == "Đánh giá ứng viên phù hợp nhất với công việc này"
    assert "semantic-field override" in result["router_output"]["reasoning"]


def test_router_node_falls_back_when_truncated_json_parses_as_list(monkeypatch):
    monkeypatch.setattr(
        "src.services.ai_agent.nodes._get_llm",
        lambda: SimpleNamespace(
            generate=lambda prompt: SimpleNamespace(
                text='{"is_recruitment_related": true, "refusal_message": null, "relevant_fields": ["full_name", "email"]',
                provider="test",
                model="fake",
                usage=None,
                raw={"choices": [{"finish_reason": "length"}]},
            )
        ),
    )

    result = router_node({"question": "Bạn đang có thông tin những ứng viên nào"})

    assert result["router_output"] == {
        "is_recruitment_related": True,
        "refusal_message": None,
        "relevant_fields": [],
        "dsl_question_query": None,
        "llm_question_query": "Bạn đang có thông tin những ứng viên nào",
        "dsl_relevant_fields": [],
        "llm_relevant_fields": [],
        "reasoning": "Parse failure – fell back to LLM path",
    }
