import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.score_candidate import (  # noqa: E402
    _build_candidate_score,
    _coerce_passed_threshold,
    _normalize_rubric,
    _parse_rubric_response,
)


def test_parse_rubric_response_preserves_measurable_experience_criterion():
    rubric = _parse_rubric_response(
        """
        {
          "criteria": [
            {
              "key": "experience_years",
              "section": "experience",
              "requirementText": "At least 5 years of backend experience",
              "type": "must_have",
              "measurable": {"field": "experience_years", "operator": ">=", "value": 5}
            }
          ]
        }
        """
    )

    criterion = rubric["criteria"][0]
    assert criterion["section"] == "experience"
    assert criterion["type"] == "must_have"
    assert criterion["measurable"]["field"] == "experience_years"
    assert criterion["measurable"]["operator"] == ">="
    assert criterion["measurable"]["value"] == 5


def test_normalize_rubric_rejects_invalid_section_and_blank_requirement():
    rubric = _normalize_rubric(
        {
            "criteria": [
                {
                    "key": "bad",
                    "section": "salary",
                    "requirementText": "Need salary fit",
                    "type": "semantic",
                },
                {
                    "key": "also_bad",
                    "section": "skills",
                    "requirementText": " ",
                    "type": "semantic",
                },
                {
                    "key": "good",
                    "section": "skills",
                    "requirementText": "Strong Python",
                    "type": "semantic",
                },
            ]
        },
        section_weights={"skills": 70, "education": 30},
    )

    assert [criterion["key"] for criterion in rubric["criteria"]] == ["good"]
    assert rubric["sectionWeights"] == {"skills": 1.0}


def test_build_candidate_score_treats_experience_threshold_as_pass_fail_without_bonus():
    rubric = _normalize_rubric(
        {
            "criteria": [
                {
                    "key": "experience_years",
                    "section": "experience",
                    "requirementText": "5+ years experience",
                    "type": "must_have",
                    "measurable": {"field": "experience_years", "operator": ">=", "value": 5},
                }
            ]
        },
        section_weights={"experience": 100},
    )

    candidate_five = _build_candidate_score(
        candidate={
            "id": "cand-5",
            "experience_years": 5,
        },
        rubric=rubric,
        semantic_result={},
        score_threshold=Decimal("50"),
    )
    candidate_ten = _build_candidate_score(
        candidate={
            "id": "cand-10",
            "experience_years": 10,
        },
        rubric=rubric,
        semantic_result={},
        score_threshold=Decimal("50"),
    )
    candidate_four = _build_candidate_score(
        candidate={
            "id": "cand-4",
            "experience_years": 4,
        },
        rubric=rubric,
        semantic_result={},
        score_threshold=Decimal("50"),
    )

    assert candidate_five["totalScore"] == 100.0
    assert candidate_ten["totalScore"] == 100.0
    assert candidate_four["totalScore"] == 0.0


def test_build_candidate_score_uses_upper_bound_as_fixed_gate_not_arbitrary_bonus():
    rubric = _normalize_rubric(
        {
            "criteria": [
                {
                    "key": "languages.english_communication",
                    "section": "languages",
                    "requirementText": "Good English communication",
                    "type": "must_have",
                    "measurable": {"field": "languages", "operator": "contains", "value": "English"},
                },
                {
                    "key": "languages.ielts_7_5_upper_bound",
                    "section": "languages",
                    "requirementText": "IELTS 7.5+ is a plus",
                    "type": "upper_bound",
                    "measurable": {"field": "languages.ielts", "operator": ">=", "value": 7.5},
                },
            ]
        },
        section_weights={"languages": 100},
    )

    baseline = _build_candidate_score(
        candidate={"id": "cand-a", "languages_text": "English communication"},
        rubric=rubric,
        semantic_result={},
        score_threshold=Decimal("50"),
    )
    unlocked = _build_candidate_score(
        candidate={"id": "cand-b", "languages_text": "English communication. IELTS 7.5"},
        rubric=rubric,
        semantic_result={},
        score_threshold=Decimal("50"),
    )

    assert baseline["totalScore"] == 50.0
    assert unlocked["totalScore"] == 100.0


def test_coerce_passed_threshold_ignores_llm_boolean_and_uses_backend_total_score():
    normalized = _coerce_passed_threshold(
        {
            "candidateId": "cand-1",
            "totalScore": 40,
            "passedThreshold": True,
            "rationale": "LLM guessed true",
            "componentScores": [],
        },
        score_threshold=Decimal("50"),
    )

    assert normalized["passedThreshold"] is False
