import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.scoring_preferences import calculate_weighted_score, normalize_section_weights  # noqa: E402
from src.services.scoring_signature import SCORING_SIGNATURE_VERSION, compute_scoring_signature  # noqa: E402


def test_scoring_signature_changes_when_hidden_text_changes():
    first = compute_scoring_signature(
        job_description_id="jd-1",
        jd_text="Need Python",
        hidden_text="Prefer RAG",
    )
    second = compute_scoring_signature(
        job_description_id="jd-1",
        jd_text="Need Python",
        hidden_text="Prefer MLOps",
    )

    assert first != second
    assert first.startswith(f"{SCORING_SIGNATURE_VERSION}:")


def test_normalize_section_weights_rejects_zero_total():
    try:
        normalize_section_weights({"skills": 0, "experience": -5})
    except ValueError as exc:
        assert "total" in str(exc).lower()
    else:
        raise AssertionError("Expected zero-total weights to fail")


def test_calculate_weighted_score_uses_raw_percentages_without_llm():
    result = calculate_weighted_score(
        raw_component_scores=[
            {
                "criterionKey": "skills.python",
                "section": "skills",
                "criterionType": "must_have",
                "evaluationMode": "semantic",
                "requirementText": "Python",
                "scorePercent": 80,
                "evidenceSummary": "Python project.",
            },
            {
                "criterionKey": "experience.years",
                "section": "experience",
                "criterionType": "must_have",
                "evaluationMode": "measurable",
                "requirementText": "2+ years",
                "scorePercent": 50,
                "evidenceSummary": "One year listed.",
            },
        ],
        section_weights={"skills": 75, "experience": 25},
        score_threshold=Decimal("70"),
    )

    assert result["totalScore"] == 72.5
    assert result["passedThreshold"] is True
    assert result["componentScores"][0]["scorePercent"] == 80
    assert result["componentScores"][0]["weightedScore"] == 60.0


def test_calculate_weighted_score_preserves_raw_criterion_weights_when_no_preference():
    result = calculate_weighted_score(
        raw_component_scores=[
            {
                "criterionKey": "education.degree",
                "section": "education",
                "weight": 0.2,
                "scorePercent": 100,
            },
            {
                "criterionKey": "skills.python",
                "section": "skills",
                "weight": 0.4,
                "scorePercent": 90,
            },
            {
                "criterionKey": "skills.git",
                "section": "skills",
                "weight": 0.4,
                "scorePercent": 0,
            },
        ],
        section_weights=None,
        score_threshold=Decimal("50"),
    )

    assert result["totalScore"] == 56.0
    assert result["componentScores"][0]["weightedScore"] == 20.0
    assert result["componentScores"][1]["weightedScore"] == 36.0
    assert result["componentScores"][2]["weightedScore"] == 0.0


def test_calculate_weighted_score_reweights_sections_without_double_counting():
    result = calculate_weighted_score(
        raw_component_scores=[
            {
                "criterionKey": "education.degree",
                "section": "education",
                "weight": 0.2,
                "scorePercent": 100,
            },
            {
                "criterionKey": "skills.python",
                "section": "skills",
                "weight": 0.4,
                "scorePercent": 90,
            },
            {
                "criterionKey": "skills.git",
                "section": "skills",
                "weight": 0.4,
                "scorePercent": 0,
            },
        ],
        section_weights={"skills": 50, "education": 50},
        score_threshold=Decimal("50"),
    )

    assert result["totalScore"] == 72.5
    assert result["componentScores"][0]["weightedScore"] == 50.0
    assert result["componentScores"][1]["weightedScore"] == 22.5
    assert result["componentScores"][2]["weightedScore"] == 0.0
