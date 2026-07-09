import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services import score_candidate  # noqa: E402
from src.services.score_candidate import evaluate_candidate_profiles_raw  # noqa: E402


def _candidates(count: int) -> list[dict]:
    return [
        {
            "id": f"candidate-{index}",
            "full_name": f"Candidate {index}",
            "display_name": f"Candidate {index}",
            "skills_text": "Python",
        }
        for index in range(count)
    ]


def _semantic_rubric() -> dict:
    return {
        "criteria": [
            {
                "key": "skills.python",
                "section": "skills",
                "requirementText": "Strong Python",
                "type": "semantic",
                "measurable": None,
                "weight": 1.0,
            }
        ],
        "sectionWeights": {"skills": 1.0},
    }


def test_batch_raw_evaluation_extracts_rubric_once_for_ten_candidates(monkeypatch):
    candidates = _candidates(10)
    calls = {"rubric": 0, "semantic": 0}

    monkeypatch.setattr(score_candidate, "_scoring_llm_provider", lambda: object())

    def extract_rubric(**_kwargs):
        calls["rubric"] += 1
        return _semantic_rubric()

    semantic_batches = [candidates[:8], candidates[8:]]

    def generate_semantic(**_kwargs):
        batch = semantic_batches[calls["semantic"]]
        calls["semantic"] += 1
        return {
            candidate["id"]: {
                "criteria": {
                    "skills.python": {
                        "scorePercent": 80,
                        "evidenceSummary": "Python project",
                    }
                }
            }
            for candidate in batch
        }

    monkeypatch.setattr(score_candidate, "_extract_locked_rubric", extract_rubric)
    monkeypatch.setattr(
        score_candidate,
        "_generate_semantic_scores_with_retries",
        generate_semantic,
    )
    monkeypatch.setattr(score_candidate.settings, "SCORING_MAX_CANDIDATES_PER_BATCH", 8)

    results = evaluate_candidate_profiles_raw(
        candidates=candidates,
        job_description_text="Need strong Python",
    )

    assert calls == {"rubric": 1, "semantic": 2}
    assert set(results) == {candidate["id"] for candidate in candidates}
    assert all(
        result["rawComponentScores"][0]["scorePercent"] == 80
        for result in results.values()
    )


def test_batch_raw_evaluation_skips_semantic_llm_for_measurable_only_rubric(monkeypatch):
    candidates = [
        {
            **candidate,
            "experience_years": 5,
        }
        for candidate in _candidates(3)
    ]
    rubric = {
        "criteria": [
            {
                "key": "experience.years",
                "section": "experience",
                "requirementText": "At least 3 years",
                "type": "must_have",
                "measurable": {
                    "field": "experience_years",
                    "operator": ">=",
                    "value": 3,
                },
                "weight": 1.0,
            }
        ],
        "sectionWeights": {"experience": 1.0},
    }
    semantic_calls = []

    monkeypatch.setattr(score_candidate, "_scoring_llm_provider", lambda: object())
    monkeypatch.setattr(
        score_candidate,
        "_extract_locked_rubric",
        lambda **_kwargs: rubric,
    )
    monkeypatch.setattr(
        score_candidate,
        "_generate_semantic_scores_with_retries",
        lambda **kwargs: semantic_calls.append(kwargs) or {},
    )

    results = evaluate_candidate_profiles_raw(
        candidates=candidates,
        job_description_text="Need at least 3 years",
    )

    assert semantic_calls == []
    assert all(
        result["rawComponentScores"][0]["scorePercent"] == 100
        for result in results.values()
    )
