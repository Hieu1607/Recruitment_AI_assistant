import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.candidate_profile import CandidateProfile  # noqa: E402
from src.models.resume_document import ResumeDocument  # noqa: E402
from src.services import score_candidate  # noqa: E402
from src.services.llm_service import LLMProvider  # noqa: E402
from src.services.score_candidate import (  # noqa: E402
    _attach_candidate_metadata,
    _build_candidate_score,
    _coerce_passed_threshold,
    _generate_semantic_scores_with_retries,
    _generate_json_with_retries,
    _normalize_measurable,
    _parse_json_object,
    _normalize_rubric,
    _parse_rubric_response,
    _parse_semantic_scores,
    _profile_to_candidate_dict,
    _safe_parse_semantic_scores,
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


def test_profile_to_candidate_dict_prefers_profile_name_and_includes_resume_filename():
    resume = ResumeDocument(original_file_name="mai_nguyen_cv.pdf", storage_uri="s3://bucket/cv.pdf")
    profile = CandidateProfile(full_name="Mai Nguyen", resume_document=resume)

    candidate = _profile_to_candidate_dict(profile)

    assert candidate["full_name"] == "Mai Nguyen"
    assert candidate["resume_file_name"] == "mai_nguyen_cv.pdf"
    assert candidate["display_name"] == "Mai Nguyen"


def test_profile_to_candidate_dict_uses_resume_filename_when_name_missing():
    resume = ResumeDocument(original_file_name="senior_backend_candidate.pdf", storage_uri="s3://bucket/cv.pdf")
    profile = CandidateProfile(full_name="", resume_document=resume)

    candidate = _profile_to_candidate_dict(profile)

    assert candidate["display_name"] == "senior_backend_candidate.pdf"


def test_attach_candidate_metadata_updates_fallback_llm_score():
    candidate = {
        "id": "cand-1",
        "full_name": "An Tran",
        "resume_file_name": "an_tran_cv.pdf",
        "display_name": "An Tran",
    }

    score = _attach_candidate_metadata({"candidateId": "cand-1", "totalScore": 90}, candidate)

    assert score["candidateName"] == "An Tran"
    assert score["resumeFileName"] == "an_tran_cv.pdf"
    assert score["candidateDisplayName"] == "An Tran"


def test_normalize_measurable_rejects_non_candidateprofile_skill_fields():
    measurable = _normalize_measurable(
        {
            "field": "python_skill",
            "operator": "==",
            "value": True,
        }
    )

    assert measurable is None


def test_normalize_rubric_downgrades_unsupported_measurable_to_semantic():
    rubric = _normalize_rubric(
        {
            "criteria": [
                {
                    "key": "python_skill",
                    "section": "skills",
                    "requirementText": "Strong Python skills",
                    "type": "must_have",
                    "measurable": {"field": "python_skill", "operator": "==", "value": True},
                }
            ]
        },
        section_weights={"skills": 100},
    )

    criterion = rubric["criteria"][0]
    assert criterion["measurable"] is None
    assert criterion["type"] == "semantic"


def test_normalize_rubric_drops_invented_experience_threshold_when_jd_has_no_years():
    rubric = _normalize_rubric(
        {
            "criteria": [
                {
                    "key": "experience_years",
                    "section": "experience",
                    "requirementText": "5+ years of experience",
                    "type": "must_have",
                    "measurable": {"field": "experience_years", "operator": ">=", "value": 5},
                },
                {
                    "key": "skills.python",
                    "section": "skills",
                    "requirementText": "Python",
                    "type": "semantic",
                },
            ]
        },
        section_weights={"experience": 50, "skills": 50},
        source_text=(
            "AI Engineer\n"
            "Design, develop, and deploy AI/ML models. "
            "Work with Python, TensorFlow/PyTorch, Docker, and cloud platforms."
        ),
    )

    assert [criterion["key"] for criterion in rubric["criteria"]] == ["skills.python"]
    assert rubric["sectionWeights"] == {"skills": 1.0}


def test_normalize_rubric_keeps_explicit_experience_threshold_from_jd():
    rubric = _normalize_rubric(
        {
            "criteria": [
                {
                    "key": "experience_years",
                    "section": "experience",
                    "requirementText": "5+ years of experience",
                    "type": "must_have",
                    "measurable": {"field": "experience_years", "operator": ">=", "value": 5},
                }
            ]
        },
        section_weights={"experience": 100},
        source_text="Requirements: 5+ years of experience building production AI systems.",
    )

    assert rubric["criteria"][0]["measurable"] == {
        "field": "experience_years",
        "operator": ">=",
        "value": 5.0,
    }


def test_normalize_rubric_drops_invented_graduation_status_when_jd_has_no_degree_requirement():
    rubric = _normalize_rubric(
        {
            "criteria": [
                {
                    "key": "graduation_status",
                    "section": "education",
                    "requirementText": "final_year student",
                    "type": "must_have",
                    "measurable": {"field": "graduation_status", "operator": "=", "value": "final_year"},
                }
            ]
        },
        section_weights={"education": 100},
        source_text="AI Engineer. Build ML models and integrate AI services into backend systems.",
    )

    assert rubric == {"criteria": [], "sectionWeights": {}}


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
    assert candidate_five["componentScores"][0]["evaluationMode"] == "measurable"
    assert "Matched" in candidate_five["componentScores"][0]["evidenceSummary"]


def test_normalize_rubric_downgrades_unsupported_language_thresholds_to_semantic():
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

    assert all(criterion["measurable"] is None for criterion in rubric["criteria"])
    assert all(criterion["type"] == "semantic" or criterion["type"] == "upper_bound" for criterion in rubric["criteria"])


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


def test_build_candidate_score_marks_semantic_components_and_builds_score_aligned_rationale():
    rubric = _normalize_rubric(
        {
            "criteria": [
                {
                    "key": "skills.python_depth",
                    "section": "skills",
                    "requirementText": "Deep Python expertise",
                    "type": "semantic",
                }
            ]
        },
        section_weights={"skills": 100},
    )

    scored = _build_candidate_score(
        candidate={
            "id": "cand-sem",
            "full_name": "Linh Pham",
            "resume_file_name": "linh_pham.pdf",
            "display_name": "Linh Pham",
        },
        rubric=rubric,
        semantic_result={
            "rationale": "Generic LLM rationale that may not match recalculated scores.",
            "criteria": {
                "skills.python_depth": {
                    "score": 85,
                    "evidenceSummary": "Candidate led Python backend services in production.",
                }
            },
        },
        score_threshold=Decimal("50"),
    )

    component = scored["componentScores"][0]
    assert component["evaluationMode"] == "semantic"
    assert component["criterionType"] == "semantic"
    assert component["requirementText"] == "Deep Python expertise"
    assert scored["candidateName"] == "Linh Pham"
    assert scored["resumeFileName"] == "linh_pham.pdf"
    assert scored["candidateDisplayName"] == "Linh Pham"
    assert scored["rationale"] == (
        "Overall score 85.0/100. Strong matches: Deep Python expertise - "
        "Candidate led Python backend services in production."
    )


def test_safe_parse_semantic_scores_returns_empty_mapping_on_invalid_json():
    parsed = _safe_parse_semantic_scores('{"scores":[{"candidateId":"1","criteria":[{"criterionKey":"skills.python"}]}')

    assert parsed == {}


def test_parse_semantic_scores_scales_binary_and_fractional_llm_scores_to_percentages():
    parsed = _parse_semantic_scores(
        """
        {
          "scores": [
            {
              "candidateId": "cand-1",
              "criteria": [
                {"criterionKey": "skills.python", "score": 1, "evidenceSummary": "Clear Python evidence."},
                {"criterionKey": "skills.cloud", "score": 0, "evidenceSummary": "No cloud evidence."},
                {"criterionKey": "skills.ml", "score": 0.75, "evidenceSummary": "Partial ML evidence."},
                {"criterionKey": "skills.api", "score": 85, "evidenceSummary": "Strong API evidence."}
              ]
            }
          ]
        }
        """
    )

    criteria = parsed["cand-1"]["criteria"]
    assert criteria["skills.python"]["score"] == 100.0
    assert criteria["skills.cloud"]["score"] == 0.0
    assert criteria["skills.ml"]["score"] == 75.0
    assert criteria["skills.api"]["score"] == 85.0


def test_parse_json_object_repairs_trailing_commas():
    parsed = _parse_json_object(
        """
        {
          "scores": [
            {
              "candidateId": "cand-1",
              "rationale": "ok",
            }
          ],
        }
        """
    )

    assert parsed["scores"][0]["candidateId"] == "cand-1"


def test_generate_json_with_retries_falls_back_from_llama_to_gpt_oss():
    class FakeLLM:
        def __init__(self, *, provider="groq", model_name="llama-3.1-8b-instant", responses=None):
            self.provider = provider
            self.model_name = model_name
            self._responses = list(responses or [])
            self.calls = []

        def generate(self, prompt):
            self.calls.append((self.provider, self.model_name, prompt))
            text = self._responses.pop(0)

            class Resp:
                def __init__(self, text, provider, model):
                    self.text = text
                    self.provider = provider
                    self.model = model

            return Resp(text, self.provider, self.model_name)

        def clone_with_model(self, *, provider=None, model_name=None):
            return FakeLLM(
                provider=provider or self.provider,
                model_name=model_name or self.model_name,
                responses=['{"scores":[{"candidateId":"cand-1","rationale":"ok","criteria":[]}]}'],
            )

    llm = FakeLLM(
        responses=[
            '{"scores":[{"candidateId":"cand-1" "rationale":"bad"}]}',
            '{"scores":[{"candidateId":"cand-1" "rationale":"still bad"}]}',
        ]
    )

    parsed = _generate_json_with_retries(
        llm=llm,
        prompt="score this",
        parser=_parse_json_object,
        operation_name="semantic scoring",
    )

    assert parsed["scores"][0]["candidateId"] == "cand-1"
    assert llm.calls[0][1] == "llama-3.1-8b-instant"


def test_generate_semantic_scores_with_retries_returns_empty_mapping_after_invalid_json():
    class FakeLLM:
        provider = "groq"
        model_name = "mixtral-8x7b"

        def __init__(self):
            self.calls = []

        def generate(self, prompt):
            self.calls.append(prompt)

            class Resp:
                text = "not-json"

            return Resp()

    llm = FakeLLM()

    parsed = _generate_semantic_scores_with_retries(llm=llm, prompt="score this")

    assert parsed == {}
    assert len(llm.calls) == 2


def test_scoring_llm_provider_uses_higher_token_budget(monkeypatch):
    captured = {}

    class FakeProvider:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(score_candidate, "LLMProvider", FakeProvider)
    monkeypatch.setattr(score_candidate.settings, "LLM_MAX_TOKENS", 1024)

    score_candidate._scoring_llm_provider()

    assert captured["max_tokens"] == 4096


def test_clone_with_model_preserves_max_tokens_for_fallback_model(monkeypatch):
    monkeypatch.setattr("src.services.llm_service.settings.LLM_MAX_TOKENS", 1024)

    llm = LLMProvider(provider="ollama", model_name="llama3.1:8b", max_tokens=4096)
    cloned = llm.clone_with_model(provider="ollama", model_name="fallback")

    assert cloned.model_name == "fallback"
    assert cloned._adapter.max_tokens == 4096
