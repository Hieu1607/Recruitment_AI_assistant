import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prompts.build_prompts import BuildPrompts  # noqa: E402
from src.services.score_candidate import _build_scoring_job_description_text  # noqa: E402


def test_cv_parsing_prompt_requires_exhaustive_extraction():
    prompt = BuildPrompts().build_cv_parsing_prompt("Example CV text")

    assert "Keep extracted text concise" not in prompt
    assert "Preserve as much source detail as possible" in prompt
    assert "Do not summarize, shorten, paraphrase, or normalize away specifics." in prompt
    assert "If text does not clearly fit an earlier field, put it in other instead of dropping it." in prompt
    assert "Keep bullet points, lists, metrics, technologies, dates, organizations, titles, and outcomes whenever present." in prompt
    assert 'For "projects", include full project entries' in prompt
    assert 'For "experience", include full role entries' in prompt
    assert '"experience" is only for actual work history' in prompt
    assert 'Put non-employment build work into "projects"' in prompt
    assert 'For "education", include full education entries' in prompt
    assert 'For "skills", preserve grouped skill categories' in prompt
    assert '"structured_profile"' in prompt
    assert "Preserve links wherever they appear" in prompt


def test_cv_vision_prompt_requires_exhaustive_extraction():
    prompt = BuildPrompts().build_cv_vision_prompt()

    assert "Keep extracted text concise" not in prompt
    assert "Preserve as much source detail as possible" in prompt
    assert "Do not summarize, shorten, paraphrase, or normalize away specifics." in prompt
    assert "If text does not clearly fit an earlier field, put it in other instead of dropping it." in prompt
    assert "The CV may be in Vietnamese — extract text exactly as written." in prompt
    assert 'For "projects", include full project entries' in prompt
    assert 'For "experience", include full role entries' in prompt
    assert '"experience" is only for actual work history' in prompt
    assert '"structured_profile"' in prompt


def test_scoring_job_description_text_includes_recruiter_only_hidden_information():
    text = _build_scoring_job_description_text(
        public_job_description="Public requirements: Python and FastAPI.",
        hidden_text="Recruiter-only: prefer fintech domain experience.",
    )

    assert "Public job description:" in text
    assert "Public requirements: Python and FastAPI." in text
    assert "Recruiter-only hidden information:" in text
    assert "Recruiter-only: prefer fintech domain experience." in text


def test_batch_scoring_prompt_includes_all_candidate_sections():
    prompt = BuildPrompts().build_batch_scoring_prompt(
        job_description_text="Need strong communication and project depth.",
        section_weights={"projects": 60, "languages": 20, "certifications": 20},
        candidates=[
            {
                "id": "cand-1",
                "full_name": "Taylor",
                "projects_text": "Project Atlas",
                "languages_text": "IELTS 7.5",
                "certifications_text": "AWS SAA",
                "other_text": "Hackathon finalist",
            }
        ],
    )

    assert '"projects": "Project Atlas"' in prompt
    assert '"languages": "IELTS 7.5"' in prompt
    assert '"certifications": "AWS SAA"' in prompt
    assert '"other": "Hackathon finalist"' in prompt


def test_locked_rubric_semantic_prompt_includes_rubric_and_evidence_requirements():
    prompt = BuildPrompts().build_locked_rubric_semantic_scoring_prompt(
        candidates=[
            {
                "id": "cand-1",
                "full_name": "Taylor",
                "skills_text": "Python, FastAPI",
                "projects_text": "Led backend rewrite",
            }
        ],
        rubric={
            "criteria": [
                {
                    "key": "skills.python",
                    "section": "skills",
                    "requirementText": "Strong Python proficiency",
                    "type": "semantic",
                }
            ]
        },
    )

    assert "locked rubric" in prompt.lower()
    assert '"criteria"' in prompt
    assert '"projects": "Led backend rewrite"' in prompt
    assert "evidence" in prompt.lower()
