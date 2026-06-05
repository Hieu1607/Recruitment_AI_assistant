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
    assert "Scores must be numbers from 0 to 100" in prompt
    assert "Do not return binary 0/1 scores or probabilities" in prompt


def test_jd_rubric_prompt_restricts_measurable_fields_to_supported_candidateprofile_fields():
    prompt = BuildPrompts().build_jd_rubric_extraction_prompt(
        job_description_text="Need Python, Docker, and 5 years of experience.",
        section_weights={"skills": 50, "experience": 50},
    )

    assert "experience_years" in prompt
    assert "educated" in prompt
    assert "ever_studied_abroad" in prompt
    assert "Do not create custom measurable keys like python_skill, docker_skill, backend_experience, or cloud_platforms_skill." in prompt
    assert "Skills and technologies such as Python, TensorFlow, Docker, AWS, or cloud platforms must stay semantic" in prompt


def test_jd_rubric_prompt_does_not_seed_a_concrete_years_requirement_example():
    prompt = BuildPrompts().build_jd_rubric_extraction_prompt(
        job_description_text="Need Python, Docker, and AI model deployment.",
        section_weights={"skills": 100},
    )

    assert "5+ years of backend experience" not in prompt
    assert "Do not infer years of experience from seniority words such as Senior, Lead, or Principal." in prompt


def test_router_prompt_guides_name_lookup_count_and_comparison_queries():
    prompt = BuildPrompts().build_router_prompt("How many people named Hieu?")

    assert "Questions that mention explicit candidate names should use DSL with full_name." in prompt
    assert "If the user asks to compare, rank, or evaluate specifically named candidates, use both DSL and LLM when possible." in prompt


def test_answer_prompt_uses_question_language_and_adds_follow_up_guidance():
    prompt = BuildPrompts().build_answer_prompt(
        "Ứng viên nào đã từng học Đại Học Bách Khoa Hà Nội?",
        [{"id": "cand-1", "full_name": "Taylor"}],
    )

    assert "Write the answer in the SAME language as the question" in prompt
    assert "If the data is empty or no candidates match, reply with a warm, helpful no-match message" in prompt
    assert "End with 1 or 2 short follow-up suggestions" in prompt
    assert "Bạn có muốn biết thêm về ứng viên" in prompt
    assert "Bạn có muốn tìm ứng viên thỏa mãn" in prompt


def test_router_prompt_requires_friendly_same_language_refusal():
    prompt = BuildPrompts().build_router_prompt("What is the weather today?")

    assert "If false: set refusal_message to a short, warm, friendly reply in the SAME language as the question" in prompt
    assert "Offer 1 short follow-up suggestion that redirects the user back to recruitment help" in prompt


def test_chat_prompts_include_current_job_context_when_available():
    job_context = {
        "job_title": "Senior AI Engineer",
        "job_description_title": "AI Platform",
        "job_description_text": "Build LLM features and evaluate candidate fit.",
        "job_hidden_text": "Prefer hands-on production AI deployment experience.",
    }

    router_prompt = BuildPrompts().build_router_prompt(
        "Compare two candidates for this job",
        job_context=job_context,
    )
    llm_prompt = BuildPrompts().build_llm_query_prompt(
        "Who is a better fit for this job?",
        [{"id": "cand-1", "full_name": "Taylor"}],
        job_context=job_context,
    )
    answer_prompt = BuildPrompts().build_answer_prompt(
        "Who is a better fit for this job?",
        [{"id": "cand-1", "full_name": "Taylor"}],
        job_context=job_context,
    )

    for prompt in (router_prompt, llm_prompt, answer_prompt):
        assert "Current job context" in prompt
        assert "Senior AI Engineer" in prompt
        assert "Build LLM features and evaluate candidate fit." in prompt
        assert "Prefer hands-on production AI deployment experience." in prompt
        assert "Public job description" in prompt
        assert "Special recruiter-only requirements" in prompt
