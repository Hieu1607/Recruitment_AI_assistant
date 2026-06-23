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
    assert "Do not output placeholders such as N/A" in prompt
    assert "When a section has multiple projects, roles, schools, certifications, publications, achievements, or language groups, create one structured_profile entry per item." in prompt


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
    assert "Do not output placeholders such as N/A" in prompt
    assert "When a section has multiple projects, roles, schools, certifications, publications, achievements, or language groups, create one structured_profile entry per item." in prompt


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
    assert "graduation_status" in prompt
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

    assert '"response_intent": "inventory_list" | "candidate_match" | "attribute_lookup"' in prompt
    assert "Questions that mention explicit candidate names should use DSL with full_name." in prompt
    assert "If the user asks to compare, rank, or evaluate specifically named candidates, use both DSL and LLM when possible." in prompt
    assert "Use response_intent = inventory_list when the user asks to list, enumerate, show, or count the candidates currently in scope" in prompt
    assert "Use DSL for: full_name, phone, email, location_normalized, graduation_status, ever_studied_abroad, experience_years." in prompt
    assert "Use LLM for: contact, current_job_title, major, cpa" in prompt
    assert "Do not rely on graduation_status alone for broader concepts such as not yet graduating" in prompt
    assert "final-year" in prompt


def test_answer_prompt_uses_question_language_and_adds_follow_up_guidance():
    prompt = BuildPrompts().build_answer_prompt(
        "Ứng viên nào đã từng học Đại Học Bách Khoa Hà Nội?",
        [{"id": "cand-1", "full_name": "Taylor"}],
    )

    assert "Write the answer in the SAME language as the question" in prompt
    assert "If the data is empty or no candidates match, reply with a warm, helpful no-match message" in prompt
    assert "End with 1 or 2 short follow-up suggestions" in prompt
    assert "Make the follow-up suggestions dynamic and grounded in the answer" in prompt
    assert "Do not reuse the same fixed follow-up wording" in prompt
    assert "best-matching candidate" in prompt
    assert "closest matching candidates" in prompt


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
        assert "Current time (UTC):" in prompt
        assert "Current job context" in prompt
        assert "Senior AI Engineer" in prompt
        assert "Build LLM features and evaluate candidate fit." in prompt
        assert "Prefer hands-on production AI deployment experience." in prompt
        assert "Public job description" in prompt
        assert "Special recruiter-only requirements" in prompt


def test_dsl_prompt_includes_current_time_context():
    prompt = BuildPrompts().build_dsl_query_prompt("Ai có 3 năm kinh nghiệm Python?")

    assert "Current time (UTC):" in prompt
    assert "must be handled by the semantic LLM path instead" in prompt
    assert "skills_text" in prompt
    assert "experience_text" in prompt


def test_chat_semantic_map_prompt_requests_json_schema_only():
    prompt = BuildPrompts().build_chat_semantic_map_prompt(
        "Who knows Python?",
        [{"id": "cand-1", "full_name": "Taylor", "skills_text": "Python"}],
    )

    assert "Return JSON only" in prompt
    assert "qualifiedCandidates" in prompt
    assert '"id": "uuid"' in prompt
    assert '"name": "string"' in prompt
    assert '"score": 0.0' in prompt
    assert '"reason": "short string"' in prompt


def test_chat_reduce_prompt_uses_map_summaries_not_full_profiles():
    prompt = BuildPrompts().build_chat_reduce_prompt(
        "Who knows Python?",
        [
            {
                "qualifiedCandidates": [
                    {
                        "id": "cand-1",
                        "name": "Taylor",
                        "score": 0.8,
                        "reason": "Python",
                    }
                ],
                "batchQualifiedCount": 1,
            }
        ],
    )

    assert "map summaries" in prompt.lower()
    assert "rankedCandidates" in prompt
    assert "skills_text" not in prompt
    assert "experience_text" not in prompt


def test_compact_answer_prompt_uses_only_identity_fields():
    prompt = BuildPrompts().build_compact_answer_prompt(
        "Who knows Python?",
        [{"id": "cand-1", "full_name": "Taylor", "skills_text": "Python"}],
        total_count=12,
        omitted_count=11,
    )

    assert '"id": "cand-1"' in prompt
    assert '"full_name": "Taylor"' in prompt
    assert "skills_text" not in prompt
    assert "experience_text" not in prompt
    assert "12" in prompt
    assert "11" in prompt
