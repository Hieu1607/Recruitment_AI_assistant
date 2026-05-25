import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class BuildPrompts:
    """Build prompt strings for CV parsing, scoring, and CV section retrieval."""

    DEFAULT_SECTION_WEIGHTS: Dict[str, float] = {
        "skills": 0.35,
        "experience": 0.35,
        "projects": 0.15,
        "education": 0.10,
        "summary": 0.05,
    }

    def __init__(
        self, prompt_dir: Optional[Path] = None, max_prompt_chars: int = 24000
    ):
        self.prompt_dir = prompt_dir or Path(__file__).parent
        self.max_prompt_chars = max_prompt_chars

    def _clip_text(self, text: str, max_chars: Optional[int] = None) -> str:
        limit = max_chars or self.max_prompt_chars
        cleaned = (text or "").strip()
        if len(cleaned) <= limit:
            return cleaned
        head = cleaned[: int(limit * 0.7)]
        tail = cleaned[-int(limit * 0.3) :]
        return f"{head}\n\n...[TRUNCATED]...\n\n{tail}"

    def build_cv_parsing_prompt(self, cv_text: str) -> str:
        clipped = self._clip_text(cv_text)
        return f"""
		Analyze the CV text and return ONLY one valid JSON object (no markdown, no explanation).

		Required schema:
		{{
		"name": string|null,
		"phone": string|null,
		"email": string|null,
		"location": string|null,
		"contact": string|null,
		"current_job_title": string|null,
		"educated": boolean,
		"ever_studied_abroad": boolean,
		"major": string|null,
		"cpa": string|null,
		"education": string|null,
		"experience": string|null,
		"experience_years": number|null,
		"skills": string|null,
		"languages": string|null,
		"projects": string|null,
		"summary": string|null,
		"achievements": string|null,
		"publications": string|null,
		"certifications": string|null,
		"references": string|null,
		"other": string|null
		}}

		Rules:
		- Use null when unknown.
		- Preserve as much source detail as possible in the correct field.
		- Do not summarize, shorten, paraphrase, or normalize away specifics.
		- Keep bullet points, lists, metrics, technologies, dates, organizations, titles, and outcomes whenever present.
		- Classify every relevant piece of CV content into the most appropriate field from the schema.
		- If text does not clearly fit an earlier field, put it in other instead of dropping it.
		- summary should capture the candidate's own profile/objective/overview statement from the CV, not a new AI-written summary.
		- For "projects", include full project entries: project name, organization, link, date range, and all project bullet details as one multi-line string.
		- For "experience", include full role entries: title, company, location, dates, and all role bullet details as one multi-line string.
		- For "education", include full education entries: degree, institution, location, GPA, coursework, dates, and related bullet details as one multi-line string.
		- For "skills", preserve grouped skill categories and all listed tools/technologies instead of flattening to a short summary.
		- experience_years must be numeric (e.g., 3 or 4.5) or null.

		CV text:
		{clipped}
		""".strip()

    def build_cv_vision_prompt(self) -> str:
        return """Analyze the CV image(s) and return ONLY one valid JSON object (no markdown, no explanation).

Required schema:
{
"name": string|null,
"phone": string|null,
"email": string|null,
"location": string|null,
"contact": string|null,
"current_job_title": string|null,
"educated": boolean,
"ever_studied_abroad": boolean,
"major": string|null,
"cpa": string|null,
"education": string|null,
"experience": string|null,
"experience_years": number|null,
"skills": string|null,
"languages": string|null,
"projects": string|null,
"summary": string|null,
"achievements": string|null,
"publications": string|null,
"certifications": string|null,
"references": string|null,
"other": string|null
}

Rules:
- Use null when unknown.
- Preserve as much source detail as possible in the correct field.
- Do not summarize, shorten, paraphrase, or normalize away specifics.
- Keep bullet points, lists, metrics, technologies, dates, organizations, titles, and outcomes whenever present.
- Classify every relevant piece of CV content into the most appropriate field from the schema.
- If text does not clearly fit an earlier field, put it in other instead of dropping it.
- summary should capture the candidate's own profile/objective/overview statement from the CV, not a new AI-written summary.
- For "projects", include full project entries: project name, organization, link, date range, and all project bullet details as one multi-line string.
- For "experience", include full role entries: title, company, location, dates, and all role bullet details as one multi-line string.
- For "education", include full education entries: degree, institution, location, GPA, coursework, dates, and related bullet details as one multi-line string.
- For "skills", preserve grouped skill categories and all listed tools/technologies instead of flattening to a short summary.
- experience_years must be numeric (e.g., 3 or 4.5) or null.
- The CV may be in Vietnamese — extract text exactly as written.""".strip()

    def build_batch_scoring_prompt(
        self,
        *,
        job_description_text: str,
        candidates: Iterable[Dict[str, Any]],
        section_weights: Optional[Dict[str, float]] = None,
    ) -> str:
        if section_weights is not None:
            # Explicit weights provided: unmentioned sections default to 0
            weights: Dict[str, float] = {
                str(k): max(0.0, float(v))
                for k, v in section_weights.items()
                if v is not None
            }
        else:
            # No weights provided: fall back to class defaults
            weights = dict(self.DEFAULT_SECTION_WEIGHTS)

        total = sum(weights.values())
        if total <= 0:
            raise ValueError("section_weights total must be > 0")

        normalized_weights = {k: round(v / total, 4) for k, v in weights.items()}

        payload = {
            "jobDescription": self._clip_text(job_description_text, max_chars=12000),
            "sectionWeights": normalized_weights,
            "candidates": [
                {
                    "candidateId": str(
                        candidate.get("id") or candidate.get("candidateId") or ""
                    ),
                    "fullName": candidate.get("full_name") or candidate.get("fullName"),
                    "currentJobTitle": candidate.get("current_job_title")
                    or candidate.get("currentJobTitle"),
                    "education": candidate.get("education_text")
                    or candidate.get("education"),
                    "experience": candidate.get("experience_text")
                    or candidate.get("experience"),
                    "skills": candidate.get("skills_text") or candidate.get("skills"),
                    "summary": candidate.get("summary_text")
                    or candidate.get("summary"),
                }
                for candidate in candidates
            ],
            "responseFormat": {
                "scores": [
                    {
                        "candidateId": "uuid",
                        "totalScore": 0,
                        "passedThreshold": False,
                        "rationale": "string",
                        "componentScores": [
                            {
                                "criterionKey": "skills",
                                "weight": 0.4,
                                "score": 80,
                                "weightedScore": 32,
                                "evidenceSummary": "string",
                            }
                        ],
                    }
                ]
            },
        }

        return (
            "You are an objective recruitment scoring system. "
            "Use sectionWeights when calculating scores. "
            "Return valid JSON only with the shape shown in responseFormat.\n\n"
            f"{json.dumps(payload, ensure_ascii=True)}"
        )

    def build_cv_section_match_prompt(
        self,
        *,
        question: str,
        cv_section_items: Iterable[Dict[str, Any]],
        allowed_sections: Optional[List[str]] = None,
        max_items: int = 200,
    ) -> str:
        """Build prompt to find matching CVs from section-level CV payloads.

        Each item should contain:
        - cvId (or id)
        - cvName (or name/fullName)
        - sections: dict or list of section snippets
        """
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(cv_section_items):
            if idx >= max_items:
                break
            cv_id = item.get("cvId") or item.get("id") or ""
            cv_name = (
                item.get("cvName") or item.get("name") or item.get("fullName") or ""
            )
            sections = item.get("sections") or {}

            if isinstance(sections, list):
                sections_text = "\n".join(str(s) for s in sections)
                sections = {"raw": self._clip_text(sections_text, max_chars=3000)}
            elif isinstance(sections, dict):
                clipped_sections: Dict[str, str] = {}
                for key, value in sections.items():
                    if allowed_sections and key not in allowed_sections:
                        continue
                    clipped_sections[str(key)] = self._clip_text(
                        str(value), max_chars=1200
                    )
                sections = clipped_sections
            else:
                sections = {"raw": self._clip_text(str(sections), max_chars=2000)}

            normalized.append(
                {
                    "cvId": str(cv_id),
                    "cvName": str(cv_name),
                    "sections": sections,
                }
            )

        payload = {
            "question": question.strip(),
            "candidates": normalized,
            "responseFormat": {
                "matches": [
                    {
                        "cvId": "string",
                        "cvName": "string",
                        "reason": "short evidence-based explanation",
                    }
                ]
            },
        }

        instructions = (
            "You are a CV retrieval assistant. Use only candidate section data to answer the question. "
            "Return JSON only. Include only truly relevant CVs. "
            'Do not invent IDs or names. If none match, return {"matches": []}.'
        )

        return f"{instructions}\n\n{json.dumps(payload, ensure_ascii=True)}"

    def build_dsl_query_prompt(self, question: str) -> str:
        _template = """You are a recruitment data query assistant. Translate the user's question into a JSON object that can be used to query the candidate database with the following schema:
		    full_name: String, name of the candidate
			phone: String, phone number of the candidate
			email: String, email address of the candidate
			location_normalized: String, normalized location of the candidate
			contact: String, contact information of the candidate
			current_job_title: String, current job title of the candidate

			educated: Boolean, whether the candidate is educated
			ever_studied_abroad: Boolean, whether the candidate has ever studied abroad
			major: String, major of the candidate
			cpa: String, CPA status of the candidate

			education_text: String, education details of the candidate
			experience_text: String, experience details of the candidate
			experience_years: Number, years of experience of the candidate
			skills_text: String, skills of the candidate
			languages_text: String, languages spoken by the candidate
			projects_text: String, projects worked on by the candidate
			summary_text: String, summary of the candidate
			achievements_text: String, achievements of the candidate
			publications_text: String, publications of the candidate
			certifications_text: String, certifications of the candidate
			references_text: String, references of the candidate
			other_text: String, other information about the candidate
		---

### Output format:

Return a JSON object with the following structure:

{
  "filters": {
    "<field>": {
      "operator": "<eq|gte|lte|contains>",
      "value": <value>
    }
  },
  "must": [
    { "field": "<field>", "contains": "<keyword>" }
  ],
  "should": [
    { "field": "<field>", "contains": "<keyword>" }
  ]
}

---

### Rules:

1. Only include fields explicitly mentioned in the question.

2. Map conditions:
- "trên", "hơn", "ít nhất" → gte
- "dưới", "ít hơn" → lte
- exact value → eq
- text search → contains

3. Structured fields:
- experience_years → numeric filters
- location_normalized → exact match
- current_job_title → contains

4. Text fields:
- skills_text, experience_text → use "contains"

5. Logical conditions:
- "và" → MUST (AND)
- "hoặc" → SHOULD (OR)

6. Extract keywords clearly (e.g., React, Python, Golang)

7. Return valid JSON only. No explanation.

---
		Question: """
        return _template + question + "\n"

    def build_llm_query_prompt(self, question: str, candidate_data: list) -> str:

        return f"""You are a recruitment assistant. Answer the user's question based on the candidate data.
		Return a Json list object with the following schema:
		{{
		  "total_qualified_candidates": number, // the total number of candidates that meet the criteria
		  "qualified_candidates": {{candidate_id: string, reason: string}} // a dictionary of candidate id and reason for each candidate that meet the criteria
		}}

		Candidate data: {json.dumps(candidate_data, ensure_ascii=True)}
		Provide a concise and relevant answer to the user's question using only the information available in the candidate data. Do not make assumptions or include information that is not present in the candidate data. 
		If the question cannot be answered with the available data, respond with empty dictionary. Question: {question}
		
"""

    def build_answer_prompt(self, question: str, candidates: list) -> str:
        """Build a RAG prompt to generate a natural-language answer from candidate data."""
        candidate_json = json.dumps(candidates, ensure_ascii=False, indent=2)
        return f"""You are a recruitment assistant. Answer the user's question based solely on the candidate data provided below with simple explaination. Answer nicely by Vietnamese.

Rules:
- Be concise and specific. Reference candidates by name when relevant.
- Do not invent or assume information not present in the data.
- If the data is empty or no candidates match, clearly state that no matching candidates were found.

Candidate data:
{candidate_json}

Question: {question}

Answer:"""

    def build_router_prompt(self, question: str) -> str:
        return f"""You are a recruitment assistant router. Given the user's question, do two things in one response:

1. Decide if the question is related to recruitment, candidates, resumes, hiring, or HR topics.
2. If it is related, decide how to route it (structured DSL query, LLM analysis, or both).

Return ONLY a valid JSON object with this shape:
{{
  "is_recruitment_related": true | false,
  "refusal_message": string | null,
  "relevant_fields": [string],
  "dsl_question_query": string | null,
  "llm_question_query": string | null,
  "dsl_relevant_fields": [string],
  "llm_relevant_fields": [string],
  "reasoning": string
}}

Rules for is_recruitment_related:
- true: question is about candidates, CVs, resumes, job titles, skills, experience, education, hiring, HR, interviews, shortlisting.
- false: question is about unrelated topics (weather, cooking, math, general coding, etc.).
- If false: set refusal_message to a short, friendly reply in the SAME language as the question explaining you only assist with recruitment topics. Set all other fields to null or [].
- If true: set refusal_message to null and fill in the routing fields below.

Routing rules (only when is_recruitment_related is true):
- dsl_question_query: rephrase the question for structured DB filtering. Set null if not applicable.
- llm_question_query: rephrase the question for semantic LLM analysis. Set null if not applicable.
- dsl_relevant_fields / llm_relevant_fields: fields from the schema below relevant to each path.
- Use DSL for: full_name, phone, email, location_normalized, contact, current_job_title, educated, ever_studied_abroad, major, cpa, experience_years.
- Use LLM for: education_text, experience_text, skills_text, languages_text, projects_text, summary_text, achievements_text, publications_text, certifications_text, references_text, other_text.
- Both paths can apply to the same question.

Database schema:
  full_name, phone, email, location_normalized, contact, current_job_title,
  educated (Boolean), ever_studied_abroad (Boolean), major, cpa,
  education_text, experience_text, experience_years (Number), skills_text,
  languages_text, projects_text, summary_text, achievements_text,
  publications_text, certifications_text, references_text, other_text

Question: {question}"""

    def build_interview_questions_prompt(
        self,
        *,
        candidate_data: Dict[str, Any],
        job_description_text: str,
    ) -> str:
        """Build a prompt to generate structured interview questions for a candidate+JD pair."""
        clipped_jd = self._clip_text(job_description_text, max_chars=8000)
        clipped_candidate = {
            k: self._clip_text(str(v or ""), max_chars=2000)
            for k, v in candidate_data.items()
            if v
        }

        payload = {
            "jobDescription": clipped_jd,
            "candidate": clipped_candidate,
            "responseFormat": {
                "categories": [
                    {
                        "name": "string (e.g. Technical, Behavioral, Situational)",
                        "questions": [
                            {
                                "id": "unique string e.g. q1",
                                "text": "question text",
                                "difficulty": "easy | medium | hard",
                            }
                        ],
                    }
                ]
            },
        }

        return (
            "You are a recruitment assistant generating tailored interview questions. "
            "Based on the candidate profile and job description, generate 2-3 categories "
            "with 3-5 questions each. Tailor questions to the candidate's background. "
            "Return valid JSON only matching the responseFormat shape exactly.\n\n"
            f"{json.dumps(payload, ensure_ascii=True)}"
        )


build_prompts = BuildPrompts()
