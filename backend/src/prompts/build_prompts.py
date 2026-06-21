import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.core.config import settings


class BuildPrompts:
    """Build prompt strings for CV parsing, scoring, and CV section retrieval."""

    SUPPORTED_SCORING_SECTIONS: List[str] = [
        "skills",
        "experience",
        "education",
        "projects",
        "summary",
        "languages",
        "achievements",
        "certifications",
        "publications",
        "other",
    ]

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

    def _ui_language(self) -> str:
        return "en" if str(settings.APP_UI_LANGUAGE or "").strip().lower().startswith("en") else "vi"

    def _language_name(self) -> str:
        return "English" if self._ui_language() == "en" else "Vietnamese"

    def _current_time_block(self) -> str:
        now_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return f"Current time (UTC): {now_utc}\n\n"

    def _format_job_context(self, job_context: Optional[Dict[str, Any]]) -> str:
        if not job_context:
            return ""

        job_title = job_context.get("job_title")
        jd_title = job_context.get("job_description_title")
        public_jd = job_context.get("job_description_text")
        hidden_requirements = job_context.get("job_hidden_text")

        lines = ["Current job context:"]
        if job_title:
            lines.append(f"- Job title: {job_title}")
        if jd_title:
            lines.append(f"- Job description title: {jd_title}")
        if public_jd:
            lines.append("- Public job description:")
            lines.append(str(public_jd))
        if hidden_requirements:
            lines.append("- Special recruiter-only requirements:")
            lines.append(str(hidden_requirements))
            lines.append(
                "Treat these as confidential hiring preferences distinct from the public job description."
            )
        return "\n".join(lines) + "\n\n"

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
		"graduation_status": "graduated"|"final_year"|"studying"|"unknown",
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
		"other": string|null,
		"structured_profile": {{
		  "summary": {{
		    "text": string|null,
		    "links": [{{"url": string, "label": string|null}}]
		  }}|null,
		  "experience": {{
		    "entries": [{{
		      "title": string|null,
		      "subtitle": string|null,
		      "role": string|null,
		      "location": string|null,
		      "dateRange": string|null,
		      "description": string|null,
		      "bullets": [string],
		      "links": [{{"url": string, "label": string|null}}],
		      "metadata": [string]
		    }}],
		    "rawText": string|null
		  }}|null,
		  "education": same shape as experience|null,
		  "projects": same shape as experience|null,
		  "skills": same shape as experience|null,
		  "languages": same shape as experience|null,
		  "achievements": same shape as experience|null,
		  "publications": same shape as experience|null,
		  "certifications": same shape as experience|null,
		  "references": same shape as experience|null,
		  "other": same shape as experience|null
		}}|null
		}}

		Rules:
		- Use null when unknown.
		- Do not output placeholders such as N/A, NA, none, null, "-", or "not applicable"; use null for scalar fields and [] for list fields.
		- Preserve as much source detail as possible in the correct field.
		- Do not summarize, shorten, paraphrase, or normalize away specifics.
		- Keep bullet points, lists, metrics, technologies, dates, organizations, titles, and outcomes whenever present.
		- Classify every relevant piece of CV content into the most appropriate field from the schema.
		- If text does not clearly fit an earlier field, put it in other instead of dropping it.
		- summary should capture the candidate's own profile/objective/overview statement from the CV, not a new AI-written summary.
		- When a section has multiple projects, roles, schools, certifications, publications, achievements, or language groups, create one structured_profile entry per item.
		- In structured_profile entries, only fill title, subtitle, role, location, dateRange, description, bullets, links, and metadata when that value is explicitly present or inferable from the CV text.
		- For structured_profile.projects, create one entry per project whenever possible; put the project name in title, repository or portfolio URLs in links, dates in dateRange, technologies or organization in metadata, and project details in description or bullets.
		- For "projects", include full project entries: project name, organization, link, date range, and all project bullet details as one multi-line string.
		- For "experience", include full role entries: title, company, location, dates, and all role bullet details as one multi-line string.
		- "experience" is only for actual work history such as employment, internships, assistantships, apprenticeships, consulting, or freelance roles.
		- Do not place personal projects, academic projects, research builds, capstones, portfolio pieces, or product demos into "experience" unless the CV clearly presents them as paid or formal work roles in a work-experience section.
		- Put non-employment build work into "projects", keeping all attached bullets, dates, organizations, and links there.
		- For "education", include full education entries: degree, institution, location, GPA, coursework, dates, and related bullet details as one multi-line string.
		- For "skills", preserve grouped skill categories and all listed tools/technologies instead of flattening to a short summary.
		- Preserve links wherever they appear, both in the raw text fields and in structured_profile.links.
		- structured_profile must organize the same CV content into section objects with generic entries that work for any profession, not only software resumes.
		- experience_years must be numeric (e.g., 3 or 4.5) or null.
		- graduation_status must be one of: graduated, final_year, studying, unknown.
		- Use final_year when the CV says the candidate is in the final year or has an expected graduation date but has not graduated yet.
		- Use studying when the candidate is still studying but not clearly in the final year.
		- Use graduated only when the CV clearly indicates the degree was completed.

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
"graduation_status": "graduated"|"final_year"|"studying"|"unknown",
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
"other": string|null,
"structured_profile": {
  "summary": {
    "text": string|null,
    "links": [{"url": string, "label": string|null}]
  }|null,
  "experience": {
    "entries": [{
      "title": string|null,
      "subtitle": string|null,
      "role": string|null,
      "location": string|null,
      "dateRange": string|null,
      "description": string|null,
      "bullets": [string],
      "links": [{"url": string, "label": string|null}],
      "metadata": [string]
    }],
    "rawText": string|null
  }|null,
  "education": "same shape as experience"|null,
  "projects": "same shape as experience"|null,
  "skills": "same shape as experience"|null,
  "languages": "same shape as experience"|null,
  "achievements": "same shape as experience"|null,
  "publications": "same shape as experience"|null,
  "certifications": "same shape as experience"|null,
  "references": "same shape as experience"|null,
  "other": "same shape as experience"|null
}|null
}

Rules:
- Use null when unknown.
- Do not output placeholders such as N/A, NA, none, null, "-", or "not applicable"; use null for scalar fields and [] for list fields.
- Preserve as much source detail as possible in the correct field.
- Do not summarize, shorten, paraphrase, or normalize away specifics.
- Keep bullet points, lists, metrics, technologies, dates, organizations, titles, and outcomes whenever present.
- Classify every relevant piece of CV content into the most appropriate field from the schema.
- If text does not clearly fit an earlier field, put it in other instead of dropping it.
- summary should capture the candidate's own profile/objective/overview statement from the CV, not a new AI-written summary.
- When a section has multiple projects, roles, schools, certifications, publications, achievements, or language groups, create one structured_profile entry per item.
- In structured_profile entries, only fill title, subtitle, role, location, dateRange, description, bullets, links, and metadata when that value is explicitly present or inferable from the CV text.
- For structured_profile.projects, create one entry per project whenever possible; put the project name in title, repository or portfolio URLs in links, dates in dateRange, technologies or organization in metadata, and project details in description or bullets.
- For "projects", include full project entries: project name, organization, link, date range, and all project bullet details as one multi-line string.
- For "experience", include full role entries: title, company, location, dates, and all role bullet details as one multi-line string.
- "experience" is only for actual work history such as employment, internships, assistantships, apprenticeships, consulting, or freelance roles.
- Do not place personal projects, academic projects, research builds, capstones, portfolio pieces, or product demos into "experience" unless the CV clearly presents them as paid or formal work roles in a work-experience section.
- Put non-employment build work into "projects", keeping all attached bullets, dates, organizations, and links there.
- For "education", include full education entries: degree, institution, location, GPA, coursework, dates, and related bullet details as one multi-line string.
- For "skills", preserve grouped skill categories and all listed tools/technologies instead of flattening to a short summary.
- Preserve links wherever they appear, both in the raw text fields and in structured_profile.links.
- structured_profile must organize the same CV content into section objects with generic entries that work for any profession, not only software resumes.
- experience_years must be numeric (e.g., 3 or 4.5) or null.
- graduation_status must be one of: graduated, final_year, studying, unknown.
- Use final_year when the CV says the candidate is in the final year or has an expected graduation date but has not graduated yet.
- Use studying when the candidate is still studying but not clearly in the final year.
- Use graduated only when the CV clearly indicates the degree was completed.
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
            "candidates": [self._build_scoring_candidate_payload(candidate) for candidate in candidates],
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
            "Only score sections that are present in sectionWeights. "
            "Do not penalize candidates for sections not referenced by the job requirements. "
            f"Write rationale and evidenceSummary in {self._language_name()}. "
            "Return valid JSON only with the shape shown in responseFormat.\n\n"
            f"{json.dumps(payload, ensure_ascii=True)}"
        )

    def build_jd_rubric_extraction_prompt(
        self,
        *,
        job_description_text: str,
        section_weights: Optional[Dict[str, float]] = None,
    ) -> str:
        weights = section_weights or dict(self.DEFAULT_SECTION_WEIGHTS)
        payload = {
            "jobDescription": self._clip_text(job_description_text, max_chars=12000),
            "sectionWeights": weights,
            "supportedSections": self.SUPPORTED_SCORING_SECTIONS,
            "supportedMeasurableFields": {
                "experience_years": {
                    "type": "number",
                    "allowedOperators": [">=", ">", "<=", "<", "==", "="],
                },
                "graduation_status": {
                    "type": "string",
                    "allowedOperators": ["==", "="],
                    "allowedValues": ["graduated", "final_year", "studying", "unknown"],
                },
                "ever_studied_abroad": {
                    "type": "boolean",
                    "allowedOperators": ["==", "="],
                },
            },
            "criterionTypes": ["must_have", "semantic", "upper_bound"],
            "responseFormat": {
                "criteria": [
                    {
                        "key": "short_snake_case_key",
                        "section": "one_supported_section",
                        "requirementText": "exact explicit requirement copied or directly paraphrased from the job description",
                        "type": "must_have|semantic|upper_bound",
                        "measurable": {
                            "field": "one_supported_measurable_field_or_omit",
                            "operator": "allowed_operator_for_that_field",
                            "value": "number_or_boolean_from_explicit_requirement",
                        },
                    }
                ]
            },
        }
        return (
            "Extract a locked scoring rubric from the job description. "
            "Return JSON only, no markdown. "
            "Use only supportedSections. "
            "Drop empty or duplicate criteria. "
            "Extract only requirements explicitly present in the public job description or recruiter-only hidden information. "
            "Do not infer years of experience from seniority words such as Senior, Lead, or Principal. "
            "Do not add generic education, degree, or years-of-experience criteria unless the source text explicitly says so. "
            "Use measurable only when the requirement can be checked directly from supportedMeasurableFields. "
            "If a requirement cannot be expressed using supportedMeasurableFields, emit it as semantic instead. "
            "Skills and technologies such as Python, TensorFlow, Docker, AWS, or cloud platforms must stay semantic. "
            "Do not create custom measurable keys like python_skill, docker_skill, backend_experience, or cloud_platforms_skill. "
            "Use measurable checks only for supported CandidateProfile fields such as graduation_status, experience_years, or ever_studied_abroad. "
            "For bonus-style thresholds such as IELTS 7.5+ being a plus, emit type upper_bound instead of must_have. "
            "Do not invent sections outside supportedSections. "
            f"Write requirementText in {self._language_name()}.\n\n"
            f"{json.dumps(payload, ensure_ascii=True)}"
        )

    def build_locked_rubric_semantic_scoring_prompt(
        self,
        *,
        candidates: Iterable[Dict[str, Any]],
        rubric: Dict[str, Any],
    ) -> str:
        payload = {
            "rubric": rubric,
            "candidates": [self._build_scoring_candidate_payload(candidate) for candidate in candidates],
            "responseFormat": {
                "scores": [
                    {
                        "candidateId": "uuid",
                        "rationale": "string",
                        "criteria": [
                            {
                                "criterionKey": "skills.python",
                                "score": 85,
                                "evidenceSummary": "string",
                            }
                        ],
                    }
                ]
            },
        }
        return (
            "You are scoring candidates against a locked rubric that has already been approved by the backend. "
            "Score only the listed criteria. "
            "Do not add, remove, or reinterpret criteria. "
            "Use only evidence from the provided candidate sections. "
            "Scores must be numbers from 0 to 100, where 100 is a clear full match, 70 is a strong partial match, "
            "40 is weak or indirect evidence, and 0 is no evidence. "
            "Do not return binary 0/1 scores or probabilities. "
            f"Write rationale and evidenceSummary in {self._language_name()}. "
            "Return JSON only and include evidence for every scored criterion.\n\n"
            f"{json.dumps(payload, ensure_ascii=True)}"
        )

    def _build_scoring_candidate_payload(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "candidateId": str(candidate.get("id") or candidate.get("candidateId") or ""),
            "fullName": candidate.get("full_name") or candidate.get("fullName"),
            "currentJobTitle": candidate.get("current_job_title") or candidate.get("currentJobTitle"),
            "experienceYears": candidate.get("experience_years") or candidate.get("experienceYears"),
            "education": candidate.get("education_text") or candidate.get("education"),
            "experience": candidate.get("experience_text") or candidate.get("experience"),
            "skills": candidate.get("skills_text") or candidate.get("skills"),
            "projects": candidate.get("projects_text") or candidate.get("projects"),
            "summary": candidate.get("summary_text") or candidate.get("summary"),
            "languages": candidate.get("languages_text") or candidate.get("languages"),
            "achievements": candidate.get("achievements_text") or candidate.get("achievements"),
            "certifications": candidate.get("certifications_text") or candidate.get("certifications"),
            "publications": candidate.get("publications_text") or candidate.get("publications"),
            "other": candidate.get("other_text") or candidate.get("other"),
        }

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

			graduation_status: String, one of graduated, final_year, studying, unknown
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
- graduation_status → exact match

4. Text fields:
- skills_text, experience_text → use "contains"

5. Do not generate filters for current_job_title, major, cpa, or contact. These values often vary by language, abbreviations, formatting, employer suffixes, or free-text conventions, so matching them should be handled by the semantic LLM path instead.

6. Logical conditions:
- "và" → MUST (AND)
- "hoặc" → SHOULD (OR)

7. Extract keywords clearly (e.g., React, Python, Golang)

8. Return valid JSON only. No explanation.

--- 
		Question: """
        return self._current_time_block() + _template + question + "\n"

    def build_llm_query_prompt(
        self,
        question: str,
        candidate_data: list,
        job_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        job_context_block = self._format_job_context(job_context)
        return f"""You are a recruitment assistant. Answer the user's question based on the candidate data.
		Return a Json list object with the following schema:
		{{
		  "total_qualified_candidates": number, // the total number of candidates that meet the criteria
		  "qualified_candidates": {{candidate_id: string, reason: string}} // a dictionary of candidate id and reason for each candidate that meet the criteria
		}}

		{self._current_time_block()}{job_context_block}Candidate data: {json.dumps(candidate_data, ensure_ascii=True)}
		Provide a concise and relevant answer to the user's question using only the information available in the candidate data. Do not make assumptions or include information that is not present in the candidate data.
		When the question refers to "this job", "công việc này", or the current role, use the current job context above.
		Write every reason field in {self._language_name()}.
		If the question cannot be answered with the available data, respond with empty dictionary. Question: {question}
		
"""

    def build_chat_semantic_map_prompt(
        self,
        question: str,
        candidates: Iterable[Dict[str, Any]],
        job_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload = {
            "question": question,
            "jobContext": job_context or {},
            "candidates": list(candidates),
            "responseFormat": {
                "qualifiedCandidates": [
                    {
                        "id": "uuid",
                        "name": "string",
                        "score": 0.0,
                        "reason": "short string",
                    }
                ],
                "batchQualifiedCount": 0,
            },
        }
        job_context_block = self._format_job_context(job_context)
        return (
            "You are the map step in a recruitment chat map-reduce pipeline. "
            "Evaluate only this candidate batch against the user's question. "
            "Return JSON only, with no markdown and no explanation outside JSON. "
            "Include only candidates that are relevant matches. "
            "Use ids and names exactly as provided. "
            "Score each match from 0.0 to 1.0 and keep reason short and evidence-based. "
            f"Write reason in {self._language_name()}.\n\n"
            f"{self._current_time_block()}{job_context_block}"
            f"{json.dumps(payload, ensure_ascii=True)}"
        )

    def build_chat_reduce_prompt(
        self,
        question: str,
        map_results: Iterable[Dict[str, Any]],
        job_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        summaries: List[Dict[str, Any]] = []
        for result in map_results:
            qualified = []
            for candidate in result.get("qualifiedCandidates") or []:
                qualified.append(
                    {
                        "id": candidate.get("id"),
                        "name": candidate.get("name"),
                        "score": candidate.get("score"),
                        "reason": candidate.get("reason"),
                    }
                )
            summaries.append(
                {
                    "qualifiedCandidates": qualified,
                    "batchQualifiedCount": result.get(
                        "batchQualifiedCount", len(qualified)
                    ),
                }
            )

        payload = {
            "question": question,
            "jobContext": job_context or {},
            "mapSummaries": summaries,
            "responseFormat": {
                "totalQualified": 0,
                "rankedCandidates": [
                    {
                        "id": "uuid",
                        "name": "string",
                        "score": 0.0,
                        "reason": "short string",
                    }
                ],
            },
        }
        job_context_block = self._format_job_context(job_context)
        return (
            "You are the reduce step in a recruitment chat map-reduce pipeline. "
            "Use map summaries only; do not ask for or assume full candidate profiles. "
            "Deduplicate candidates by id, rank the strongest matches first, and keep reasons compact. "
            "Return JSON only, with no markdown and no explanation outside JSON. "
            f"Write reason in {self._language_name()}.\n\n"
            f"{self._current_time_block()}{job_context_block}"
            f"{json.dumps(payload, ensure_ascii=True)}"
        )

    def build_compact_answer_prompt(
        self,
        question: str,
        compact_candidates: Iterable[Dict[str, Any]],
        total_count: int,
        omitted_count: int,
        job_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        identities = [
            {
                "id": str(candidate.get("id") or candidate.get("candidateId") or ""),
                "full_name": candidate.get("full_name") or candidate.get("fullName"),
            }
            for candidate in compact_candidates
        ]
        payload = {
            "question": question,
            "totalQualifiedCandidates": int(total_count),
            "omittedCandidateCount": int(omitted_count),
            "candidates": identities,
        }
        job_context_block = self._format_job_context(job_context)
        return f"""You are a recruitment assistant. Answer using only candidate ids and full_name values from the compact candidate list.

Rules:
- Write the answer in the SAME language as the question.
- Explain that the result set is large and the displayed list is compact.
- Mention the total qualified candidate count and omitted candidate count when omittedCandidateCount is greater than 0.
- Do not invent profile details, skills, education, experience, or reasons that are not present in the compact list.
- Keep the answer concise and recruiter-focused.
- Offer one short follow-up suggestion to narrow or inspect candidates in more detail.

{self._current_time_block()}{job_context_block}Compact candidate payload:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Answer:"""

    def build_answer_prompt(
        self,
        question: str,
        candidates: list,
        job_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a RAG prompt to generate a natural-language answer from candidate data."""
        candidate_json = json.dumps(candidates, ensure_ascii=False, indent=2)
        job_context_block = self._format_job_context(job_context)
        return f"""You are a recruitment assistant. Answer the user's question based solely on the candidate data provided below.

Rules:
- Provide a detailed, comprehensive, and well-structured answer. Explain clearly why candidates match or do not match the criteria.
- Reference candidates by name when relevant. For each matching candidate, present key details such as their qualifications, current job title, years of experience, key skills, projects, and education where relevant to the query.
- Use bold formatting (e.g. **Candidate Name**) to highlight candidate names, key technologies, or major credentials.
- Use bullet points or numbered lists to organize candidate profiles clearly and make the response highly scannable.
- Do not invent or assume information not present in the data.
- NEVER expose internal database field names, schema details, or raw technical metadata in your answer. For example, do NOT write "graduation_status: graduated", "experience_years", "skills_text", "location_normalized", or any JSON key names from the candidate data. Instead, rephrase naturally — e.g. say "đã tốt nghiệp" or "đang học năm cuối" instead of raw status values, say "5 năm kinh nghiệm" instead of "experience_years: 5".
- Write the answer in the SAME language as the question, not the UI language.
- If the data is empty or no candidates match, reply with a warm, helpful no-match message in the SAME language as the question.
- For no-match replies, briefly suggest how the user could broaden or adjust the search.
- When the question refers to "this job", "công việc này", or the current role, use the current job context below.
- Keep the wording natural, friendly, and recruiter-focused.
- End with 1 or 2 short follow-up suggestions in the SAME language as the question.
- Make the follow-up suggestions dynamic and grounded in the answer, not generic or repetitive.
- Do not reuse the same fixed follow-up wording across responses.
- Base each follow-up on the most useful next step from the candidate data, such as:
  - learning more about the best-matching candidate,
  - reviewing the closest matching candidates who nearly meet the criteria,
  - comparing 2 promising candidates,
  - broadening or narrowing a filter,
  - focusing on a specific skill, experience level, education background, location, or language.
- Phrase follow-ups as natural conversational suggestions, not rigid templates.
- If there is a strong top match, one follow-up can invite the user to explore that candidate in more detail.
- If there are near matches or no exact matches, one follow-up can suggest reviewing the closest matching candidates or relaxing a constraint.
- Keep each follow-up short, specific, and directly relevant to the answer that was just given.
- Example styles in Vietnamese (do not copy verbatim; adapt them to the actual answer):
  - "Bạn có muốn mình phân tích kỹ hơn ứng viên phù hợp nhất cho vị trí này không?"
  - "Bạn có muốn xem những ứng viên gần đạt nhất nếu mình nới điều kiện một chút không?"
  - "Bạn có muốn mình so sánh nhanh 2 ứng viên nổi bật nhất theo yêu cầu này không?"

{self._current_time_block()}{job_context_block}Candidate data:
{candidate_json}

Question: {question}

Answer:"""

    def build_router_prompt(
        self,
        question: str,
        job_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        job_context_block = self._format_job_context(job_context)
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
- If false: set refusal_message to a short, warm, friendly reply in the SAME language as the question explaining you only assist with recruitment topics.
- Offer 1 short follow-up suggestion that redirects the user back to recruitment help, still in the SAME language as the question.
- Set all other fields to null or [] when is_recruitment_related is false.
- If true: set refusal_message to null and fill in the routing fields below.

Routing rules (only when is_recruitment_related is true):
- dsl_question_query: rephrase the question for structured DB filtering. Set null if not applicable.
- llm_question_query: rephrase the question for semantic LLM analysis. Set null if not applicable.
- dsl_relevant_fields / llm_relevant_fields: fields from the schema below relevant to each path.
- Use DSL for: full_name, phone, email, location_normalized, graduation_status, ever_studied_abroad, experience_years.
- Use LLM for: contact, current_job_title, major, cpa, education_text, experience_text, skills_text, languages_text, projects_text, summary_text, achievements_text, publications_text, certifications_text, references_text, other_text.
- Questions that mention explicit candidate names should use DSL with full_name.
- If the user asks to count candidates matching a name or other structured attribute, prefer DSL.
- If the user asks to compare, rank, or evaluate specifically named candidates, use both DSL and LLM when possible.
- Candidate role/title, major, CPA, and contact matching should prefer LLM routing because these values can be multilingual, abbreviated, decorated, or embedded in free text.
- Do not rely on graduation_status alone for broader concepts such as not yet graduating, still studying, final-year students, expected graduation, or "chưa tốt nghiệp".
- For those graduation-status semantics, prefer LLM routing with education_text and summary_text, because the relevant evidence may be described in free text and may span multiple statuses.
- Both paths can apply to the same question.

Database schema:
  full_name, phone, email, location_normalized, contact, current_job_title,
  graduation_status (String), ever_studied_abroad (Boolean), major, cpa,
  education_text, experience_text, experience_years (Number), skills_text,
  languages_text, projects_text, summary_text, achievements_text,
  publications_text, certifications_text, references_text, other_text

{self._current_time_block()}{job_context_block}If the question says "this job", "công việc này", or otherwise refers to the current role, use the current job context above to interpret it.

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
            f"Write category names and question text in {self._language_name()}. "
            "Return valid JSON only matching the responseFormat shape exactly.\n\n"
            f"{json.dumps(payload, ensure_ascii=True)}"
        )


build_prompts = BuildPrompts()
