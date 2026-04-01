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

	def __init__(self, prompt_dir: Optional[Path] = None, max_prompt_chars: int = 24000):
		self.prompt_dir = prompt_dir or Path(__file__).parent
		self.max_prompt_chars = max_prompt_chars

	def _clip_text(self, text: str, max_chars: Optional[int] = None) -> str:
		limit = max_chars or self.max_prompt_chars
		cleaned = (text or "").strip()
		if len(cleaned) <= limit:
			return cleaned
		head = cleaned[: int(limit * 0.7)]
		tail = cleaned[-int(limit * 0.3):]
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
		- Keep extracted text concise and faithful to CV.
		- experience_years must be numeric (e.g., 3 or 4.5) or null.

		CV text:
		{clipped}
		""".strip()

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

		normalized_weights = {
			k: round(v / total, 4)
			for k, v in weights.items()
		}

		payload = {
			"jobDescription": self._clip_text(job_description_text, max_chars=12000),
			"sectionWeights": normalized_weights,
			"candidates": [
				{
					"candidateId": str(candidate.get("id") or candidate.get("candidateId") or ""),
					"fullName": candidate.get("full_name") or candidate.get("fullName"),
					"currentJobTitle": candidate.get("current_job_title")
					or candidate.get("currentJobTitle"),
					"education": candidate.get("education_text") or candidate.get("education"),
					"experience": candidate.get("experience_text") or candidate.get("experience"),
					"skills": candidate.get("skills_text") or candidate.get("skills"),
					"summary": candidate.get("summary_text") or candidate.get("summary"),
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
			cv_name = item.get("cvName") or item.get("name") or item.get("fullName") or ""
			sections = item.get("sections") or {}

			if isinstance(sections, list):
				sections_text = "\n".join(str(s) for s in sections)
				sections = {"raw": self._clip_text(sections_text, max_chars=3000)}
			elif isinstance(sections, dict):
				clipped_sections: Dict[str, str] = {}
				for key, value in sections.items():
					if allowed_sections and key not in allowed_sections:
						continue
					clipped_sections[str(key)] = self._clip_text(str(value), max_chars=1200)
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
			"Do not invent IDs or names. If none match, return {\"matches\": []}."
		)

		return f"{instructions}\n\n{json.dumps(payload, ensure_ascii=True)}"

	def build_dsl_query_prompt(self, question: str, current_candidate_list: list) -> str:
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
	
	def build_llm_query_prompt(self, question: str, current_candidate_list: list, relevant_fields: list) -> str:

		candidate_data = [] # Placeholder for candidate data formatting logic

		candidate_count = len(current_candidate_list)
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
	
	def build_router_prompt(self, question: str, candidate_list: list) -> str:
		return f"""You are a recruitment assistant. Return a JSON object indicating whether the user's question can be answered using structured data querying or if it requires unstructured LLM analysis. The JSON should have the following format:
{{
	"relevant_fields": [string], // list of relevant structured data fields that can be used to answer the question, e.g. ["skills_text", "experience_years"]
	"dsl_question_query": string, // question rephrased to be answered by structured data querying, if applicable . Set None if not applicable
	"llm_question_query": string, // question rephrased to be answered by LLM analysis, if applicable. Set None if not applicable
	"dsl_relevant_fields": [string], // list of structured data fields that are relevant to the question and can be used in the DSL query
	"llm_relevant_fields": [string] // list of structured data fields that are relevant to the question and can be used in the LLM analysis
	"reasoning": string // a brief explanation of the reasoning behind the routing decision
}}

If the question relevant to full_name, phone, email, location_normalized, contact, current_job_title, educated, ever_studied_abroad, major, cpa, then the question is relevant to structured data querying. Otherwise, the question is relevant to LLM analysis.
Question: {question}
"""

build_prompts = BuildPrompts()
