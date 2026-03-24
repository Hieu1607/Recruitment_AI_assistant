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
		weights = dict(self.DEFAULT_SECTION_WEIGHTS)
		if section_weights:
			for key, value in section_weights.items():
				if value is None:
					continue
				weights[str(key)] = max(0.0, float(value))

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


build_prompts = BuildPrompts()
