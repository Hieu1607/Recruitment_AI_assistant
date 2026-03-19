from __future__ import annotations

import json
from typing import Any

from src.models.candidate import CandidateProfile
from src.services.llm.llm_client import LLMClientFactory, LLMRequest


def _fallback_scores(candidates: list[CandidateProfile]) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        has_skills = 1.0 if (candidate.skills_text or "").strip() else 0.0
        has_education = 1.0 if bool(candidate.educated) else 0.0
        has_experience = 1.0 if (candidate.experience_text or "").strip() else 0.0

        component_scores = [
            {
                "criterionKey": "skills",
                "weight": 0.4,
                "score": has_skills * 100,
                "weightedScore": has_skills * 40,
                "evidenceSummary": "Derived from normalized skills section",
            },
            {
                "criterionKey": "education",
                "weight": 0.3,
                "score": has_education * 100,
                "weightedScore": has_education * 30,
                "evidenceSummary": "Derived from education indicators",
            },
            {
                "criterionKey": "experience",
                "weight": 0.3,
                "score": has_experience * 100,
                "weightedScore": has_experience * 30,
                "evidenceSummary": "Derived from experience section",
            },
        ]
        total = sum(item["weightedScore"] for item in component_scores)
        scores.append(
            {
                "candidateId": str(candidate.id),
                "totalScore": total,
                "passedThreshold": total >= 60,
                "rationale": "Fallback heuristic applied because structured LLM output was unavailable.",
                "componentScores": component_scores,
                "scoreListIndex": index,
            }
        )
    return scores


def execute_batch_scoring(prompt: str, candidates: list[CandidateProfile]) -> list[dict[str, Any]]:
    request = LLMRequest(prompt=prompt, temperature=0.0)
    raw = LLMClientFactory.create().generate(request)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return _fallback_scores(candidates)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("scores"), list):
        return payload["scores"]
    return _fallback_scores(candidates)
