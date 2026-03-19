from __future__ import annotations

import json
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models.candidate import CandidateProfile
from src.services.llm.llm_client import LLMRequest, generate_json


@dataclass
class LLMSemanticResult:
    candidate_ids: list[str]
    matched_count: int
    trace: dict


def _fallback_semantic_match(session: Session, question: str, limit: int) -> list[str]:
    lowered = question.lower()
    tokens = [token for token in lowered.replace(",", " ").split() if len(token) >= 4][:8]
    rows = list(
        session.execute(
            select(
                CandidateProfile.id,
                CandidateProfile.summary_text,
                CandidateProfile.skills_text,
                CandidateProfile.experience_text,
            ).limit(500)
        )
    )

    scored: list[tuple[str, int]] = []
    for candidate_id, summary_text, skills_text, experience_text in rows:
        haystack = " ".join(
            [
                summary_text or "",
                skills_text or "",
                experience_text or "",
            ]
        ).lower()
        score = sum(1 for token in tokens if token in haystack)
        if score > 0:
            scored.append((str(candidate_id), score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return [candidate_id for candidate_id, _ in scored[: max(1, min(limit, 200))]]


def run_llm_semantic_search(session: Session, question: str, limit: int = 200) -> LLMSemanticResult:
    candidates = list(
        session.execute(
            select(
                CandidateProfile.id,
                CandidateProfile.full_name,
                CandidateProfile.summary_text,
                CandidateProfile.skills_text,
                CandidateProfile.experience_text,
            ).limit(200)
        )
    )

    candidate_payload = [
        {
            "id": str(candidate_id),
            "name": full_name,
            "summary": summary_text or "",
            "skills": skills_text or "",
            "experience": experience_text or "",
        }
        for candidate_id, full_name, summary_text, skills_text, experience_text in candidates
    ]

    prompt = (
        "Given the recruiter question and candidate snippets, return JSON only with key matchedCandidateIds "
        "as an ordered array of IDs that best satisfy the query. Return at most "
        f"{max(1, min(limit, 200))} IDs. Question: {question}\n"
        f"Candidates: {json.dumps(candidate_payload, ensure_ascii=True)}"
    )

    try:
        response = generate_json(
            LLMRequest(
                prompt=prompt,
                system_prompt="You are a precise recruiter search assistant. Output strict JSON.",
                temperature=0.0,
            )
        )
        matched_candidate_ids = [str(item) for item in response.get("matchedCandidateIds", [])]
    except Exception:
        matched_candidate_ids = _fallback_semantic_match(session, question, limit)

    unique_ids: list[str] = []
    seen = set()
    for candidate_id in matched_candidate_ids:
        if candidate_id not in seen:
            unique_ids.append(candidate_id)
            seen.add(candidate_id)

    return LLMSemanticResult(
        candidate_ids=unique_ids,
        matched_count=len(unique_ids),
        trace={"tool": "llm_semantic", "candidate_pool_size": len(candidate_payload)},
    )
