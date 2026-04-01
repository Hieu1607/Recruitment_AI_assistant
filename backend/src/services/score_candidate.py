from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.models.candidate_profile import CandidateProfile
from src.models.enums import MatchRunStatus
from src.models.job_matching import JobDescription, MatchResult, MatchRun
from src.prompts.build_prompts import build_prompts
from src.services.llm_service import LLMProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profile_to_candidate_dict(profile: CandidateProfile) -> Dict[str, Any]:
    """Convert a CandidateProfile ORM row to the dict shape expected by the prompt builder."""
    return {
        "id": str(profile.id),
        "full_name": profile.full_name,
        "current_job_title": profile.current_job_title,
        "education_text": profile.education_text,
        "experience_text": profile.experience_text,
        "skills_text": profile.skills_text,
        "projects_text": profile.projects_text,
        "summary_text": profile.summary_text,
        "languages_text": profile.languages_text,
        "achievements_text": profile.achievements_text,
        "certifications_text": profile.certifications_text,
        "publications_text": profile.publications_text,
        "other_text": profile.other_text,
    }


def _parse_llm_scores(raw_text: str) -> List[Dict[str, Any]]:
    """Extract the scores list from the LLM JSON response."""
    content = (raw_text or "").strip()

    # Strip markdown fences if present
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            content = "\n".join(lines[1:-1]).strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM did not return a valid JSON object")
        parsed = json.loads(content[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("LLM response is not a JSON object")

    scores = parsed.get("scores", [])
    if not isinstance(scores, list):
        raise ValueError("LLM response 'scores' field is not a list")

    return scores


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def score_candidates(
    *,
    db: Session,
    job_description_id: uuid.UUID,
    initiated_by_user_id: uuid.UUID,
    score_threshold: Decimal = Decimal("50.0"),
    candidate_profile_ids: Optional[List[uuid.UUID]] = None,
    section_weights: Optional[Dict[str, float]] = None,
    batch_size: int = 10,
) -> Dict[str, Any]:
    """Score candidate profiles against a job description using the LLM.

    Args:
        db: SQLAlchemy session.
        job_description_id: The JD to score against.
        initiated_by_user_id: The user initiating the score run.
        score_threshold: Minimum total score out of 100 to be considered a pass.
        candidate_profile_ids: Explicit list of candidate profile UUIDs to score.
            When None, all profiles are scored.
        section_weights: Mapping of section key → weight. Only the keys provided
            will be weighted; all other sections default to 0. When None, the
            prompt builder's DEFAULT_SECTION_WEIGHTS are used.
        batch_size: Number of candidates sent to the LLM in a single call (1–50).

    Returns:
        A dict with ``match_run_id``, ``job_description_id``, ``total_candidates``,
        ``total_passed_candidates``, ``batches``, and ``scores`` (flat list of 
        per-candidate score objects from the LLM).

    Raises:
        ValueError: If JD is not found or no candidates are available.
    """
    # -- Fetch JD ----------------------------------------------------------
    jd: Optional[JobDescription] = db.get(JobDescription, job_description_id)
    if jd is None:
        raise ValueError(f"Job description {job_description_id} not found")

    # -- Fetch candidates --------------------------------------------------
    query = db.query(CandidateProfile)
    if candidate_profile_ids:
        query = query.filter(CandidateProfile.id.in_(candidate_profile_ids))
    profiles: List[CandidateProfile] = query.all()

    if not profiles:
        raise ValueError("No candidate profiles found for the given parameters")

    candidate_dicts = [_profile_to_candidate_dict(p) for p in profiles]
    valid_c_ids = {p.id for p in profiles}

    # -- Create MatchRun ---------------------------------------------------
    match_run = MatchRun(
        job_description_id=jd.id,
        score_threshold=score_threshold,
        initiated_by_user_id=initiated_by_user_id,
        run_status=MatchRunStatus.RUNNING.value,
    )
    db.add(match_run)
    db.commit()
    db.refresh(match_run)

    # -- Batch processing --------------------------------------------------
    llm = LLMProvider()
    all_scores: List[Dict[str, Any]] = []
    batch_size = max(1, min(batch_size, 50))
    batches_run = 0
    passed_candidates_count = 0

    try:
        global_idx = 0
        for i in range(0, len(candidate_dicts), batch_size):
            batch = candidate_dicts[i : i + batch_size]
            prompt = build_prompts.build_batch_scoring_prompt(
                job_description_text=jd.jd_text,
                candidates=batch,
                section_weights=section_weights,
            )
            response = llm.generate(prompt)
            batch_scores = _parse_llm_scores(response.text)
            all_scores.extend(batch_scores)
            batches_run += 1

            # -- Save MatchResults for this batch --------------------------
            for score_data in batch_scores:
                c_id_str = score_data.get("candidateId")
                if not c_id_str:
                    continue
                try:
                    c_id = uuid.UUID(c_id_str)
                except ValueError:
                    continue

                if c_id not in valid_c_ids:
                    continue

                total_score = Decimal(str(score_data.get("totalScore", 0)))
                passed = bool(score_data.get("passedThreshold", False))
                if "passedThreshold" not in score_data:
                    passed = total_score >= score_threshold

                if passed:
                    passed_candidates_count += 1

                rationale = score_data.get("rationale", "")
                components = score_data.get("componentScores", [])

                mr = MatchResult(
                    match_run_id=match_run.id,
                    candidate_profile_id=c_id,
                    score_list_index=global_idx,
                    total_score=total_score,
                    passed_threshold=passed,
                    rationale_summary=rationale,
                    component_scores=components,
                )
                db.add(mr)
                global_idx += 1
            db.commit()

        match_run.run_status = MatchRunStatus.COMPLETED.value
        match_run.completed_at = datetime.now(timezone.utc)
        db.commit()

    except Exception as exc:
        db.rollback()
        match_run_db = db.get(MatchRun, match_run.id)
        if match_run_db:
            match_run_db.run_status = MatchRunStatus.FAILED.value
            match_run_db.completed_at = datetime.now(timezone.utc)
            db.commit()
        raise ValueError(f"Matching failed: {exc}") from exc

    return {
        "match_run_id": str(match_run.id),
        "job_description_id": str(job_description_id),
        "total_candidates": len(candidate_dicts),
        "total_passed_candidates": passed_candidates_count,
        "batches": batches_run,
        "scores": all_scores,
    }
