from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.models.candidate_profile import CandidateProfile
from src.models.enums import MatchRunStatus
from src.models.job_matching import JobDescription, MatchResult, MatchRun
from src.models.resume_document import ResumeDocument
from src.prompts.build_prompts import build_prompts
from src.services.llm_service import LLMProvider


SUPPORTED_SCORING_SECTIONS = tuple(build_prompts.SUPPORTED_SCORING_SECTIONS)
SUPPORTED_CRITERION_TYPES = {"must_have", "semantic", "upper_bound"}
NUMERIC_OPERATORS = {">=", ">", "<=", "<", "==", "="}


def _clamp_score(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return round(max(0.0, min(100.0, numeric)), 2)


def _parse_json_object(raw_text: str) -> Dict[str, Any]:
    content = (raw_text or "").strip()
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
    return parsed


def _profile_to_candidate_dict(profile: CandidateProfile) -> Dict[str, Any]:
    """Convert a CandidateProfile ORM row to the dict shape expected by the scoring pipeline."""
    return {
        "id": str(profile.id),
        "full_name": profile.full_name,
        "current_job_title": profile.current_job_title,
        "experience_years": float(profile.experience_years) if profile.experience_years is not None else None,
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
    parsed = _parse_json_object(raw_text)
    scores = parsed.get("scores", [])
    if not isinstance(scores, list):
        raise ValueError("LLM response 'scores' field is not a list")
    return scores


def _parse_rubric_response(raw_text: str) -> Dict[str, Any]:
    parsed = _parse_json_object(raw_text)
    if "rubric" in parsed and isinstance(parsed["rubric"], dict):
        parsed = parsed["rubric"]
    criteria = parsed.get("criteria", [])
    if not isinstance(criteria, list):
        raise ValueError("Rubric response 'criteria' field is not a list")
    return {"criteria": criteria}


def _build_scoring_job_description_text(
    *,
    public_job_description: str,
    hidden_text: Optional[str] = None,
) -> str:
    """Combine public JD and recruiter-only criteria for scoring prompts."""
    parts = [
        "Public job description:",
        (public_job_description or "").strip(),
    ]
    hidden = (hidden_text or "").strip()
    if hidden:
        parts.extend(
            [
                "",
                "Recruiter-only hidden information:",
                hidden,
            ]
        )
    return "\n".join(parts).strip()


def _normalize_runtime_section_weights(
    section_weights: Optional[Dict[str, float]],
    active_sections: List[str],
) -> Dict[str, float]:
    raw_weights = (
        {str(k): float(v) for k, v in section_weights.items() if v is not None and float(v) > 0}
        if section_weights is not None
        else dict(build_prompts.DEFAULT_SECTION_WEIGHTS)
    )
    filtered = {section: raw_weights.get(section, 0.0) for section in active_sections if raw_weights.get(section, 0.0) > 0}
    if not filtered:
        filtered = {section: 1.0 for section in active_sections}
    total = sum(filtered.values())
    if total <= 0:
        raise ValueError("section_weights total must be > 0")
    return {section: round(weight / total, 4) for section, weight in filtered.items()}


def _normalize_measurable(measurable: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(measurable, dict):
        return None
    field = str(measurable.get("field") or measurable.get("path") or "").strip()
    operator = str(measurable.get("operator") or measurable.get("op") or "").strip()
    value = measurable.get("value")
    if not field or not operator or value in (None, ""):
        return None
    return {
        "field": field,
        "operator": operator,
        "value": value,
    }


def _normalize_rubric(
    rubric: Dict[str, Any],
    section_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    raw_criteria = rubric.get("criteria", [])
    normalized_criteria: List[Dict[str, Any]] = []

    for idx, criterion in enumerate(raw_criteria):
        if not isinstance(criterion, dict):
            continue
        section = str(criterion.get("section") or "").strip().lower()
        requirement_text = str(criterion.get("requirementText") or "").strip()
        criterion_type = str(criterion.get("type") or "semantic").strip().lower()
        if section not in SUPPORTED_SCORING_SECTIONS or not requirement_text:
            continue
        if criterion_type not in SUPPORTED_CRITERION_TYPES:
            criterion_type = "semantic"
        key = str(criterion.get("key") or f"{section}.{idx + 1}").strip() or f"{section}.{idx + 1}"
        normalized_criteria.append(
            {
                "key": key,
                "section": section,
                "requirementText": requirement_text,
                "type": criterion_type,
                "measurable": _normalize_measurable(criterion.get("measurable")),
            }
        )

    if not normalized_criteria:
        return {"criteria": [], "sectionWeights": {}}

    active_sections = list(dict.fromkeys(criterion["section"] for criterion in normalized_criteria))
    normalized_section_weights = _normalize_runtime_section_weights(section_weights, active_sections)
    section_counts = Counter(criterion["section"] for criterion in normalized_criteria)

    for criterion in normalized_criteria:
        section_weight = normalized_section_weights[criterion["section"]]
        criterion["weight"] = round(section_weight / section_counts[criterion["section"]], 4)

    return {
        "criteria": normalized_criteria,
        "sectionWeights": normalized_section_weights,
    }


def _extract_candidate_field(candidate: Dict[str, Any], field: str) -> Any:
    if field == "experience_years":
        return candidate.get("experience_years")

    if field.startswith("languages."):
        languages_text = str(candidate.get("languages_text") or candidate.get("languages") or "")
        exam = field.split(".", 1)[1].lower()
        if exam == "ielts":
            match = re.search(r"ielts[^0-9]{0,8}(\d(?:\.\d)?)", languages_text, re.IGNORECASE)
            return float(match.group(1)) if match else None
        if exam == "toeic":
            match = re.search(r"toeic[^0-9]{0,8}(\d{3,4})", languages_text, re.IGNORECASE)
            return float(match.group(1)) if match else None
        if exam == "toefl":
            match = re.search(r"toefl[^0-9]{0,8}(\d{2,3})", languages_text, re.IGNORECASE)
            return float(match.group(1)) if match else None
        return languages_text

    if field in SUPPORTED_SCORING_SECTIONS:
        return candidate.get(f"{field}_text") or candidate.get(field)

    return candidate.get(field)


def _compare_measurable(candidate_value: Any, operator: str, expected_value: Any) -> bool:
    if candidate_value in (None, ""):
        return False

    if operator == "contains":
        return str(expected_value).lower() in str(candidate_value).lower()

    if operator in NUMERIC_OPERATORS:
        try:
            left = float(candidate_value)
            right = float(expected_value)
        except (TypeError, ValueError):
            return False
        if operator == ">=":
            return left >= right
        if operator == ">":
            return left > right
        if operator == "<=":
            return left <= right
        if operator == "<":
            return left < right
        return left == right

    return str(candidate_value).strip().lower() == str(expected_value).strip().lower()


def _score_measurable_criterion(
    candidate: Dict[str, Any],
    criterion: Dict[str, Any],
) -> tuple[float, str]:
    measurable = criterion.get("measurable") or {}
    field = measurable.get("field", "")
    operator = measurable.get("operator", "")
    expected_value = measurable.get("value")
    actual_value = _extract_candidate_field(candidate, field)
    matched = _compare_measurable(actual_value, operator, expected_value)
    score = 100.0 if matched else 0.0
    evidence = f"{field} {operator} {expected_value}; candidate value: {actual_value!s}"
    return score, evidence


def _parse_semantic_scores(raw_text: str) -> Dict[str, Dict[str, Any]]:
    scores = _parse_llm_scores(raw_text)
    by_candidate: Dict[str, Dict[str, Any]] = {}
    for entry in scores:
        candidate_id = str(entry.get("candidateId") or "").strip()
        if not candidate_id:
            continue
        criteria_rows = entry.get("criteria")
        if not isinstance(criteria_rows, list):
            criteria_rows = entry.get("componentScores", [])
        mapped_criteria: Dict[str, Dict[str, Any]] = {}
        if isinstance(criteria_rows, list):
            for row in criteria_rows:
                if not isinstance(row, dict):
                    continue
                criterion_key = str(row.get("criterionKey") or "").strip()
                if not criterion_key:
                    continue
                mapped_criteria[criterion_key] = {
                    "score": _clamp_score(row.get("score", 0)),
                    "evidenceSummary": str(row.get("evidenceSummary") or "").strip(),
                }
        by_candidate[candidate_id] = {
            "rationale": str(entry.get("rationale") or "").strip(),
            "criteria": mapped_criteria,
        }
    return by_candidate


def _build_candidate_score(
    *,
    candidate: Dict[str, Any],
    rubric: Dict[str, Any],
    semantic_result: Dict[str, Any],
    score_threshold: Decimal,
) -> Dict[str, Any]:
    semantic_criteria = semantic_result.get("criteria", {}) if isinstance(semantic_result, dict) else {}
    component_scores: List[Dict[str, Any]] = []

    for criterion in rubric.get("criteria", []):
        weight = float(criterion.get("weight", 0.0))
        if criterion.get("measurable"):
            score, evidence = _score_measurable_criterion(candidate, criterion)
        else:
            semantic_detail = semantic_criteria.get(criterion["key"], {})
            score = _clamp_score(semantic_detail.get("score", 0))
            evidence = str(semantic_detail.get("evidenceSummary") or "").strip()
        component_scores.append(
            {
                "criterionKey": criterion["key"],
                "weight": round(weight, 4),
                "score": score,
                "weightedScore": round(weight * score, 2),
                "evidenceSummary": evidence,
            }
        )

    total_score = round(sum(component["weightedScore"] for component in component_scores), 2)
    rationale = str(semantic_result.get("rationale") or "").strip()
    if not rationale:
        rationale = "; ".join(
            component["evidenceSummary"]
            for component in component_scores
            if component["evidenceSummary"]
        )[:1000]

    return {
        "candidateId": str(candidate.get("id") or candidate.get("candidateId") or ""),
        "totalScore": _clamp_score(total_score),
        "passedThreshold": _clamp_score(total_score) >= float(score_threshold),
        "rationale": rationale,
        "componentScores": component_scores,
    }


def _coerce_passed_threshold(score_data: Dict[str, Any], score_threshold: Decimal) -> Dict[str, Any]:
    total_score = _clamp_score(score_data.get("totalScore", 0))
    normalized = dict(score_data)
    normalized["totalScore"] = total_score
    normalized["passedThreshold"] = total_score >= float(score_threshold)
    return normalized


def _extract_locked_rubric(
    *,
    llm: LLMProvider,
    job_description_text: str,
    section_weights: Optional[Dict[str, float]],
) -> Optional[Dict[str, Any]]:
    try:
        response = llm.generate(
            build_prompts.build_jd_rubric_extraction_prompt(
                job_description_text=job_description_text,
                section_weights=section_weights,
            )
        )
        rubric = _normalize_rubric(_parse_rubric_response(response.text), section_weights)
        return rubric if rubric.get("criteria") else None
    except Exception:
        return None


def _save_batch_scores(
    *,
    db: Session,
    match_run_id: uuid.UUID,
    batch_scores: List[Dict[str, Any]],
    valid_candidate_ids: set[uuid.UUID],
    score_threshold: Decimal,
    starting_index: int,
) -> tuple[int, int]:
    global_idx = starting_index
    passed_candidates_count = 0

    for raw_score in batch_scores:
        score_data = _coerce_passed_threshold(raw_score, score_threshold)
        candidate_id_text = score_data.get("candidateId")
        if not candidate_id_text:
            continue
        try:
            candidate_id = uuid.UUID(str(candidate_id_text))
        except ValueError:
            continue
        if candidate_id not in valid_candidate_ids:
            continue

        if score_data["passedThreshold"]:
            passed_candidates_count += 1

        db.add(
            MatchResult(
                match_run_id=match_run_id,
                candidate_profile_id=candidate_id,
                score_list_index=global_idx,
                total_score=Decimal(str(score_data["totalScore"])),
                passed_threshold=bool(score_data["passedThreshold"]),
                rationale_summary=str(score_data.get("rationale") or ""),
                component_scores=score_data.get("componentScores") or [],
            )
        )
        global_idx += 1

    return global_idx, passed_candidates_count


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
    """Score candidate profiles against a job description using rubric-locked scoring."""
    jd: Optional[JobDescription] = db.get(JobDescription, job_description_id)
    if jd is None:
        raise ValueError(f"Job description {job_description_id} not found")

    query = db.query(CandidateProfile).join(
        ResumeDocument, ResumeDocument.id == CandidateProfile.resume_document_id
    ).filter(ResumeDocument.job_id == jd.job_id)
    if candidate_profile_ids:
        query = query.filter(CandidateProfile.id.in_(candidate_profile_ids))
    profiles: List[CandidateProfile] = query.all()
    if not profiles:
        raise ValueError("No candidate profiles found for the given parameters")

    candidate_dicts = [_profile_to_candidate_dict(profile) for profile in profiles]
    valid_candidate_ids = {profile.id for profile in profiles}
    scoring_jd_text = _build_scoring_job_description_text(
        public_job_description=jd.jd_text,
        hidden_text=getattr(jd, "hidden_text", ""),
    )

    match_run = MatchRun(
        job_description_id=jd.id,
        score_threshold=score_threshold,
        initiated_by_user_id=initiated_by_user_id,
        run_status=MatchRunStatus.RUNNING.value,
    )
    db.add(match_run)
    db.commit()
    db.refresh(match_run)

    llm = LLMProvider()
    rubric = _extract_locked_rubric(
        llm=llm,
        job_description_text=scoring_jd_text,
        section_weights=section_weights,
    )

    all_scores: List[Dict[str, Any]] = []
    batch_size = max(1, min(batch_size, 50))
    batches_run = 0
    passed_candidates_count = 0

    try:
        global_idx = 0
        for i in range(0, len(candidate_dicts), batch_size):
            batch = candidate_dicts[i : i + batch_size]

            if rubric is None:
                response = llm.generate(
                    build_prompts.build_batch_scoring_prompt(
                        job_description_text=scoring_jd_text,
                        candidates=batch,
                        section_weights=section_weights,
                    )
                )
                batch_scores = [_coerce_passed_threshold(score, score_threshold) for score in _parse_llm_scores(response.text)]
            else:
                semantic_criteria = [criterion for criterion in rubric["criteria"] if criterion.get("measurable") is None]
                semantic_by_candidate: Dict[str, Dict[str, Any]] = {}
                if semantic_criteria:
                    response = llm.generate(
                        build_prompts.build_locked_rubric_semantic_scoring_prompt(
                            candidates=batch,
                            rubric={"criteria": semantic_criteria},
                        )
                    )
                    semantic_by_candidate = _parse_semantic_scores(response.text)

                batch_scores = []
                for candidate in batch:
                    semantic_result = semantic_by_candidate.get(str(candidate["id"]), {})
                    batch_scores.append(
                        _build_candidate_score(
                            candidate=candidate,
                            rubric=rubric,
                            semantic_result=semantic_result,
                            score_threshold=score_threshold,
                        )
                    )

            all_scores.extend(batch_scores)
            batches_run += 1

            global_idx, passed_delta = _save_batch_scores(
                db=db,
                match_run_id=match_run.id,
                batch_scores=batch_scores,
                valid_candidate_ids=valid_candidate_ids,
                score_threshold=score_threshold,
                starting_index=global_idx,
            )
            passed_candidates_count += passed_delta
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
