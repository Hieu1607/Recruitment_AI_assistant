from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy.orm import Session, joinedload

from src.models.candidate_profile import CandidateProfile

from src.models.enums import CandidateEvaluationStatus
from src.models.job_matching import JobDescription
from src.models.resume_document import ResumeDocument
from src.models.scoring_evaluation import CandidateEvaluation, JobScoringPreference
from src.services import score_candidate
from src.services.scoring_preferences import calculate_weighted_score
from src.services.scoring_signature import compute_scoring_signature


def _status_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def get_job_scoring_preference(db: Session, job_id: uuid.UUID) -> JobScoringPreference | None:
    return db.get(JobScoringPreference, job_id)


def current_signature_for_jd(jd: JobDescription) -> str:
    return compute_scoring_signature(
        job_description_id=jd.id,
        jd_text=jd.jd_text,
        hidden_text=jd.hidden_text,
    )


def mark_job_evaluations_outdated(
    *,
    db: Session,
    job_id: uuid.UUID,
    current_scoring_signature: str,
) -> int:
    rows = (
        db.query(CandidateEvaluation)
        .filter(
            CandidateEvaluation.job_id == job_id,
            CandidateEvaluation.scoring_signature != current_scoring_signature,
            CandidateEvaluation.status == CandidateEvaluationStatus.COMPLETED.value,
        )
        .all()
    )
    for row in rows:
        row.status = CandidateEvaluationStatus.OUTDATED.value
    return len(rows)


def serialize_candidate_evaluation(
    evaluation: CandidateEvaluation,
    preference: JobScoringPreference | None = None,
) -> dict[str, Any]:
    profile = evaluation.candidate_profile
    resume = profile.resume_document if profile is not None else None
    candidate_name = str(profile.full_name or "").strip() if profile is not None else ""
    resume_file_name = str(resume.original_file_name or "").strip() if resume is not None else ""
    status_value = _status_value(evaluation.status)
    weighted = calculate_weighted_score(
        raw_component_scores=evaluation.raw_component_scores or [],
        section_weights=preference.section_weights if preference else None,
        score_threshold=preference.score_threshold if preference else Decimal("50.00"),
    )
    return {
        "id": str(evaluation.id),
        "job_id": str(evaluation.job_id),
        "job_description_id": str(evaluation.job_description_id),
        "candidate_profile_id": str(evaluation.candidate_profile_id),
        "candidateName": candidate_name or None,
        "resumeFileName": resume_file_name or None,
        "candidateDisplayName": candidate_name or resume_file_name or None,
        "scoring_signature": evaluation.scoring_signature,
        "status": status_value,
        "rationale": evaluation.rationale_summary,
        "error_message": evaluation.error_message,
        "scored_at": evaluation.scored_at,
        **weighted,
    }


def list_latest_job_evaluations(db: Session, job_id: uuid.UUID) -> list[CandidateEvaluation]:
    rows = (
        db.query(CandidateEvaluation)
        .options(
            joinedload(CandidateEvaluation.candidate_profile).joinedload(
                CandidateProfile.resume_document
            )
        )
        .filter(CandidateEvaluation.job_id == job_id)
        .order_by(CandidateEvaluation.updated_at.desc(), CandidateEvaluation.created_at.desc())
        .all()
    )
    current_jd = (
        db.query(JobDescription)
        .filter(
            JobDescription.job_id == job_id,
            JobDescription.is_active.is_(True),
        )
        .order_by(JobDescription.created_at.desc())
        .first()
    )
    current_signature = current_signature_for_jd(current_jd) if current_jd is not None else None
    latest_by_candidate: dict[uuid.UUID, CandidateEvaluation] = {}
    for row in rows:
        existing = latest_by_candidate.get(row.candidate_profile_id)
        if existing is None:
            latest_by_candidate[row.candidate_profile_id] = row
            continue
        if current_signature and row.scoring_signature == current_signature and existing.scoring_signature != current_signature:
            latest_by_candidate[row.candidate_profile_id] = row
    return list(latest_by_candidate.values())


def get_latest_candidate_evaluation(
    *,
    db: Session,
    job_id: uuid.UUID,
    candidate_profile_id: uuid.UUID,
) -> CandidateEvaluation | None:
    rows = (
        db.query(CandidateEvaluation)
        .options(
            joinedload(CandidateEvaluation.candidate_profile).joinedload(
                CandidateProfile.resume_document
            )
        )
        .filter(
            CandidateEvaluation.job_id == job_id,
            CandidateEvaluation.candidate_profile_id == candidate_profile_id,
        )
        .order_by(CandidateEvaluation.updated_at.desc(), CandidateEvaluation.created_at.desc())
        .all()
    )
    if not rows:
        return None
    current_jd = (
        db.query(JobDescription)
        .filter(
            JobDescription.job_id == job_id,
            JobDescription.is_active.is_(True),
        )
        .order_by(JobDescription.created_at.desc())
        .first()
    )
    current_signature = current_signature_for_jd(current_jd) if current_jd is not None else None
    if current_signature is None:
        return rows[0]
    for row in rows:
        if row.scoring_signature == current_signature:
            return row
    return rows[0]


def upsert_job_scoring_preference(
    *,
    db: Session,
    job_id: uuid.UUID,
    section_weights: dict[str, float],
    score_threshold: Decimal,
    updated_by_user_id: uuid.UUID,
) -> JobScoringPreference:
    preference = db.get(JobScoringPreference, job_id)
    if preference is None:
        preference = JobScoringPreference(job_id=job_id)
        db.add(preference)
    preference.section_weights = section_weights
    preference.score_threshold = score_threshold
    preference.updated_by_user_id = updated_by_user_id
    db.commit()
    db.refresh(preference)
    return preference


def enqueue_missing_current_evaluations(*, db: Session, job_id: uuid.UUID, jd: JobDescription) -> dict[str, Any]:
    from worker.tasks import evaluate_candidate

    signature = current_signature_for_jd(jd)
    profiles = (
        db.query(CandidateProfile)
        .options(joinedload(CandidateProfile.resume_document))
        .join(ResumeDocument, ResumeDocument.id == CandidateProfile.resume_document_id)
        .filter(ResumeDocument.job_id == job_id)
        .all()
    )
    queued_profile_ids: list[str] = []
    for profile in profiles:
        if _ensure_pending_evaluation_for_signature(
            db=db,
            profile=profile,
            jd=jd,
            signature=signature,
        ):
            queued_profile_ids.append(str(profile.id))
    db.commit()
    for candidate_profile_id in queued_profile_ids:
        evaluate_candidate.delay(candidate_profile_id)
    return {"queued": len(queued_profile_ids), "total_candidates": len(profiles)}


def queue_candidate_evaluation_for_current_jd(*, db: Session, candidate_profile_id: uuid.UUID) -> bool:
    from worker.tasks import evaluate_candidate

    profile = (
        db.query(CandidateProfile)
        .options(joinedload(CandidateProfile.resume_document))
        .filter(CandidateProfile.id == candidate_profile_id)
        .first()
    )
    if profile is None or profile.resume_document is None:
        return False

    jd = (
        db.query(JobDescription)
        .filter(
            JobDescription.job_id == profile.resume_document.job_id,
            JobDescription.is_active.is_(True),
        )
        .order_by(JobDescription.created_at.desc())
        .first()
    )
    if jd is None:
        return False

    signature = current_signature_for_jd(jd)
    should_queue = _ensure_pending_evaluation_for_signature(
        db=db,
        profile=profile,
        jd=jd,
        signature=signature,
    )
    db.commit()
    if should_queue:
        evaluate_candidate.delay(str(profile.id))
    return should_queue


def _ensure_pending_evaluation_for_signature(
    *,
    db: Session,
    profile: CandidateProfile,
    jd: JobDescription,
    signature: str,
) -> bool:
    resume = profile.resume_document
    if resume is None:
        return False

    evaluation = (
        db.query(CandidateEvaluation)
        .filter(
            CandidateEvaluation.job_description_id == jd.id,
            CandidateEvaluation.candidate_profile_id == profile.id,
            CandidateEvaluation.scoring_signature == signature,
        )
        .order_by(CandidateEvaluation.updated_at.desc(), CandidateEvaluation.created_at.desc())
        .first()
    )
    if evaluation is not None and _status_value(evaluation.status) in {
        CandidateEvaluationStatus.COMPLETED.value,
        CandidateEvaluationStatus.PENDING.value,
        CandidateEvaluationStatus.RUNNING.value,
    }:
        return False

    if evaluation is None:
        evaluation = CandidateEvaluation(
            job_id=resume.job_id,
            job_description_id=jd.id,
            candidate_profile_id=profile.id,
            scoring_signature=signature,
            rubric_payload={},
            raw_component_scores=[],
            rationale_summary="",
            status=CandidateEvaluationStatus.PENDING.value,
        )
        db.add(evaluation)

    evaluation.job_id = resume.job_id
    evaluation.job_description_id = jd.id
    evaluation.candidate_profile_id = profile.id
    evaluation.scoring_signature = signature
    evaluation.rubric_payload = {}
    evaluation.raw_component_scores = []
    evaluation.rationale_summary = ""
    evaluation.error_message = None
    evaluation.scored_at = None
    evaluation.status = CandidateEvaluationStatus.PENDING.value
    db.flush()
    return True


def evaluate_candidate_for_current_jd(
    *,
    db: Session,
    candidate_profile_id: uuid.UUID,
) -> CandidateEvaluation:
    profile = (
        db.query(CandidateProfile)
        .options(joinedload(CandidateProfile.resume_document))
        .filter(CandidateProfile.id == candidate_profile_id)
        .first()
    )
    if profile is None:
        raise ValueError(f"Candidate profile '{candidate_profile_id}' not found")

    resume = profile.resume_document
    if resume is None:
        raise ValueError(f"Resume document for candidate '{candidate_profile_id}' not found")

    jd = (
        db.query(JobDescription)
        .filter(
            JobDescription.job_id == resume.job_id,
            JobDescription.is_active.is_(True),
        )
        .order_by(JobDescription.created_at.desc())
        .first()
    )
    if jd is None:
        raise ValueError(f"Active job description not found for job '{resume.job_id}'")

    signature = current_signature_for_jd(jd)
    existing_completed = (
        db.query(CandidateEvaluation)
        .filter(
            CandidateEvaluation.job_description_id == jd.id,
            CandidateEvaluation.candidate_profile_id == profile.id,
            CandidateEvaluation.scoring_signature == signature,
            CandidateEvaluation.status == CandidateEvaluationStatus.COMPLETED.value,
        )
        .first()
    )
    if existing_completed is not None:
        return existing_completed

    evaluation = (
        db.query(CandidateEvaluation)
        .filter(
            CandidateEvaluation.job_description_id == jd.id,
            CandidateEvaluation.candidate_profile_id == profile.id,
            CandidateEvaluation.scoring_signature == signature,
        )
        .first()
    )
    if evaluation is None:
        evaluation = CandidateEvaluation(
            job_id=resume.job_id,
            job_description_id=jd.id,
            candidate_profile_id=profile.id,
            scoring_signature=signature,
            rubric_payload={},
            raw_component_scores=[],
            rationale_summary="",
            status=CandidateEvaluationStatus.PENDING.value,
        )
        db.add(evaluation)

    evaluation.status = CandidateEvaluationStatus.RUNNING.value
    evaluation.error_message = None
    db.flush()

    try:
        raw_result = score_candidate.evaluate_candidate_profile_raw(
            candidate=score_candidate._profile_to_candidate_dict(profile),
            job_description_text=score_candidate._build_scoring_job_description_text(
                public_job_description=jd.jd_text,
                hidden_text=jd.hidden_text,
            ),
        )
        evaluation.rubric_payload = raw_result.get("rubricPayload") or {}
        evaluation.raw_component_scores = raw_result.get("rawComponentScores") or []
        evaluation.rationale_summary = str(raw_result.get("rationaleSummary") or "").strip()
        evaluation.status = CandidateEvaluationStatus.COMPLETED.value
        evaluation.scored_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(evaluation)
        return evaluation
    except Exception as exc:
        evaluation.status = CandidateEvaluationStatus.FAILED.value
        evaluation.error_message = str(exc)
        db.commit()
        db.refresh(evaluation)
        return evaluation
