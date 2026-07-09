from __future__ import annotations

import json
import logging
import time
import uuid
from json import JSONDecodeError

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from src.models.interview_invitation import InterviewInvitation
from src.models.interview_session import InterviewReport, InterviewSession, InterviewTranscriptTurn
from src.models.session import SessionLocal
from src.schemas.interview_report import (
    InterviewReportFailure,
    InterviewReportPayload,
    InterviewReportSummary,
    InterviewReportTaskState,
)
from src.core.config import settings
from src.services.llm_service import LLMProvider


logger = logging.getLogger(__name__)


def _ui_language() -> str:
    return "en" if str(settings.APP_UI_LANGUAGE or "").strip().lower().startswith("en") else "vi"


REPORT_SYSTEM_PROMPT = """
You generate descriptive HR interview summaries from transcript evidence.
Return valid JSON only.
Do not include hiring recommendations, accept/reject decisions, or final verdicts.
Ground every competency in transcript evidence.
Each evidence item must reference the exact transcript turn it came from.
""".strip()


PERMANENT_REPORT_EXCEPTIONS = (HTTPException, ValidationError, JSONDecodeError, ValueError)


def generate_interview_report(db: Session, *, interview_session_id: uuid.UUID) -> InterviewReport:
    generation_started_at = time.perf_counter()
    context_started_at = time.perf_counter()
    session_record = _get_session_with_context(db, interview_session_id)
    context_load_ms = (time.perf_counter() - context_started_at) * 1000
    transcript_turns = sorted(session_record.transcript_turns, key=lambda turn: turn.turn_index)
    if not transcript_turns:
        raise HTTPException(status_code=409, detail="Interview session transcript is empty")

    report_summary = _generate_report_summary(session_record, transcript_turns)
    validation_started_at = time.perf_counter()
    _validate_report_evidence_links(report_summary, transcript_turns)
    markdown_summary = _render_markdown_summary(report_summary)
    validation_render_ms = (time.perf_counter() - validation_started_at) * 1000
    persistence_started_at = time.perf_counter()
    report = _get_or_create_report(db, session_record)
    report.interview_template_id = session_record.invitation.interview_template_id
    report.summary_text = markdown_summary
    report.report_payload = InterviewReportPayload(
        status="completed",
        summary=report_summary,
    ).to_payload()
    db.commit()
    db.refresh(report)
    persistence_ms = (time.perf_counter() - persistence_started_at) * 1000
    logger.info(
        "interview_report_generation_completed session_id=%s total_ms=%.3f "
        "context_load_ms=%.3f validation_render_ms=%.3f persistence_ms=%.3f "
        "turn_count=%d",
        interview_session_id,
        (time.perf_counter() - generation_started_at) * 1000,
        context_load_ms,
        validation_render_ms,
        persistence_ms,
        len(transcript_turns),
    )
    return report


def generate_interview_report_for_session(interview_session_id: uuid.UUID) -> dict[str, str]:
    db: Session = SessionLocal()
    try:
        report = generate_interview_report(db, interview_session_id=interview_session_id)
        return {"status": "completed", "report_id": str(report.id)}
    finally:
        db.close()


def mark_interview_report_pending(
    interview_session_id: uuid.UUID,
    *,
    task_id: str | None,
    retry_count: int = 0,
    state: str = "queued",
) -> dict[str, str]:
    db: Session = SessionLocal()
    try:
        _mark_interview_report_pending_in_db(
            db,
            interview_session_id=interview_session_id,
            task_id=task_id,
            retry_count=retry_count,
            state=state,
        )
        return {"status": "pending"}
    finally:
        db.close()


def mark_interview_report_failure(
    interview_session_id: uuid.UUID,
    *,
    stage: str,
    message: str,
    retryable: bool,
    retry_count: int = 0,
) -> dict[str, str]:
    db: Session = SessionLocal()
    try:
        result = mark_interview_report_failure_in_db(
            db,
            interview_session_id=interview_session_id,
            stage=stage,
            message=message,
            retryable=retryable,
            retry_count=retry_count,
        )
        return {"status": "failed", "report_id": str(result.id)}
    finally:
        db.close()


def is_permanent_report_error(exc: Exception) -> bool:
    return isinstance(exc, PERMANENT_REPORT_EXCEPTIONS)


def mark_interview_report_pending_in_db(
    db: Session,
    *,
    interview_session_id: uuid.UUID,
    task_id: str | None,
    retry_count: int = 0,
    state: str = "queued",
) -> InterviewReport:
    return _mark_interview_report_pending_in_db(
        db,
        interview_session_id=interview_session_id,
        task_id=task_id,
        retry_count=retry_count,
        state=state,
    )


def mark_interview_report_failure_in_db(
    db: Session,
    *,
    interview_session_id: uuid.UUID,
    stage: str,
    message: str,
    retryable: bool,
    retry_count: int = 0,
) -> InterviewReport:
    session_record = _get_session_with_context(db, interview_session_id)
    report = _get_or_create_report(db, session_record)
    report.interview_template_id = session_record.invitation.interview_template_id
    report.summary_text = None
    report.report_payload = InterviewReportPayload(
        status="failed",
        failure=InterviewReportFailure(
            stage=stage,
            message=message,
            retryable=retryable,
        ),
        task=InterviewReportTaskState(
            state="failed",
            task_id=_extract_task_id(report.report_payload),
            retry_count=retry_count,
        ),
    ).to_payload()
    db.commit()
    db.refresh(report)
    return report


def _generate_report_summary(
    session_record: InterviewSession,
    transcript_turns: list[InterviewTranscriptTurn],
) -> InterviewReportSummary:
    prompt_started_at = time.perf_counter()
    prompt = _build_report_prompt(session_record, transcript_turns)
    prompt_build_ms = (time.perf_counter() - prompt_started_at) * 1000
    llm = LLMProvider()
    request_started_at = time.perf_counter()
    response = llm.generate(
        prompt,
        system_prompt=REPORT_SYSTEM_PROMPT,
    )
    request_ms = (time.perf_counter() - request_started_at) * 1000
    parse_started_at = time.perf_counter()
    summary = InterviewReportSummary.model_validate(_parse_json_response(response.text))
    parse_validate_ms = (time.perf_counter() - parse_started_at) * 1000
    logger.info(
        "interview_report_llm_completed session_id=%s provider=%s model=%s "
        "prompt_build_ms=%.3f request_ms=%.3f parse_validate_ms=%.3f "
        "prompt_chars=%d response_chars=%d turn_count=%d",
        session_record.id,
        response.provider,
        response.model,
        prompt_build_ms,
        request_ms,
        parse_validate_ms,
        len(prompt),
        len(response.text),
        len(transcript_turns),
    )
    return summary


def _build_report_prompt(
    session_record: InterviewSession,
    transcript_turns: list[InterviewTranscriptTurn],
) -> str:
    invitation = session_record.invitation
    template = invitation.interview_template
    candidate_name = invitation.candidate_profile.full_name if invitation.candidate_profile is not None else "Candidate"
    transcript_lines = []
    for turn in transcript_turns:
        question_key = (turn.payload or {}).get("question_key")
        transcript_lines.append(
            json.dumps(
                {
                    "transcript_turn_id": str(turn.id),
                    "turn_index": turn.turn_index,
                    "speaker_role": turn.speaker_role,
                    "question_key": question_key,
                    "transcript_text": turn.transcript_text,
                },
                ensure_ascii=True,
            )
        )

    rubric = template.report_rubric if template is not None else {}
    question_payload = template.question_payload if template is not None else {}
    language_name = "English" if _ui_language() == "en" else "Vietnamese"
    return (
        "Summarize this completed interview as structured JSON.\n"
        "Required JSON shape:\n"
        "{\n"
        '  "candidate_overview": "string",\n'
        '  "questions": [\n'
        '    {\n'
        '      "question_key": "string or null",\n'
        '      "question_text": "string (the interviewer question, copied from the assistant turn transcript_text)",\n'
        '      "question_transcript_turn_id": "uuid (the assistant turn that asked this question)",\n'
        '      "question_turn_index": 0,\n'
        '      "answer_text": "string (the candidate answer, copied from the candidate turn transcript_text)",\n'
        '      "answer_transcript_turn_id": "uuid (the candidate turn that answered this question)",\n'
        '      "answer_turn_index": 0,\n'
        '      "evaluation": "string (a descriptive, evidence-based assessment of this specific answer)"\n'
        "    }\n"
        "  ],\n"
        '  "overall_summary": "string"\n'
        "}\n"
        "Constraints:\n"
        "- Include one entry in \"questions\" for every assistant question that received a candidate answer, in transcript order.\n"
        "- question_text and answer_text must be copied verbatim from the transcript metadata above.\n"
        "- question_transcript_turn_id/question_turn_index must reference the assistant turn; "
        "answer_transcript_turn_id/answer_turn_index must reference the candidate turn.\n"
        "- The evaluation must be descriptive and grounded strictly in that answer's content.\n"
        "- Do not provide hiring recommendations or accept/reject language anywhere.\n"
        "- Use only details supported by the transcript.\n"
        f"- Write every text field in {language_name}.\n"
        f"Candidate: {candidate_name}\n"
        f"Template rubric: {json.dumps(rubric, ensure_ascii=True)}\n"
        f"Template questions: {json.dumps(question_payload, ensure_ascii=True)}\n"
        "Transcript:\n"
        f"{chr(10).join(transcript_lines)}"
    )


def _parse_json_response(raw_text: str) -> dict:
    candidate = raw_text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
    return json.loads(candidate)


def _render_markdown_summary(report_summary: InterviewReportSummary) -> str:
    vi = _ui_language() == "vi"
    lines = [
        "# Báo cáo phỏng vấn" if vi else "# Interview Report",
        "",
        "## Tổng quan ứng viên" if vi else "## Candidate Overview",
        report_summary.candidate_overview,
        "",
        "## Câu hỏi & Đánh giá" if vi else "## Questions & Answers",
    ]
    for index, question_item in enumerate(report_summary.questions, start=1):
        heading = f"### {vi and 'Câu hỏi' or 'Question'} {index}: {question_item.question_text}"
        lines.extend(
            [
                heading,
                f"**{'Trả lời' if vi else 'Answer'}:** {question_item.answer_text}",
                f"**{'Đánh giá' if vi else 'Evaluation'}:** {question_item.evaluation}",
                "",
            ]
        )

    lines.extend(["## Tổng kết chung" if vi else "## Overall Summary", report_summary.overall_summary])
    return "\n".join(lines).strip()


def _validate_report_evidence_links(
    report_summary: InterviewReportSummary,
    transcript_turns: list[InterviewTranscriptTurn],
) -> None:
    transcript_lookup = {str(turn.id): turn for turn in transcript_turns}
    for question_item in report_summary.questions:
        _validate_report_turn_link(
            transcript_lookup,
            transcript_turn_id=question_item.question_transcript_turn_id,
            turn_index=question_item.question_turn_index,
            expected_speaker_role="assistant",
            transcript_text=question_item.question_text,
            question_key=question_item.question_key,
        )
        _validate_report_turn_link(
            transcript_lookup,
            transcript_turn_id=question_item.answer_transcript_turn_id,
            turn_index=question_item.answer_turn_index,
            expected_speaker_role="candidate",
            transcript_text=question_item.answer_text,
            question_key=question_item.question_key,
        )


def _validate_report_turn_link(
    transcript_lookup: dict[str, InterviewTranscriptTurn],
    *,
    transcript_turn_id: str,
    turn_index: int,
    expected_speaker_role: str,
    transcript_text: str,
    question_key: str | None,
) -> None:
    transcript_turn = transcript_lookup.get(transcript_turn_id)
    if transcript_turn is None:
        raise ValueError(f"Evidence references unknown transcript turn: {transcript_turn_id}")
    expected_question_key = (transcript_turn.payload or {}).get("question_key")
    if transcript_turn.turn_index != turn_index:
        raise ValueError(f"Evidence turn_index mismatch for transcript turn {transcript_turn_id}")
    if expected_question_key != question_key:
        raise ValueError(f"Evidence question_key mismatch for transcript turn {transcript_turn_id}")
    if transcript_turn.speaker_role != expected_speaker_role:
        raise ValueError(f"Evidence speaker_role mismatch for transcript turn {transcript_turn_id}")
    if transcript_turn.transcript_text != transcript_text:
        raise ValueError(f"Evidence transcript_text mismatch for transcript turn {transcript_turn_id}")


def _get_session_with_context(db: Session, interview_session_id: uuid.UUID) -> InterviewSession:
    session_record = (
        db.execute(
            select(InterviewSession)
            .options(
                joinedload(InterviewSession.invitation).joinedload(InterviewInvitation.candidate_profile),
                joinedload(InterviewSession.invitation).joinedload(InterviewInvitation.interview_template),
                joinedload(InterviewSession.transcript_turns),
                joinedload(InterviewSession.report),
            )
            .where(InterviewSession.id == interview_session_id)
        )
        .scalars()
        .unique()
        .one_or_none()
    )
    if session_record is None:
        raise HTTPException(status_code=404, detail="Interview session not found")
    if session_record.status != "completed" or session_record.completed_at is None:
        raise HTTPException(status_code=409, detail="Interview session is not completed")
    return session_record


def _get_or_create_report(db: Session, session_record: InterviewSession) -> InterviewReport:
    report = session_record.report
    if report is None:
        report = InterviewReport(
            interview_session_id=session_record.id,
            interview_template_id=session_record.invitation.interview_template_id,
        )
        db.add(report)
        db.flush()
        session_record.report = report
    return report


def _extract_task_id(report_payload: dict | None) -> str | None:
    if not isinstance(report_payload, dict):
        return None
    task_payload = report_payload.get("task")
    if not isinstance(task_payload, dict):
        return None
    task_id = task_payload.get("task_id")
    return task_id if isinstance(task_id, str) else None


def _mark_interview_report_pending_in_db(
    db: Session,
    *,
    interview_session_id: uuid.UUID,
    task_id: str | None,
    retry_count: int,
    state: str,
) -> InterviewReport:
    session_record = _get_session_with_context(db, interview_session_id)
    report = _get_or_create_report(db, session_record)
    report.interview_template_id = session_record.invitation.interview_template_id
    report.summary_text = None
    report.report_payload = InterviewReportPayload(
        status="pending",
        task=InterviewReportTaskState(
            state=state,
            task_id=task_id,
            retry_count=retry_count,
        ),
    ).to_payload()
    db.commit()
    db.refresh(report)
    return report
