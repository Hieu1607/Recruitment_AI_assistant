from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from src.api.errors import AppError
from src.models.candidate import CandidateProfile
from src.models.engagement import InterviewQuestionSet
from src.models.matching import JobDescription


@dataclass
class InterviewQuestionInput:
    candidate_id: uuid.UUID
    job_description_id: uuid.UUID
    generated_by_user_id: uuid.UUID
    question_count: int


class InterviewQuestionService:
    def generate_questions(self, session: Session, payload: InterviewQuestionInput) -> InterviewQuestionSet:
        candidate = session.get(CandidateProfile, payload.candidate_id)
        if not candidate:
            raise AppError(code="candidate_not_found", message="Candidate not found", status_code=404)

        job_description = session.get(JobDescription, payload.job_description_id)
        if not job_description:
            raise AppError(code="job_description_not_found", message="Job description not found", status_code=404)

        questions = self._build_questions(candidate, job_description.jd_text, payload.question_count)
        question_set = InterviewQuestionSet(
            candidate_profile_id=payload.candidate_id,
            job_description_id=payload.job_description_id,
            generated_by_user_id=payload.generated_by_user_id,
            question_payload={"questions": questions},
        )
        session.add(question_set)
        session.flush()
        return question_set

    @staticmethod
    def _build_questions(candidate: CandidateProfile, jd_text: str, question_count: int) -> list[dict[str, str]]:
        base_categories = ["experience", "skills", "problem_solving", "collaboration", "domain"]
        role_hint = candidate.current_job_title or "this role"
        jd_excerpt = jd_text[:240].replace("\n", " ").strip()

        questions: list[dict[str, str]] = []
        for index in range(max(3, min(question_count, 25))):
            category = base_categories[index % len(base_categories)]
            questions.append(
                {
                    "prompt": (
                        f"Question {index + 1}: For {role_hint}, explain how your past work relates to this requirement: "
                        f"{jd_excerpt}"
                    ),
                    "category": category,
                    "difficulty": "medium" if index % 2 == 0 else "hard",
                }
            )
        return questions


interview_question_service = InterviewQuestionService()
