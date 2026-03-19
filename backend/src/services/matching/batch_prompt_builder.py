from __future__ import annotations

import json

from src.models.candidate import CandidateProfile


def build_batch_scoring_prompt(
    *,
    job_description_text: str,
    candidates: list[CandidateProfile],
    scoring_prompt_template: str,
) -> str:
    payload = {
        "jobDescription": job_description_text,
        "scoringPromptTemplate": scoring_prompt_template,
        "candidates": [
            {
                "candidateId": str(candidate.id),
                "fullName": candidate.full_name,
                "currentJobTitle": candidate.current_job_title,
                "education": candidate.education_text,
                "experience": candidate.experience_text,
                "skills": candidate.skills_text,
                "summary": candidate.summary_text,
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
        "Return valid JSON only with the shape shown in responseFormat.\n\n"
        f"{json.dumps(payload, ensure_ascii=True)}"
    )
