import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.ai_agent.nodes import _resolve_candidates  # noqa: E402


def test_resolve_candidates_uses_scoped_current_candidates(monkeypatch):
    monkeypatch.setattr(
        "src.services.ai_agent.nodes._fetch_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("DB fetch should not run when current_candidates is provided")
        ),
    )

    state = {
        "current_candidates": [
            {
                "id": "candidate-1",
                "full_name": "Scoped One",
                "email": "one@example.com",
                "experience_years": "3.5",
                "skills_text": "Python",
            },
            {
                "id": "candidate-2",
                "full_name": "Scoped Two",
                "email": "two@example.com",
                "experience_years": "7",
                "skills_text": "Go",
            },
        ]
    }

    rows = _resolve_candidates(
        state,
        fields=["email", "experience_years"],
        candidate_ids=["candidate-2"],
    )

    assert rows == [
        {
            "id": "candidate-2",
            "full_name": "Scoped Two",
            "email": "two@example.com",
            "experience_years": 7.0,
        }
    ]


def test_resolve_candidates_falls_back_to_db_fetch_without_scope(monkeypatch):
    monkeypatch.setattr(
        "src.services.ai_agent.nodes._fetch_candidates",
        lambda fields, candidate_ids=None: [
            {"id": "db-candidate", "full_name": "DB Candidate"}
        ],
    )

    rows = _resolve_candidates({}, fields=["skills_text"], candidate_ids=["db-candidate"])

    assert rows == [{"id": "db-candidate", "full_name": "DB Candidate"}]
