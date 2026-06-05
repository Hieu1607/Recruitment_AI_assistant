import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

sys.modules.pop("worker", None)
sys.modules.pop("worker.tasks", None)
tasks = importlib.import_module("worker.tasks")


def test_send_interview_invitation_email_marks_sent_after_success(monkeypatch):
    invitation = SimpleNamespace(
        id="invitation-id",
        sent_at=None,
        status="pending",
        candidate_profile=SimpleNamespace(email="candidate@example.com", full_name="Candidate One"),
        job=SimpleNamespace(title="Platform Engineer"),
        sent_by_user_id="user-id",
        public_token="public-token",
        expires_at=None,
    )
    user = SimpleNamespace(id="user-id", email="recruiter@example.com")
    identity = SimpleNamespace(refresh_token_encrypted="encrypted-refresh")
    committed = {"value": False}

    class FakeDb:
        def get(self, model, key):
            name = getattr(model, "__name__", "")
            if name == "InterviewInvitation":
                return invitation
            if name == "UserAccount":
                return user
            return None

        def execute(self, statement):
            return SimpleNamespace(scalar_one_or_none=lambda: identity)

        def commit(self):
            committed["value"] = True

        def rollback(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr("src.models.session.SessionLocal", lambda: FakeDb())
    monkeypatch.setattr(
        "src.services.interview_invitation_service.build_interview_public_url",
        lambda token: f"http://test/interviews/{token}",
    )
    monkeypatch.setattr("src.services.mail_service.send_email", lambda **kwargs: {"id": "gmail-id"})

    result = tasks.send_interview_invitation_email.run("invitation-id")

    assert result == {"sent": True, "gmail_message_id": "gmail-id"}
    assert invitation.sent_at is not None
    assert committed["value"] is True
