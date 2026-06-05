from src.services.email_templates import (
    build_interview_invitation_email,
    build_outreach_email,
)


def test_build_interview_invitation_email_contains_candidate_and_url():
    subject, body = build_interview_invitation_email(
        candidate_name="Candidate One",
        job_title="Platform Engineer",
        public_url="http://localhost:5173/interviews/token",
        expires_at_text="2026-06-08 10:00 UTC",
    )

    assert subject == "Interview invitation for Platform Engineer"
    assert "Hi Candidate One," in body
    assert "http://localhost:5173/interviews/token" in body
    assert "2026-06-08 10:00 UTC" in body


def test_build_outreach_email_trims_subject_and_body():
    subject, body = build_outreach_email(subject=" Hello ", body=" Body ")

    assert subject == "Hello"
    assert body == "Body"
