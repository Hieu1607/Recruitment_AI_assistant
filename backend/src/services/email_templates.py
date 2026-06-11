from __future__ import annotations

from src.services.outreach_service import normalize_rich_message


def build_interview_invitation_email(
    *,
    candidate_name: str | None,
    job_title: str,
    public_url: str,
    expires_at_text: str | None,
) -> tuple[str, str]:
    display_name = candidate_name or "there"
    subject = f"Interview invitation for {job_title}"
    expiry_line = f"\nThis link is available until {expires_at_text}." if expires_at_text else ""
    body = (
        f"Hi {display_name},\n\n"
        f"Thank you for your interest in the {job_title} role. "
        "We would like to invite you to complete a short voice interview.\n\n"
        f"Interview link: {public_url}"
        f"{expiry_line}\n\n"
        "Best regards,\n"
        "Recruitment Team"
    )
    return subject, body


def build_outreach_email(
    *,
    subject: str,
    body_text: str | None,
    body_html: str | None,
) -> tuple[str, str, str]:
    normalized_text, normalized_html = normalize_rich_message(
        body_text=body_text,
        body_html=body_html,
    )
    return subject.strip(), normalized_text, normalized_html
