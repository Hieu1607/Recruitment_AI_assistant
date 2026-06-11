"""Add voice interview domain tables.

Revision ID: 20260522_0006
Revises: 20260509_0005
Create Date: 2026-05-22 00:06:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260522_0006"
down_revision = "20260509_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "interview_templates",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("language_code", sa.String(length=16), nullable=False, server_default="vi-VN"),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="draft"),
        sa.Column("intro_script", sa.Text(), nullable=True),
        sa.Column("closing_script", sa.Text(), nullable=True),
        sa.Column("question_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("report_rubric", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_interview_templates"),
        sa.UniqueConstraint("id", "job_id", name="uq_interview_templates_id_job_id"),
    )
    op.create_index("ix_interview_templates_job_id", "interview_templates", ["job_id"], unique=False)

    op.create_table(
        "interview_invitations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("candidate_profile_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("interview_template_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "public_token",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("md5(random()::text || clock_timestamp()::text)"),
        ),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("sent_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.CheckConstraint("max_attempts > 0", name="ck_interview_invitations_max_attempts_positive"),
        sa.CheckConstraint("attempt_count >= 0", name="ck_interview_invitations_attempt_count_non_negative"),
        sa.CheckConstraint("attempt_count <= max_attempts", name="ck_interview_invitations_attempt_count_within_max"),
        sa.ForeignKeyConstraint(["candidate_profile_id"], ["candidate_profiles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["sent_by_user_id"], ["user_accounts.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["interview_template_id", "job_id"],
            ["interview_templates.id", "interview_templates.job_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_interview_invitations"),
    )
    op.create_index("ix_interview_invitations_candidate_profile_id", "interview_invitations", ["candidate_profile_id"], unique=False)
    op.create_index("ix_interview_invitations_interview_template_id", "interview_invitations", ["interview_template_id"], unique=False)
    op.create_index("ix_interview_invitations_job_id", "interview_invitations", ["job_id"], unique=False)
    op.create_index("ix_interview_invitations_public_token", "interview_invitations", ["public_token"], unique=True)

    op.create_table(
        "interview_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("interview_invitation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("provider", sa.String(length=100), nullable=True),
        sa.Column("provider_session_id", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="created"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("device_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("browser_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("connection_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["interview_invitation_id"], ["interview_invitations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_interview_sessions"),
    )
    op.create_index("ix_interview_sessions_interview_invitation_id", "interview_sessions", ["interview_invitation_id"], unique=False)
    op.create_index("ix_interview_sessions_provider_session_id", "interview_sessions", ["provider_session_id"], unique=False)

    op.create_table(
        "interview_response_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("interview_session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("question_key", sa.String(length=255), nullable=False),
        sa.Column("question_order", sa.Integer(), nullable=True),
        sa.Column("prompt_text", sa.Text(), nullable=True),
        sa.Column("response_text", sa.Text(), nullable=True),
        sa.Column("response_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["interview_session_id"], ["interview_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_interview_response_items"),
        sa.UniqueConstraint("interview_session_id", "question_key", name="uq_interview_response_items_session_question"),
    )
    op.create_index("ix_interview_response_items_interview_session_id", "interview_response_items", ["interview_session_id"], unique=False)

    op.create_table(
        "interview_transcript_turns",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("interview_session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("response_item_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("speaker_role", sa.String(length=50), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("transcript_text", sa.Text(), nullable=False),
        sa.Column("time_offset_ms", sa.Integer(), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["interview_session_id"], ["interview_sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["response_item_id"], ["interview_response_items.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id", name="pk_interview_transcript_turns"),
        sa.UniqueConstraint(
            "interview_session_id",
            "turn_index",
            name="uq_interview_transcript_turns_session_turn_index",
        ),
    )
    op.create_index("ix_interview_transcript_turns_interview_session_id", "interview_transcript_turns", ["interview_session_id"], unique=False)
    op.create_index("ix_interview_transcript_turns_response_item_id", "interview_transcript_turns", ["response_item_id"], unique=False)

    op.create_table(
        "interview_reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("interview_session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("interview_template_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("summary_text", sa.Text(), nullable=True),
        sa.Column("report_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["interview_session_id"], ["interview_sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["interview_template_id"], ["interview_templates.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id", name="pk_interview_reports"),
    )
    op.create_index("ix_interview_reports_interview_session_id", "interview_reports", ["interview_session_id"], unique=True)
    op.create_index("ix_interview_reports_interview_template_id", "interview_reports", ["interview_template_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_interview_reports_interview_template_id", table_name="interview_reports")
    op.drop_index("ix_interview_reports_interview_session_id", table_name="interview_reports")
    op.drop_table("interview_reports")

    op.drop_index("ix_interview_transcript_turns_response_item_id", table_name="interview_transcript_turns")
    op.drop_index("ix_interview_transcript_turns_interview_session_id", table_name="interview_transcript_turns")
    op.drop_table("interview_transcript_turns")

    op.drop_index("ix_interview_response_items_interview_session_id", table_name="interview_response_items")
    op.drop_table("interview_response_items")

    op.drop_index("ix_interview_sessions_provider_session_id", table_name="interview_sessions")
    op.drop_index("ix_interview_sessions_interview_invitation_id", table_name="interview_sessions")
    op.drop_table("interview_sessions")

    op.drop_index("ix_interview_invitations_public_token", table_name="interview_invitations")
    op.drop_index("ix_interview_invitations_job_id", table_name="interview_invitations")
    op.drop_index("ix_interview_invitations_interview_template_id", table_name="interview_invitations")
    op.drop_index("ix_interview_invitations_candidate_profile_id", table_name="interview_invitations")
    op.drop_table("interview_invitations")

    op.drop_index("ix_interview_templates_job_id", table_name="interview_templates")
    op.drop_table("interview_templates")
