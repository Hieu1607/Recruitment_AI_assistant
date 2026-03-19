"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-03-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_accounts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("display_name", sa.String(length=120), nullable=False),
        sa.Column("status", sa.Enum("active", "suspended", name="user_status"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_user_accounts_email", "user_accounts", ["email"], unique=True)

    op.create_table(
        "role_assignments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role_name", sa.Enum("admin", "recruiter", "viewer", name="role_name"), nullable=False),
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["user_accounts.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("user_id", "role_name", name="uq_role_assignments_user_role"),
    )

    op.create_table(
        "resume_documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("original_file_name", sa.String(length=255), nullable=False),
        sa.Column("storage_provider", sa.String(length=32), nullable=False),
        sa.Column("minio_bucket", sa.String(length=128), nullable=False),
        sa.Column("minio_object_key", sa.String(length=512), nullable=False),
        sa.Column("storage_uri", sa.String(length=1024), nullable=False),
        sa.Column("mime_type", sa.String(length=100), nullable=False),
        sa.Column(
            "upload_status",
            sa.Enum("uploaded", "processing", "processed", "failed", name="upload_status"),
            nullable=False,
        ),
        sa.Column(
            "parse_status",
            sa.Enum("not_started", "text_extracted", "normalized", "failed", name="parse_status"),
            nullable=False,
        ),
        sa.Column("language_detected", sa.String(length=10), nullable=True),
        sa.Column("duplicate_group_key", sa.String(length=120), nullable=True),
        sa.Column("uploaded_by_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retention_expires_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "candidate_profiles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("resume_document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=False),
        sa.Column("phone", sa.String(length=50), nullable=True),
        sa.Column("email", sa.String(length=320), nullable=True),
        sa.Column("location_normalized", sa.String(length=255), nullable=True),
        sa.Column("contact", sa.String(length=255), nullable=True),
        sa.Column("current_job_title", sa.String(length=255), nullable=True),
        sa.Column("educated", sa.Boolean(), nullable=False),
        sa.Column("ever_studied_abroad", sa.Boolean(), nullable=False),
        sa.Column("major", sa.String(length=255), nullable=True),
        sa.Column("cpa", sa.String(length=255), nullable=True),
        sa.Column("education_text", sa.Text(), nullable=True),
        sa.Column("experience_text", sa.Text(), nullable=True),
        sa.Column("experience_years", sa.Numeric(4, 1), nullable=True),
        sa.Column("skills_text", sa.Text(), nullable=True),
        sa.Column("languages_text", sa.Text(), nullable=True),
        sa.Column("projects_text", sa.Text(), nullable=True),
        sa.Column("summary_text", sa.Text(), nullable=True),
        sa.Column("achievements_text", sa.Text(), nullable=True),
        sa.Column("publications_text", sa.Text(), nullable=True),
        sa.Column("certifications_text", sa.Text(), nullable=True),
        sa.Column("references_text", sa.Text(), nullable=True),
        sa.Column("other_text", sa.Text(), nullable=True),
        sa.Column(
            "profile_status",
            sa.Enum("draft", "reviewed", "approved", "archived", name="profile_status"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["resume_document_id"], ["resume_documents.id"], ondelete="CASCADE"),
    )

    op.create_table(
        "job_descriptions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("jd_text", sa.Text(), nullable=False),
        sa.Column("created_by_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
    )

    op.create_table(
        "match_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("job_description_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("initiated_by_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("scoring_prompt_template", sa.Text(), nullable=False),
        sa.Column("score_threshold", sa.Numeric(5, 2), nullable=False),
        sa.Column(
            "run_status",
            sa.Enum("queued", "running", "completed", "failed", name="match_run_status"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["job_description_id"], ["job_descriptions.id"], ondelete="CASCADE"),
    )

    op.create_table(
        "query_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_title", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "query_turns",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("query_session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_question", sa.Text(), nullable=False),
        sa.Column(
            "routing_strategy",
            sa.Enum("sql_only", "llm_only", "hybrid", name="routing_strategy"),
            nullable=False,
        ),
        sa.Column("answer_text", sa.Text(), nullable=False),
        sa.Column("matched_candidate_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("matched_count", sa.Integer(), nullable=True),
        sa.Column("tool_trace_masked", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["query_session_id"], ["query_sessions.id"], ondelete="CASCADE"),
    )

    op.create_table(
        "extraction_traces",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("resume_document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("candidate_profile_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("source_page", sa.Integer(), nullable=False),
        sa.Column("source_bbox", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("source_text_snippet", sa.Text(), nullable=False),
        sa.Column("mapped_field_name", sa.String(length=120), nullable=False),
        sa.Column("confidence_score", sa.Numeric(5, 2), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["resume_document_id"], ["resume_documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["candidate_profile_id"], ["candidate_profiles.id"], ondelete="SET NULL"),
    )

    op.create_table(
        "match_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("match_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("candidate_profile_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("score_list_index", sa.Integer(), nullable=False),
        sa.Column("total_score", sa.Numeric(5, 2), nullable=False),
        sa.Column("passed_threshold", sa.Boolean(), nullable=False),
        sa.Column("rationale_summary", sa.Text(), nullable=False),
        sa.Column("confidence_level", sa.Numeric(5, 2), nullable=True),
        sa.Column("component_scores", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["match_run_id"], ["match_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["candidate_profile_id"], ["candidate_profiles.id"], ondelete="CASCADE"),
    )

    op.create_table(
        "shortlist_collections",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("name", sa.String(length=150), nullable=False),
        sa.Column("created_by_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_query_turn_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["source_query_turn_id"], ["query_turns.id"], ondelete="SET NULL"),
    )

    op.create_table(
        "shortlist_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("shortlist_collection_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("candidate_profile_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("added_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["shortlist_collection_id"], ["shortlist_collections.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["candidate_profile_id"], ["candidate_profiles.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "shortlist_collection_id",
            "candidate_profile_id",
            name="uq_shortlist_item_collection_candidate",
        ),
    )

    op.create_table(
        "outreach_messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("candidate_profile_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_by_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("content_source", sa.Enum("ai_draft", "template", name="content_source"), nullable=False),
        sa.Column("subject", sa.String(length=255), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column(
            "approval_status",
            sa.Enum("draft", "approved", "rejected", name="approval_status"),
            nullable=False,
        ),
        sa.Column(
            "sent_status",
            sa.Enum("not_sent", "sent", "failed", name="sent_status"),
            nullable=False,
        ),
        sa.Column("approved_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["candidate_profile_id"], ["candidate_profiles.id"], ondelete="CASCADE"),
    )

    op.create_table(
        "interview_question_sets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("candidate_profile_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_description_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("generated_by_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("question_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["candidate_profile_id"], ["candidate_profiles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["job_description_id"], ["job_descriptions.id"], ondelete="CASCADE"),
    )


def downgrade() -> None:
    op.drop_table("interview_question_sets")
    op.drop_table("outreach_messages")
    op.drop_table("shortlist_items")
    op.drop_table("shortlist_collections")
    op.drop_table("match_results")
    op.drop_table("extraction_traces")
    op.drop_table("query_turns")
    op.drop_table("query_sessions")
    op.drop_table("match_runs")
    op.drop_table("job_descriptions")
    op.drop_table("candidate_profiles")
    op.drop_table("resume_documents")
    op.drop_table("role_assignments")
    op.drop_table("user_accounts")

    sa.Enum(name="sent_status").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="approval_status").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="content_source").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="routing_strategy").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="match_run_status").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="profile_status").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="parse_status").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="upload_status").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="role_name").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="user_status").drop(op.get_bind(), checkfirst=True)
